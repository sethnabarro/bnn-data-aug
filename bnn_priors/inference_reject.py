"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/inference_reject.py
"""
import tqdm
import torch
from .utils import get_cosine_schedule
from . import mcmc
import math
from .exp_utils import evaluate_model

from .inference import SGLDRunner


class VerletSGLDRunnerReject(SGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.VerletSGLD(
            params=params,
            lr=self.learning_rate, num_data=self.eff_num_data,
            momentum=self.momentum, temperature=self.temperature)

    def _exact_model_potential_and_grad(self, dataloader):
        self.optimizer.zero_grad()
        log_prior = self.model.log_prior()
        log_norm_prior = log_prior / -self.eff_num_data
        log_norm_prior.backward()

        loss = 0.
        for x, y in dataloader:
            x, y = x.to(self._params[0].device), y.to(self._params[0].device)
            this_loss = self.model.log_likelihood(x,
                                                  y,
                                                  -x.size(0)/self.eff_num_data)
            this_loss.backward()
            loss = loss + this_loss

        potential = loss + log_norm_prior
        return loss, log_prior, potential

    def run(self, progressbar=False):
        self.optimizer = self._make_optimizer(self._params)
        self.scheduler = self._make_scheduler(self.optimizer)

        if progressbar:
            progressbar = tqdm.tqdm(total=self.cycles*self.epochs_per_cycle, mininterval=2.0)
        assert progressbar is False or isinstance(progressbar, tqdm.std.tqdm)

        def _enter_epoch(desc, temperature):
            "Run this at the beginning of each epoch"
            if progressbar:
                progressbar.set_description(desc, refresh=False)
            for g in self.optimizer.param_groups:
                g['temperature'] = temperature

        def _is_sampling_epoch(_epoch):
            "Are we storing a sample at the end of this epoch?"
            _epoch = _epoch % self.epochs_per_cycle
            sampling_epoch = _epoch - (self.descent_epochs + self.warmup_epochs)
            return (0 <= sampling_epoch) and (sampling_epoch % self.skip == 0)

        # Use an exact gradient for the initial step and loss
        loss, log_prior, potential = self._exact_model_potential_and_grad(self.dataloader)
        self.optimizer.sample_momentum()
        self.optimizer.initial_step(calc_metrics=True, save_state=self.reject_samples)
        step = 0
        self.store_metrics(i=step, loss=loss.item(), log_prior=log_prior.item(),
                           potential=potential.item(), acc=0.,
                           lr=self.optimizer.param_groups[0]["lr"], corresponds_to_sample=True,
                           delta_energy=0., total_energy=0., rejected=False)
        self._initial_potential = potential.item()
        self._total_energy = 0.

        assert self.dataloader.sampler.generator is None
        generator = self.dataloader.sampler.generator = torch.Generator()
        postfix = {}
        for cycle in range(self.cycles):
            generator.seed()
            cycle_random_state = generator.get_state()
            for epoch in range(self.epochs_per_cycle):
                if epoch < self.descent_epochs:
                    _enter_epoch(f"Cycle {cycle}, epoch {epoch}, Descent", 0.)
                elif epoch - self.descent_epochs < self.warmup_epochs:
                    _enter_epoch(f"Cycle {cycle}, epoch {epoch}, Warmup", self.temperature)
                else:
                    _enter_epoch(f"Cycle {cycle}, epoch {epoch}, Sampling", self.temperature)

                # Run one epoch of potentially-stochastic gradient descent
                # make sure the epochs' data points are always in the same order for this cycle.
                generator.set_state(cycle_random_state)

                for i, (x, y) in enumerate(self.dataloader):
                    x, y = x.to(self._params[0].device), y.to(self._params[0].device)
                    step += 1
                    loss, log_prior, potential, acc = \
                        self._model_potential_and_grad(x, y)
                    store_metrics = (step % self.metrics_skip) == 0
                    self.optimizer.step(calc_metrics=store_metrics)

                    if store_metrics:
                        delta_energy = self.optimizer.delta_energy(self._initial_potential, potential)
                        self.store_metrics(i=step,
                                           loss=loss.item(),
                                           log_prior=log_prior.item(),
                                           potential=potential.item(),
                                           acc=acc.item(),
                                           lr=self.optimizer.param_groups[0]["lr"],
                                           corresponds_to_sample=False,
                                           delta_energy=delta_energy,
                                           total_energy=self._total_energy+delta_energy)
                        if progressbar:
                            postfix["train/loss"] = loss.item()
                            postfix["train/acc"] = acc.item()
                            postfix["Δₑ"] = delta_energy
                            progressbar.set_postfix(postfix, refresh=False)

                    # Omit the scheduler step in the last iteration, because we
                    # want to run it after `optimizer.final_step`
                    if i < len(self.dataloader) - 1:
                        self.scheduler.step()


                if _is_sampling_epoch(epoch):
                    step += 1
                    # Do the sample's `final_step` using an exact gradient
                    loss, log_prior, potential = self._exact_model_potential_and_grad(self.dataloader)
                    self.optimizer.final_step(calc_metrics=True)
                    delta_energy = self.optimizer.delta_energy(self._initial_potential, potential)
                    self._total_energy += delta_energy
                    self._initial_potential = potential.item()

                    rejected = False
                    if self.reject_samples:
                        rejected, _ = self.optimizer.maybe_reject(delta_energy)
                    self.store_metrics(i=step,
                                       loss=loss.item(),
                                       log_prior=log_prior.item(),
                                       potential=potential.item(),
                                       # TODO: do not use stale `acc`, calculate for full training set
                                       acc=acc.item(),
                                       lr=self.optimizer.param_groups[0]["lr"],
                                       corresponds_to_sample=True,
                                       delta_energy=delta_energy,
                                       total_energy=self._total_energy,
                                       rejected=rejected)

                    # Evaluate test accuracy and save to disk the current sample
                    # (correctly rolled back to the previous if rejected)
                    state_dict = self.model.state_dict()
                    self._save_sample(state_dict, cycle, epoch, step)
                    if epoch % self.epochs_per_eval == 0:
                        eval_results = self._evaluate_model(state_dict, step)
                        if progressbar:
                            postfix.update(eval_results)
                            postfix["train/loss"] = loss.item()
                            postfix["Δₑ"] = delta_energy
                            progressbar.set_postfix(postfix, refresh=False)
                    self.scheduler.step()

                    self.optimizer.initial_step(
                        calc_metrics=False, save_state=self.reject_samples)

                else:  # Not an epoch that stores a sample at the end
                    # Evaluate test accuracy every epoch
                    if epoch % self.epochs_per_eval == 0:
                        eval_results = self._evaluate_model(self.model.state_dict(), step)
                        if progressbar:
                            postfix.update(eval_results)
                            progressbar.set_postfix(postfix, refresh=False)
                    self.scheduler.step()

                # Update preconditioner, increment progressbar at the end of the epoch
                if self.precond_update is not None and (epoch+1) % self.precond_update == 0:
                    self.optimizer.update_preconditioner()

                # Important to put here because no new metrics are added
                # Write metrics to disk every 30 seconds
                self.metrics_saver.flush(every_s=30)

                if progressbar:
                    progressbar.update(1)
        # Close the progressbar at the end of the training procedure
        if progressbar:
            progressbar.close()
