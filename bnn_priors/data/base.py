"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/data/base.py
"""
import datetime as dt
import json
import numpy as np
import os
import random
import torch as t
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


__all__ = ('Dataset', 'DatasetFromTorch', 'ReshapeTransform')


class Dataset:
    """
    Represents the full dataset.  We will have two copies: one normalised, one unnormalized.
    """
    def __init__(self, X, y, index_train, index_test, device="cpu"):
        self.X = X.to(device)
        self.y = y.to(device)

        self.train_X = self.X[index_train]
        self.train_y = self.y[index_train]
        self.test_X  = self.X[index_test]
        self.test_y  = self.y[index_test]

        self.train = TensorDataset(self.train_X, self.train_y)
        self.test  = TensorDataset(self.test_X,  self.test_y)


def load_batch(dset, batchsize=16):
    loader = DataLoader(dset, batch_size=batchsize, shuffle=False)
    return next(iter(loader))


def load_all(dset):
    return load_batch(dset, batchsize=len(dset))


class DatasetFromTorch(Dataset):
    def __init__(self, train, test, device, all_on_device=True):
        self.train = train
        self.test = test

        # load_all() can consume too much memory in cases
        # with large data augmentation axis
        load_fn = load_all if all_on_device else load_batch
        self.train_X, self.train_y = (a.to(device) for a in load_fn(train))
        self.test_X, self.test_y = (a.to(device) for a in load_fn(test))


class ReshapeTransform:
    """
    From
    https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/7
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return t.reshape(img, self.new_size)


class StackedDataset(Dataset):
    """Same as `dataset_to_stack` given as input,
    but each item copied `n_samples` times in `__getitem__()`
    method, stacked along new axis. Each copy has independent
    data augmentation.

    The series of augmentations can be fixed by supplying `base_seed`.
    """
    # If transforms are fixed (seeded), will do the augmentation upfront,
    # so long as n_repeats <= MAX_SAMPLES_UPFRONT
    # Otherwise memory required to store augmented dataset will be prohibitive
    MAX_SAMPLES_UPFRONT = 16
    def __init__(self,
                 dataset_to_stack: Dataset,
                 n_samples: int,
                 n_subsample: [int, None] = None,
                 base_seed: int = None,
                 save_path: str = None,
                 load_path: str = None
                 ) -> None:
        if save_path or load_path:
            assert base_seed is not None, \
                "Can only save/load augmented datasets with fixed transformations. " \
                "Need to provide seed."
            assert n_samples <= self.MAX_SAMPLES_UPFRONT, \
                f"Too many samples ({n_samples} > {self.MAX_SAMPLES_UPFRONT}) " \
                f"to save/load full augmented dataset"
        if n_subsample:
            assert base_seed is not None, \
                "Subsampling only supported when using fixed transforms (with seed)"
        self.transform = dataset_to_stack.transform
        self.target_transform = dataset_to_stack.target_transform
        self.data = dataset_to_stack.data
        self.targets = dataset_to_stack.targets

        self.n_repeats = n_samples
        self.n_subsample = None   # Updated after preloading below
        self.base_seed = base_seed
        self.transform_on_fly = True
        if self.base_seed is not None and \
                n_samples <= self.MAX_SAMPLES_UPFRONT:
            # Does all the augmentation up front
            # NOTE: when doing augmentation, we need self.transform_on_fly==True and
            # self.n_subsample==None, then when done, set them to correct values:
            # self.transform_on_fly=False, self.n_subsample=n_subsample
            if load_path:
                self.load_augmented_dataset(load_path)
            else:
                # Pre-compute the augmentations
                new_data = []
                for i in range(len(self.data)):
                    new_data.append(self[i][0])
                self.data = t.stack(new_data)
                if save_path:
                    self.save_augmented_dataset(save_path)
            self.transform_on_fly = False
        self.n_subsample = n_subsample

    def load_augmented_dataset(self, load_path):
        with open(f"{load_path}.meta", 'r') as metain:
            self.check_meta(json.load(metain))
        print(f"Loading {load_path}")
        self.data = np.load(load_path)
        print("Done.")

    def save_augmented_dataset(self, save_path):
        print(f"Saving {save_path}")
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(save_path, self.data.numpy())
        print("Done.")
        with open(f"{save_path}.meta", 'w') as metaout:
            meta_dict = self.get_meta()
            meta_dict['saved_at_utc'] = str(dt.datetime.utcnow())
            json.dump(meta_dict, metaout)

    def get_meta(self):
        transform_str = str(self.transform)
        return {'seed': self.base_seed,
                'transform': self._remove_none_transform_memadd(transform_str),
                'n_samples': self.n_repeats}

    def _remove_none_transform_memadd(self, tform_str):
        if 'NoneTransform' in tform_str:
            nonetform_str = tform_str.split('<')[1].split('>')[0]
            assert 'NoneTransform' in nonetform_str
            return tform_str.split('<')[0] + 'NoneTransform' + tform_str.split('>')[1]
        else:
            return tform_str

    def check_meta(self, meta_load):
        meta_self = self.get_meta()
        # meta_load['transform'] = self._remove_none_transform_memadd(meta_load['transform'])
        assert all([meta_self[m] == meta_load[m] for m in meta_self]), \
            "Loaded dataset is different config:\n\n" \
            f"Loaded: {meta_load}\n\nCurrent: {meta_self}"

    def apply_transforms(self, img):
        def _set_tform_seed(seed):
            "https://github.com/pytorch/pytorch/issues/42331"
            t.manual_seed(seed)
            random.seed(seed)

        # Stochastic transforms don't vectorise!
        imgs_tform = []
        if isinstance(self.n_subsample, int):
            tform_ids = t.randint(low=0,
                                  high=self.n_repeats,
                                  size=(self.n_subsample,))
        else:
            tform_ids = range(self.n_repeats)
        for tform_id in tform_ids:
            if self.base_seed is not None:
                # Enforce same sequence of tforms for each input image
                _set_tform_seed(self.base_seed + tform_id)
            imgs_tform.append(self.transform(img))

        # Combine all tformed images into tensor
        return t.stack(imgs_tform)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (images, target) where target is index of the target class and
                `images` has shape (n_repeats, C, H, W)

        """
        img, target = self.data[index], self.targets[index]

        if not self.transform_on_fly:
            return self.subsample(img), target

        if self.transform is not None and self.transform_on_fly:
            img = self.apply_transforms(img=transforms.ToPILImage()(img))
        # else:
        #     # imgs_tensor = transforms.ToTensor()(img)
        #     imgs_rep = imgs_tensor.unsqueeze(0).expand([self.n_repeats, 3, 32, 32])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def subsample(self, imgs):
        if isinstance(self.n_subsample, int):
            assert imgs.shape[0] == self.n_repeats, \
                "Number of images in stack should be equal to self.n_repeats. " \
                f"Got {imgs.shape[0]} images in stack and self.n_repeats={self.n_repeats}"
            subsamp_idx = t.randint(low=0,
                                    high=imgs.shape[0],
                                    size=(self.n_subsample,))
            if self.n_subsample == 1:
                # TODO (Seth): Hack - fix!
                subsamp_idx = [subsamp_idx]
            return imgs[subsamp_idx]
        else:
            return imgs

    def __len__(self):
        return len(self.data)


class NoneTransform(object):
    """
    Does nothing to the image
    See https://discuss.pytorch.org/t/
    passing-none-to-transform-compose-causes-a-problem/22602

    Args:
        image in, image out, nothing is done
    """

    def __call__(self, image):
        return image
