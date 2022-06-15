"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/data/CIFAR/cifar.py
"""
import os
import torch as t
import torchvision
from torchvision import transforms
import numpy as np
from ..base import Dataset, DatasetFromTorch, load_all, StackedDataset

__all__ = ('CIFAR10', 'CIFAR10Augmented', 'CIFAR10DataAugmentPrior')


class CIFAR10:
    """
    The usage is:
    ```
    cifar10 = CIFAR10()
    ```
    e.g. normalized training dataset:
    ```
    cifar10.norm.train
    ```
    """

    def __init__(self,
                 dtype='float32',
                 device="cpu",
                 download=False,
                 data_subset_size: int = 50000):
        n_classes = 10
        train_data_size = 50000
        assert train_data_size % data_subset_size == 0, \
            "`data_subset_size` must divide into original training data set exactly. " \
            f"Training set has {train_data_size} examples, and `data_subset_size`={data_subset_size}."
        assert data_subset_size % n_classes == 0, \
            f"`data_subset_size` must be multiple of number of classes ({n_classes})."
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        self.dtype = dtype
        self.device = device

        # load data
        data_train = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=False)

        if data_subset_size < train_data_size:
            data_train = get_data_subset(data_train, subset_size=data_subset_size, keep_original_size=False)

        self._save_datasets(data_train.data, data_test.data, data_train.targets, data_test.targets)

    def _save_datasets(self, inputs_train, inputs_test, labels_train, labels_test, permutation=(0, 3, 1, 2)):
        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([inputs_train, inputs_test]).astype(self.dtype)).permute(permutation)
        y = t.from_numpy(np.concatenate([labels_train, labels_test]).astype('int'))
        # alternative version to yield one-hot vectors
        # y = t.from_numpy(np.eye(10)[np.concatenate([data_train.targets, data_test.targets])].astype(dtype))

        # train / test split
        index_train = np.arange(len(inputs_train))
        index_test = np.arange(len(inputs_train), len(inputs_train) + len(inputs_test))

        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, self.device)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, (0, 2, 3), keepdims=True)
        self.X_mean = t.mean(self.unnorm.train_X, (0, 2, 3), keepdims=True)

        # create normalized data set
        X_norm = (self.unnorm.X - self.X_mean) / self.X_std
        self.norm = Dataset(X_norm, y, index_train, index_test, self.device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]


class CIFAR10Augmented:
    def __init__(self,
                 dtype='float32',
                 device="cpu",
                 download=False,
                 padding: int = 4,
                 crop_size: int = 32,
                 do_hflip: bool = True,
                 max_rotation: float = 0.,
                 crop_before_norm: bool = True,
                 data_subset_size: int = 50000):
        n_classes = 10
        train_data_size = 50000
        check_data_subset_size(data_subset_size, train_data_size, n_classes)

        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        dtype = getattr(t, dtype)
        self.dtype = dtype
        self.device = device

        unnorm_train = torchvision.datasets.CIFAR10(
            dataset_dir, download=download, train=True, transform=transforms.ToTensor())
        unnorm_x, _ = load_all(unnorm_train)
        X_mean = unnorm_x.mean(dim=(0, 2, 3), keepdims=True)
        X_std = unnorm_x.std(dim=(0, 2, 3), keepdims=True)
        self.X_mean = X_mean
        self.X_std = X_std

        X_mean_tuple = tuple(a.item() for a in X_mean.view(-1))
        X_std_tuple = tuple(a.item() for a in X_std.view(-1))

        if crop_before_norm:
            transform_train = [
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform(),
                transforms.ToTensor(),
                transforms.Normalize(X_mean_tuple, X_std_tuple)]
        else:
            transform_train = [
                transforms.ToTensor(),
                transforms.Normalize(X_mean_tuple, X_std_tuple),
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform(),
            ]
        if max_rotation > 0.:
            transform_train.append(transforms.RandomRotation(degrees=max_rotation))
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(X_mean_tuple, X_std_tuple),
        ])
        data_train = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=True, transform=transform_train)
        data_test = torchvision.datasets.CIFAR10(dataset_dir, download=download, train=False, transform=transform_test)

        if data_subset_size < train_data_size:
            data_train = get_data_subset(data_train, subset_size=data_subset_size, keep_original_size=False)
        self.norm = DatasetFromTorch(data_train, data_test, device=device)
        self.num_train_set = len(data_train)
        self.in_shape = t.Size([3, crop_size, crop_size])
        self.out_shape = t.Size([])


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


class CIFAR10DataAugmentPrior:
    """With data augmentation, such that `n_mc_augment` samples are drawn
    for each input image. Same augmentation transforms as in CIFAR10Augmented."""
    # TODO (Seth): refactor so that less code duplicated between
    #  CIFAR10Augmented and CIFAR10AugmentPrior
    for_data_augment_prior = True

    def __init__(self,
                 n_mc_augment: int,
                 n_subsample: [int, None] = None,
                 dtype='float32',
                 device="cpu",
                 download=False,
                 seed=None,
                 test_time_aug=False,
                 n_mc_aug_test: [int, None] = None,
                 n_subsample_test: [int, None] = None,
                 padding: int = 4,
                 crop_size: int = 32,
                 max_rotation: float = 0.,
                 do_hflip: bool = True,
                 crop_before_norm: bool = True,
                 save: bool = False,
                 load: bool = False,
                 data_subset_size: int = 50000,
                 root_dir=None):
        n_classes = 10
        train_data_size = 50000
        assert not (save and load)
        check_data_subset_size(data_subset_size, train_data_size, n_classes)

        _ROOT = root_dir or os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/cifar10/'
        dtype = getattr(t, dtype)
        self.dtype = dtype
        self.device = device

        unnorm_train = torchvision.datasets.CIFAR10(
            dataset_dir, download=download, train=True, transform=transforms.ToTensor())
        unnorm_x, _ = load_all(unnorm_train)
        X_mean = unnorm_x.mean(dim=(0, 2, 3), keepdims=True)
        X_std = unnorm_x.std(dim=(0, 2, 3), keepdims=True)
        self.X_mean = X_mean
        self.X_std = X_std
        X_mean_tuple = tuple(a.item() for a in X_mean.view(-1))
        X_std_tuple = tuple(a.item() for a in X_std.view(-1))

        if crop_before_norm:
            transform_train = [
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform(),
                transforms.ToTensor(),
                transforms.Normalize(X_mean_tuple, X_std_tuple)]
        else:
            transform_train = [
                transforms.ToTensor(),
                transforms.Normalize(X_mean_tuple, X_std_tuple),
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform()]
        if max_rotation > 0.:
            transform_train.extend([transforms.ToPILImage(),
                                    transforms.RandomRotation(degrees=max_rotation),
                                    transforms.ToTensor()])
        transform_train = transforms.Compose(transform_train)
        saveload_path_train = self.get_save_path(dataset_dir,
                                                 n_mc_augment=n_mc_augment,
                                                 seed=seed,
                                                 padding=padding,
                                                 crop_size=crop_size,
                                                 do_hflip=do_hflip,
                                                 max_rotation=max_rotation,
                                                 train=True)

        data_train = torchvision.datasets.CIFAR10(dataset_dir,
                                                  download=download,
                                                  train=True,
                                                  transform=transform_train)
        if data_subset_size < train_data_size:
            data_train = get_data_subset(data_train, subset_size=data_subset_size, keep_original_size=False)
        data_train_stacked = StackedDataset(dataset_to_stack=data_train,
                                            n_samples=n_mc_augment,
                                            n_subsample=n_subsample,
                                            base_seed=seed,
                                            save_path=saveload_path_train if save else None,
                                            load_path=saveload_path_train if load else None)

        if test_time_aug:
            assert n_mc_aug_test is not None, \
                "Must provide number of augmentation samples " \
                "to do at test time."
            transform_test = transform_train
        else:
            n_mc_aug_test = 1
            transform_test = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(X_mean_tuple, X_std_tuple),
            ])
        data_test = torchvision.datasets.CIFAR10(dataset_dir,
                                                 download=download,
                                                 train=False,
                                                 transform=transform_test)
        saveload_path_test = self.get_save_path(root_dir=dataset_dir,
                                                n_mc_augment=n_mc_aug_test if test_time_aug else 1,
                                                seed=seed,
                                                padding=padding if test_time_aug else 0,
                                                crop_size=crop_size if test_time_aug else 32,
                                                do_hflip=do_hflip if test_time_aug else False,
                                                max_rotation=max_rotation if test_time_aug else 0.,
                                                train=False)
        data_test_stacked = StackedDataset(dataset_to_stack=data_test,
                                           n_samples=n_mc_aug_test,
                                           n_subsample=n_subsample_test,
                                           base_seed=seed,
                                           save_path=saveload_path_test if save and test_time_aug else None,
                                           load_path=saveload_path_test if load and test_time_aug else None)

        # 8 is largest data augmentation axis where all pre-augmented data can fit on GPU
        all_data_on_device = n_mc_augment == 1
        self.norm = DatasetFromTorch(data_train_stacked,
                                     data_test_stacked,
                                     device=device,
                                     all_on_device=all_data_on_device)
        self.num_train_set = len(data_train)
        self.in_shape = t.Size([3, crop_size, crop_size])
        self.out_shape = t.Size([])

    def get_save_path(self,
                      root_dir,
                      n_mc_augment,
                      seed, padding,
                      crop_size,
                      do_hflip,
                      max_rotation,
                      train=True):
        filename = f"{n_mc_augment}samples_{padding}pad_{crop_size}cropsize" + \
                   (f"_rotation{max_rotation}" if max_rotation > 0. else "") + \
                   f"{'_hflip' if do_hflip else ''}_seed{seed}_{'train' if train else 'test'}.npy"
        return os.path.join(root_dir, "augmented", filename)


def get_data_subset(dataset, subset_size, n_class=10, keep_original_size=False):
    """
    Subselects examples from the dataset, keeping classes balanced
    :param keep_original_size: bool, if True will make copies of subsampled dataset
    so overall number of examples is unchanged.
    """
    subset_size_per_class = int(subset_size / n_class)
    n_original_examples = len(dataset.targets)

    subset_ids = []
    for clss in range(n_class):
        where_class = np.where(np.equal(dataset.targets, clss))[0]
        where_class_sub = where_class[:subset_size_per_class]
        subset_ids += where_class_sub.tolist()

    subset_ids = np.sort(np.array(subset_ids).astype(int))
    if keep_original_size:
        n_copies = int(n_original_examples / len(subset_ids))
        subset_ids = np.tile(subset_ids, n_copies)
        assert len(subset_ids) == n_original_examples
    dataset.data = dataset.data[subset_ids]
    dataset.targets = np.array(dataset.targets)[subset_ids].tolist()
    return dataset


def check_data_subset_size(data_subset_size, train_data_size, n_classes):
    assert data_subset_size <= train_data_size, \
        "Selecting `data_subset_size` from original dataset. " \
        f"Must be less than original dataset size ({train_data_size})."
    # assert train_data_size % data_subset_size == 0, \
    #     "`data_subset_size` must divide into original training data set exactly. " \
    #     f"Training set has {train_data_size} examples, and `data_subset_size`={data_subset_size}."
    if data_subset_size < train_data_size:
        assert data_subset_size % n_classes == 0, \
            f"`data_subset_size` must be multiple of number of classes ({n_classes}). " \
            f"Got `data_subset_size`={data_subset_size}."
