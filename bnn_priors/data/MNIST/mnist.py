"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/data/MNIST/mnist.py
"""
import os
import torch as t
import torchvision
from torchvision import transforms
import numpy as np
from scipy import ndimage
from bnn_priors.data import Dataset, DatasetFromTorch
from bnn_priors.data.base import load_all, StackedDataset, NoneTransform

__all__ = ('MNIST', 'RotatedMNIST', 'MNISTDataAugmentPrior', 'FashionMNIST', 'FashionMNISTDataAugmentPrior')


class MNIST:
    """
    The usage is:
    ```
    mnist = MNIST()
    ```
    e.g. normalized training dataset:
    ```
    mnist.norm.train
    ```
    """
    dataset_name = 'mnist'
    torch_dataset_fn = torchvision.datasets.MNIST

    def __init__(self, dtype='float32', device="cpu", download=False, root_dir=None):
        _ROOT = root_dir or os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/{self.dataset_name}/'

        # load data
        data_train_unnorm = self.torch_dataset_fn(dataset_dir, download=download,
                                                  train=True, transform=transforms.ToTensor())

        unnorm_x, unnorm_y = load_all(data_train_unnorm)
        X_mean = unnorm_x.mean(dim=(0, 2, 3), keepdims=True).to(device)
        X_std = unnorm_x.std(dim=(0, 2, 3), keepdims=True).to(device)
        self.X_mean = X_mean
        self.X_std = X_std
        X_mean_tuple = tuple(a.item() for a in X_mean.view(-1))
        X_std_tuple = tuple(a.item() for a in X_std.view(-1))
        tform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(X_mean_tuple, X_std_tuple)])

        # create normalized data set
        train = self.torch_dataset_fn(
            dataset_dir, download=download, train=True,
            transform=tform)  # Dataset(X_norm, y, index_train, index_test, device)
        test = self.torch_dataset_fn(
            dataset_dir, download=download, train=False, transform=tform)
        self.norm = DatasetFromTorch(train, test, device, all_on_device=False)

        # save some data shapes and properties
        self.num_train_set = unnorm_x.shape[0]
        self.in_shape = unnorm_x.shape[1:]
        self.out_shape = unnorm_y.shape[1:]
        self.dtype = getattr(t, dtype)
        self.device = device


class RotatedMNIST:
    """
    The usage is:
    ```
    rot_mnist = RotatedMNIST()
    ```
    e.g. normalized training dataset:
    ```
    rot_mnist.norm.train
    ```
    """

    def __init__(self, dtype='float32', device="cpu", download=False):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/mnist/'

        # load data
        data_train = torchvision.datasets.MNIST(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.MNIST(dataset_dir, download=download, train=False)

        # Rotate the images
        np.random.seed(1337)

        data_test_rot_small = np.zeros_like(data_test.data)
        labels_rot_small = np.zeros_like(data_test.targets)

        for i, img in enumerate(data_test.data):
            angle = np.random.randint(low=-45, high=45)
            img_rot = ndimage.rotate(img, angle, reshape=False)
            data_test_rot_small[i] = img_rot
            labels_rot_small[i] = data_test.targets[i]

        data_test_rot_large = np.zeros_like(data_test.data)
        labels_rot_large = np.zeros_like(data_test.targets)

        for i, img in enumerate(data_test.data):
            angle = np.random.randint(low=-90, high=90)
            img_rot = ndimage.rotate(img, angle, reshape=False)
            data_test_rot_large[i] = img_rot
            labels_rot_large[i] = data_test.targets[i]

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data, data_test_rot_small,
                                                data_test_rot_large]).astype(dtype)).reshape([-1, 784])
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets, labels_rot_small,
                                         labels_rot_large]).astype('int'))

        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + 3 * len(data_test))

        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)

        # create normalized data set
        X_norm = self.unnorm.X / 255.
        self.norm = Dataset(X_norm, y, index_train, index_test, device)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]


class MNISTDataAugmentPrior:
    for_data_augment_prior = True
    torch_dataset_fn = torchvision.datasets.MNIST

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
                 max_rotation: float = 30.,
                 max_zoom_scale: [float, None] = None,
                 crop_before_norm: bool = False,
                 padding: int = 2,
                 do_hflip: bool = False,
                 save: bool = False,
                 load: bool = False,
                 data_subset_size: int = 50000,
                 root_dir=None):
        crop_size = 28
        assert not (save and load)
        assert data_subset_size == 50000, "Subsampling MNIST not yet supported"
        dataname = str(self.torch_dataset_fn).split('.')[-1].split("'")[0]
        _ROOT = root_dir or os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/{dataname.lower()}/'
        dtype = getattr(t, dtype)
        self.dtype = dtype
        self.device = device

        unnorm_train = self.torch_dataset_fn(
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
                transforms.ToTensor(),
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomRotation(degrees=[-max_rotation, max_rotation]),
                transforms.Normalize(mean=X_mean_tuple, std=X_std_tuple),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform()]
        else:
            transform_train = [
                transforms.ToTensor(),
                transforms.Normalize(mean=X_mean_tuple, std=X_std_tuple),
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomRotation(degrees=[-max_rotation, max_rotation]),
                transforms.RandomHorizontalFlip() if do_hflip else NoneTransform()]
            # ReshapeTransform((784,))]
        if max_zoom_scale is not None:
            transform_train += [transforms.RandomResizedCrop((28, 28),
                                                             scale=[1. - max_zoom_scale, 1. + max_zoom_scale],
                                                             ratio=(1., 1.))]
        transform_train = transforms.Compose(transform_train)
        saveload_path_train = self.get_save_path(dataset_dir,
                                                 n_mc_augment=n_mc_augment,
                                                 seed=seed,
                                                 do_hflip=do_hflip,
                                                 crop_size=crop_size,
                                                 padding=padding,
                                                 max_rotation=max_rotation,
                                                 train=True)

        data_train = self.torch_dataset_fn(dataset_dir,
                                           download=download,
                                           train=True,
                                           transform=transform_train)

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
                transforms.ToTensor(),
                transforms.Normalize(mean=X_mean_tuple, std=X_std_tuple),
            ])
        data_test = self.torch_dataset_fn(dataset_dir,
                                          download=download,
                                          train=False,
                                          transform=transform_test)
        saveload_path_test = self.get_save_path(root_dir=dataset_dir,
                                                n_mc_augment=n_mc_aug_test if test_time_aug else 1,
                                                seed=seed,
                                                do_hflip=do_hflip,
                                                crop_size=crop_size,
                                                padding=padding,
                                                max_rotation=max_rotation if test_time_aug else 0.,
                                                train=False)
        data_test_stacked = StackedDataset(dataset_to_stack=data_test,
                                           n_samples=n_mc_aug_test,
                                           n_subsample=n_subsample_test,
                                           base_seed=seed,
                                           save_path=saveload_path_test if save and test_time_aug else None,
                                           load_path=saveload_path_test if load and test_time_aug else None)

        all_data_on_device = n_mc_augment == 16
        self.norm = DatasetFromTorch(data_train_stacked,
                                     data_test_stacked,
                                     device=device,
                                     all_on_device=all_data_on_device)
        self.num_train_set = len(data_train)
        self.in_shape = t.Size([1, 28, 28])
        self.out_shape = t.Size([])

    def get_save_path(self,
                      root_dir,
                      n_mc_augment,
                      seed,
                      do_hflip,
                      max_rotation,
                      padding,
                      crop_size,
                      train=True, **kwargs_ignore):
        filename = f"{n_mc_augment}samples_{padding}pad_{crop_size}cropsize" + \
                   (f"_rotation{max_rotation}" if max_rotation > 0. else "") + \
                   f"{'_hflip' if do_hflip else ''}_seed{seed}_{'train' if train else 'test'}.npy"
        return os.path.join(root_dir, "augmented", filename)


class FashionMNISTDataAugmentPrior(MNISTDataAugmentPrior):
    torch_dataset_fn = torchvision.datasets.FashionMNIST


class FashionMNIST(MNIST):
    """
    The usage is:
    ```
    fmnist = FashionMNIST()
    ```
    e.g. normalized training dataset:
    ```
    fmnist.norm.train
    ```
    """

    def __init__(self, dtype='float32', device="cpu", download=False, root_dir=None):
        _ROOT = root_dir or os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/mnist/'

        # load data
        data_train = torchvision.datasets.FashionMNIST(dataset_dir, download=download, train=True)
        data_test = torchvision.datasets.FashionMNIST(dataset_dir, download=download, train=False)

        # get data into right shape and type
        X_unnorm = t.from_numpy(np.concatenate([data_train.data, data_test.data]).astype(dtype)).reshape([-1, 784])
        y = t.from_numpy(np.concatenate([data_train.targets, data_test.targets]).astype('int'))

        # train / test split
        index_train = np.arange(len(data_train))
        index_test = np.arange(len(data_train), len(data_train) + len(data_test))

        # create unnormalized data set
        self.unnorm = Dataset(X_unnorm, y, index_train, index_test, device)

        unnorm_train = torchvision.datasets.FashionMNIST(
            dataset_dir, download=download, train=True, transform=transforms.ToTensor())
        unnorm_x, _ = load_all(unnorm_train)
        X_mean = unnorm_x.mean(dim=(0, 2, 3), keepdims=True).to(device)
        X_std = unnorm_x.std(dim=(0, 2, 3), keepdims=True).to(device)
        self.X_mean = X_mean
        self.X_std = X_std
        X_mean_tuple = tuple(a.item() for a in X_mean.view(-1))
        X_std_tuple = tuple(a.item() for a in X_std.view(-1))
        tform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(X_mean_tuple, X_std_tuple)])

        # create normalized data set
        # X_norm = (self.unnorm.X - X_mean) / X_std
        train = torchvision.datasets.FashionMNIST(
            dataset_dir, download=download, train=True,
            transform=tform)  # Dataset(X_norm, y, index_train, index_test, device)
        test = torchvision.datasets.FashionMNIST(
            dataset_dir, download=download, train=False, transform=tform)
        self.norm = DatasetFromTorch(train, test, device, all_on_device=False)

        # save some data shapes
        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]
