'''
File to house dataset-related utility functions used throughout the project.
See train.py for example usage
'''
import os
import torch
import numpy as np

from itertools import chain
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split


def dataset_from_args(args):
    '''Initializes a dataset from cli args. See cli_utils.py for avail. args.

    Args:
        args (Namespace): CLI arguments to create the dataset

    Returns:
        ProjectDataset: Dataset class (MedicalMNISTDataset or RetinalOCTDataset)
    '''
    # The different options to create a dataset class (values), based on the
    # provided argument for the dataset name (keys)
    dataset_options = {
        'medical-mnist': MedicalMNISTDataset,
        'retinal-oct': RetinalOCTDataset,
    }
    dataset_class = dataset_options.get(args.dataset_type, 'retinal-oct')
    return dataset_class(
        root=args.dataset_dir,
        val_size=args.percent_val,
        test_size=args.percent_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        weighted_sampling=args.weighted_sampling,
        seed=args.seed,
    )


def flatten(iterable):
    '''Flattens an iterable (e.g., a list of lists) into one list

    Args:
        iterable (iterable): The iterable to flatten

    Returns:
        list: The flattened iterable as a list
    '''
    return list(chain.from_iterable(iterable))


@dataclass
class ProjectDataset(ABC):
    '''Parent class for datasets used in this project.
    '''
    root: str # The root image folder
    val_size: float # Percentage of data to use as validation data
    test_size: float # Percentage of data to use as training data
    batch_size: int # The batch size to use in the DataLoaders
    num_workers: int # The number of workers to use in the DataLoaders
    pin_memory: bool # Whether to pin memory in DataLoaders or not
    # Default image mean and standard deviation are from ImageNet
    image_mean: list = (0.485, 0.456, 0.406) # Per-channel means for transforms
    image_std: tuple = (0.229, 0.224, 0.225) # Per-channel stds for transforms
    image_size: tuple = (64, 64) # The image size to use for image transforms
    weighted_sampling: bool = False # Whether to weigh classes in train sampler
    seed: int = 42 # Random seed to use for reproducibility

    def __post_init__(self):
        '''Creates the following dataset components:

            image_transforms (dict): train/target image transforms
            image_folders (dict): train/val/test ImageFolders
            subsets (dict): train/val/test ImageFolder Subsets
            dataloaders (dict): train/val/test DataLoaders from respective subsets
        '''
        self.image_transforms = self.get_transforms()
        self.image_folders = self.get_image_folders()
        self.subsets = self.get_subsets()
        self.class_weights = self.get_class_weights()
        self.dataloaders = self.get_dataloaders()
        self.label_id_maps = self.get_label_id_maps()
        self.num_classes = len(self.image_folders['train'].classes)

    def get_transforms(self):
        '''Get train/target image transforms for augmentation/normalization,
        based on the image mean, std, and size.

        Returns:
            dict: Dictionary of train/test image transforms
        '''
        train_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=60),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )
        target_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.image_mean, self.image_std),
            ]
        )
        return {'train': train_transform, 'target': target_transform}

    @abstractmethod
    def get_image_folders(self):
        '''Get train/val/test ImageFolders from root folder and train/target transforms.

        Raises:
            NotImplementedError: Thrown if not implemented in child class
        '''
        raise NotImplementedError

    @abstractmethod
    def get_subsets(self):
        '''Gets stratified train/val/test Subsets from their respective ImageFolders.

        Raises:
            NotImplementedError: Thrown if not implemented in child class
        '''
        raise NotImplementedError

    @abstractmethod
    def get_targets(self, split='train'):
        '''Gets the dataset's targets (from a Subset) for a given split.

        Args:
            split (str, optional): Dataset's split to use. Defaults to 'train'.

        Raises:
            NotImplementedError: Thrown if not implemented in child class
        '''
        raise NotImplementedError

    def get_dataloaders(self):
        '''Get train/val/test DataLoaders from their respective Subsets

        Returns:
            dict: Dictionary of train/val/test DataLoaders
        '''
        train_sub = self.subsets['train']
        val_sub = self.subsets['val']
        test_sub = self.subsets['test']
        # If weighing classes, use WeightedRandomSampler (for training)
        if self.weighted_sampling:
            weights = self.class_weights[self.get_targets()]
            train_sampler = WeightedRandomSampler(weights, len(weights))
            # Use RandomSampler if we don't want to use class weights (for training)
        else:
            train_sampler = RandomSampler(train_sub)
        # Never use weighted sampler for val/test sets
        val_sampler = RandomSampler(val_sub)
        test_sampler = RandomSampler(test_sub)
        # Create the train/val/test DataLoaders
        train_dataloader = DataLoader(
            train_sub,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_dataloader = DataLoader(
            val_sub,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_dataloader = DataLoader(
            test_sub,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return {
            'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader,
        }

    def get_class_weights(self):
        '''Gets class weights based on the training Subset.

        Returns:
            torch.Tensor: A PyTorch (double) Tensor, containing the class weights
        '''
        targets = self.get_targets()
        counts = np.unique(targets, return_counts=True)[1]
        return torch.from_numpy(1. / counts).float()

    def get_label_id_maps(self, split='train'):
        '''Gets ImageFolder [label: label_id] and [label_id: label] maps.

        Args:
            split (str, optional): Dataset's split to use. Defaults to 'train'.

        Returns:
            dict: The ImageFolder [label: label_id] and [label_id: label] maps
        '''
        label_to_idx = self.image_folders[split].class_to_idx
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return {
            'label_to_id': label_to_idx,
            'id_to_label': idx_to_label,
        }


@dataclass
class MedicalMNISTDataset(ProjectDataset):
    '''Class for the Medical-MNIST dataset on Kaggle.
    See: https://www.kaggle.com/datasets/andrewmvd/medical-mnist
    '''
    def get_image_folders(self):
        '''Get train/val/test ImageFolders based on the root folder and
        the train/target transforms. For this dataset, THERE ARE NOT specific
        train/val/test folders, so they all share the same root folder.

        Returns:
            dict: train/val/test ImageFolders w/ their respective image transforms
        '''
        train_transform = self.image_transforms['train']
        target_transform = self.image_transforms['target']
        return {
            'train': datasets.ImageFolder(self.root, transform=train_transform),
            'val': datasets.ImageFolder(self.root, transform=train_transform),
            'test': datasets.ImageFolder(self.root, transform=target_transform),
        }

    def get_targets(self, split='train'):
        '''Gets the dataset's targets (from a Subset) for a given split.

        Args:
            split (str, optional): Dataset's split to use. Defaults to 'train'.

        Returns:
            list: The targets corresponding to the split's sample indices
        '''
        indices = self.subsets[split].indices
        targets = self.subsets[split].dataset.targets
        return [targets[i] for i in indices]

    def get_subsets(self):
        '''Gets stratified train/val/test Subsets from their respective ImageFolders.
        For this dataset, there ARE NOT pre-defined train/val/test images, so we
        split all the images into our own train/val/test Subsets.

        Returns:
            dict: Dictionary of train/val/test Subsets
        '''
        train_set = self.image_folders['train']
        test_set = self.image_folders['test']
        num_labels = len(train_set.targets)
        # Given the full dataset, separate the train/val samples from test samples
        indices = list(range(num_labels))
        train_val_idx, target_idx, train_val_labels, _ = train_test_split(
            indices,
            train_set.targets,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=train_set.targets,
        )
        # Given the train/val dataset, separate the train and val samples
        train_val_ratio = num_labels / len(train_val_labels)
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx,
            train_val_labels,
            test_size=(self.val_size * train_val_ratio),
            stratify=train_val_labels,
        )
        # Get train/val/test Subsets using their respective ImageFolder/indices
        return {
            'train': Subset(train_set, train_idx),
            'val': Subset(train_set, val_idx),
            'test': Subset(test_set, target_idx),
        }


@dataclass
class RetinalOCTDataset(ProjectDataset):
    '''Class for the Retinal OCT dataset on Kaggle.
    See: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
    '''
    def get_image_folders(self):
        '''Get train/val/test ImageFolders based on the root folder and
        the train/target transforms. For this dataset, THERE ARE specific
        train/val/test folders, so each as a different root folder.

        Returns:
            dict: train/val/test ImageFolders w/ their respective image transforms
        '''
        train_root = os.path.join(self.root, 'train')
        val_root = os.path.join(self.root, 'val')
        test_root = os.path.join(self.root, 'test')
        train_transform = self.image_transforms['train']
        target_transform = self.image_transforms['target']
        return {
            'train': datasets.ImageFolder(train_root, transform=train_transform),
            'val': datasets.ImageFolder(val_root, transform=train_transform),
            'test': datasets.ImageFolder(test_root, transform=target_transform),
        }

    def get_targets(self, split='train'):
        '''Gets the dataset's targets (from a Subset) for a given split.
        Overrides parent class, because this dataset uses ConcatDataset objects
        to combine the pre-defined train/val/test sets on Kaggle, so we can
        make our own custom splits.

        Args:
            split (str, optional): Dataset's split to use. Defaults to 'train'.

        Returns:
            list: The targets corresponding to the split's sample indices
        '''
        indices = self.subsets[split].indices
        concatenated_data = self.subsets[split].dataset.datasets
        targets = flatten([data.targets for data in concatenated_data])
        return [targets[i] for i in indices]

    def get_subsets(self):
        '''Gets stratified train/val/test Subsets from their respective ImageFolders.
        For this dataset, there ARE pre-defined train/val/test images. We combine the
        training and validation images and split them based on val_size and leave
        the test images as they are.

        Returns:
            dict: Dictionary of train/val/test Subsets
        '''
        train_set = self.image_folders['train']
        val_set = self.image_folders['val']
        train_val_set = ConcatDataset([train_set, val_set])
        # Get the train and validation targets and flatten them
        train_val_targets = [d.targets for d in train_val_set.datasets]
        all_targets = [tgt for targets in train_val_targets for tgt in targets]

        num_labels = len(train_val_set)
        # Given the full dataset, separate the train/val samples from test samples
        indices = list(range(num_labels))
        train_idx, val_idx, _, _ = train_test_split(
            indices,
            all_targets,
            test_size=self.val_size,
            random_state=self.seed,
            stratify=all_targets,
        )
        # Get train/val/test Subsets using their respective ImageFolder/indices
        return {
            'train': Subset(train_val_set, train_idx),
            'val': Subset(train_val_set, val_idx),
            'test': self.image_folders['test'],
        }
