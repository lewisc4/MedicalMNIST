'''
File to house dataset-related utility functions used throughout the project.
See train.py for an example
'''
import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# Default image transform
DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def create_image_dataset(image_mean, image_std, image_size, args):
	''' Builds and returns train/val/test dataloaders '''
	# Get the train/test image transforms
	image_transforms = get_train_test_transforms(
		image_mean=image_size,
		image_std=image_size,
		image_size=image_size,
	)
	# Get the train/test ImageFolders using the transforms
	image_data_folders = get_image_folder_datasets(
		root=args.dataset_dir,
		train_transform=image_transforms['train'],
		test_transform=image_transforms['test'],
	)
	# Get the stratified train/val/test Subsets using the ImageFolders
	image_subsets = get_subsets(
		train_set=image_data_folders['train'],
		test_set=image_data_folders['test'],
		train_size=args.percent_train,
		val_size=args.percent_val,
		test_size=args.percent_test,
		seed=args.seed,
	)
	# Get the train/val/test DataLoaders using respective Subsets
	dataloaders = get_dataloaders(
		train_subset=image_subsets['train'],
		val_subset=image_subsets['val'],
		test_subset=image_subsets['test'],
		bs=args.batch_size,
		weigh_classes=args.weigh_classes,
	)
	return dataloaders


def get_dataloaders(
	train_subset,
	val_subset,
	test_subset,
	bs,
	weigh_classes=False
):	
	''' Get train/val/test DataLoaders from respective subsets. '''
	# If weighing classes, use WeightedRandomSampler (for training)
	if weigh_classes:
		# Get weights based on our training data's targets
		targets = [train_subset.dataset.targets[i] for i in train_subset.indices]
		weights = get_target_weights(targets)
		train_sampler = WeightedRandomSampler(weights, len(weights))
	# Use RandomSampler if we don't want to use class weights (for training)
	else:
		train_sampler = RandomSampler(train_subset)
	# Never use weighted sampler for val/test sets
	val_sampler = RandomSampler(val_subset)
	test_sampler = RandomSampler(test_subset)
	# 
	return {
		'train': DataLoader(train_subset, batch_size=bs, sampler=train_sampler),
		'val': DataLoader(val_subset, batch_size=bs, sampler=val_sampler),
		'test': DataLoader(test_subset, batch_size=bs, sampler=test_sampler),
	}


def get_target_weights(targets):
	''' Given a list of class targets, get their respectiv weights. '''
	counts = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
	weight_per_target = 1. / counts
	target_weights = np.array([weight_per_target[t] for t in targets])
	target_weights = torch.from_numpy(target_weights)
	return target_weights.double()


def get_subsets(train_set, test_set, train_size, val_size, test_size, seed):
	''' Get stratified train/val/test Subsets given train/test ImageFolders. '''
	# Make sure train and test ImageFolders have the same data
	# They should only differ in their transform functions
	assert train_set.targets == test_set.targets, 'train/test targets differ'
	num_labels = len(train_set.targets)
	# Given the full dataset, separate the train/val samples from test samples
	indices = list(range(num_labels))
	train_val_indices, test_indices, train_val_labels, _ = train_test_split(
		indices,
		train_set.targets,
		test_size=test_size,
		random_state=seed,
		stratify=train_set.targets,
	)
	# Given the train/val dataset, separate the train samples from val samples
	train_val_ratio = num_labels / len(train_val_labels)
	train_indices, val_indices, _, _ = train_test_split(
		train_val_indices,
		train_val_labels,
		test_size=(val_size * train_val_ratio),
		stratify=train_val_labels,
	)
	# Get train/val/test Subsets using their respective ImageFolder & indices
	return {
		'train': Subset(train_set, train_indices),
		'val': Subset(train_set, val_indices),
		'test': Subset(test_set, test_indices),
	}


def get_image_folder_datasets(root, train_transform=None, test_transform=None):
	''' Get train/test ImageFolder datasets from data dir and transforms. '''
	# If no transforms provided, use the defaults
	if train_transform is None:
		train_transform = DEFAULT_TRANSFORM
	if test_transform is None:
		test_transform = DEFAULT_TRANSFORM
	return {
		'train': datasets.ImageFolder(root=root, transform=train_transform),
		'test': datasets.ImageFolder(root=root, transform=test_transform),
	}


def get_train_test_transforms(image_mean, image_std, image_size=(64, 64)):
	''' Get train/test image transforms from image mean, std, and size. '''
	train_transform = transforms.Compose(
		[
			transforms.Resize(image_size),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(degrees=60),
			transforms.ToTensor(),
			transforms.Normalize(image_mean, image_std),
		]
	)
	test_transform = transforms.Compose(
		[
			transforms.Resize(image_size),
			transforms.ToTensor(),
			transforms.Normalize(image_mean, image_std),
		]
	)
	return {'train': train_transform, 'test': test_transform}

