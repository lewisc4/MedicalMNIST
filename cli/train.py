'''
File to train a model via CLI arguments
'''
import sys
# append the path of the
# parent directory
sys.path.append("..")

import logging

from datasets import load_metric
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import create_image_dataset


# Filename to create/log to during training
TRAIN_LOG_FNAME = 'train_log.log'
# ResNet image mean and std
RESNET_MEAN = (0.4914, 0.4822, 0.4465)
RESNET_STD = (0.2023, 0.1994, 0.2010)
# Medical MNIST dataset image size
MNIST_IMAGE_SIZE = (64, 64)

# Setup/initialize logging
logger = logging.getLogger()
f_handler = logging.FileHandler(filename=TRAIN_LOG_FNAME, mode='a')
formatter = logging.Formatter(
	fmt='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%m/%d/%Y %H:%M:%S',
)
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)
logger.setLevel(logging.INFO)

# Accuracy metric to use during training
accuracy = load_metric('accuracy')


def train_model(
	model,
	train_dataloader,
	val_dataloader,
	optimizer,
	scheduler,
	args
):
	''' Trains a model and returns it. '''
	pass


def evaluate_model(model, dataloader, args):
	''' Evaluates a given model on a given DataLoader. '''
	pass


def main():
	# Parse the cli arguments
	args = parse_args()
	# Get/build our train/val/test DataLoaders
	dataloaders = create_image_dataset(
		image_mean=RESNET_MEAN,
		image_std=RESNET_STD,
		image_size=MNIST_IMAGE_SIZE,
		args=args,
	)

	train_dataloader = dataloaders['train']
	val_dataloader = dataloaders['val']
	test_dataloader = dataloaders['test']


if __name__ == '__main__':
	main()
