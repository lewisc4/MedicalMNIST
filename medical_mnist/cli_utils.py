'''
File to house CLI-related utility functions used throughout the project.
See cli/train.py for example usage.
'''
import os
import copy
import argparse
import torch

from os.path import join, splitext


def parse_args():
	'''
	This function creates argument parser and parses the scrip input arguments.
	This is the most common way to define input arguments in python.
	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script, for example:

	python3 train.py --output_dir=outputs --weight_decay=0.01
	'''
	parser = argparse.ArgumentParser(description='Fine-tune image classification model')
	# Dataset/model arguments
	parser.add_argument(
		'--dataset_dir',
		type=str,
		default='dataset',
		help='Root directory where the dataset is stored.',
	)
	parser.add_argument(
		'--dataset_type',
		type=str,
		default='retinal-oct',
		choices=['medical-mnist', 'retinal-oct'],
		help='Name (i.e., the type of) dataset to use for training.',
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default='outputs',
		help='Where to store the final model.',
	)
	parser.add_argument(
		'--model_file',
		type=str,
		default='saved_model',
		help='Name of the model file to save to/load from'
	)
	parser.add_argument(
		'--model_architecture',
		type=str,
		default='resnet-18',
		choices=['resnet-18', 'resnet-50', 'vgg-16', 'alexnet'],
		help='The type of (CNN) model architecture to use.',
	)
	parser.add_argument(
		'--use_pretrained',
		default=True,
		action='store_true',
		help='Tells the model to use pretrained ImageNet weights',
	)
	parser.add_argument(
		'--from_scratch',
		dest='use_pretrained',
		action='store_false',
		help='Tells the model to NOT use pretrained ImageNet weights (i.e. train from scratch)',
	)
	parser.add_argument(
		'--percent_val',
		type=float,
		default=0.1,
		help='Percentage of the data to use for validation (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--percent_test',
		type=float,
		default=0.2,
		help='Percentage of the data to use for testing (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--weighted_sampling',
		default=False,
		action='store_true',
		help='Controls if class weights are used when sampling during training.',
	)
	parser.add_argument(
		'--weighted_loss',
		default=False,
		action='store_true',
		help='Controls if class weights are used by the loss function.',
	)
	# Training arguments
	parser.add_argument(
		'--device',
		default='cuda' if torch.cuda.is_available() else 'cpu',
		help='Device (cuda or cpu) on which the code should run',
	)
	parser.add_argument(
		'--non_blocking',
		default=False,
		action='store_true',
		help='Controls if non-blocking transfers are used in .to(device) for batch data.',
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=128,
		help='Batch size (per device) for each DataLoader.',
	)
	parser.add_argument(
		'--num_workers',
		type=int,
		default=0,
		help='Number of workers to use for each DataLoader.',
	)
	parser.add_argument(
		'--pin_memory',
		default=False,
		action='store_true',
		help='Controls if each DataLoader to use pinned memory.',
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=5e-4,
		help='Initial learning rate (after the potential warmup period) to use.',
	)
	parser.add_argument(
		'--weight_decay',
		type=float,
		default=0.0,
		help='Weight decay to use.',
	)
	parser.add_argument(
		'--num_epochs',
		type=int,
		default=15,
		help='Total number of training epochs to perform.',
	)
	parser.add_argument(
		'--eval_every',
		type=int,
		default=40,
		help='Perform evaluation every n network updates.',
	)
	parser.add_argument(
		'--log_every',
		type=int,
		default=20,
		help='Compute and log training batch metrics every n steps.',
	)
	parser.add_argument(
		'--checkpoint_every',
		type=int,
		default=500,
		help='Saves a model checkpoint every n steps.',
	)
	parser.add_argument(
		'--max_steps',
		type=int,
		default=None,
		help='Number of training steps to perform. If provided, overrides num_epochs.',
	)
	parser.add_argument(
		'--lr_scheduler_type',
		type=str,
		default='linear',
		choices=[
			'linear', 'cosine', 'cosine_with_restarts',
			'polynomial', 'constant', 'constant_with_warmup',
		],
		help='The learning rate scheduler type to use.',
	)
	parser.add_argument(
		'--num_warmup_steps',
		type=int,
		default=0,
		help='Number of steps for the warmup in the lr scheduler.'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='A seed for reproducible training.',
	)
	parser.add_argument(
		'--metric_dir',
		type=str,
		default='metrics',
		help='Directory to save metrics to.'
	)
	parser.add_argument(
		'--metric_pfx',
		type=str,
		default='',
		help='Prefix used in metric file names/titles to uniquely identify them.'
	)
	parser.add_argument(
		'--scores_file',
		type=str,
		default='scores',
		help='Name of the file to save scores (precision, recall, etc.) to.'
	)
	parser.add_argument(
		'--conf_mat_file',
		type=str,
		default='confusion_matrix',
		help='Name of the file to save a confusion matrix figure to.'
	)
	parser.add_argument(
		'--demo_gradcam',
		default=False,
		action='store_true',
		help='Tells the demo to use GradCAM instead of standard model.',
	)
	# Weights and biases (wandb) arguments
	parser.add_argument(
		'--wandb_project',
		default='medical_mnist',
		help='wandb project name to log metrics to'
	)
	parser.add_argument(
		'--upload_model',
		default=False,
		action='store_true',
		help='Whether to upload the trained model to wandb.',
	)
	# Parse the CLI arguments and validate them
	args = parser.parse_args()
	valid_args = validate_args(args)
	return valid_args


def validate_args(args):
	'''Validates/updates the CLI arguments parsed in the parse_args function,
	defined in this file.

	Args:
		args (Namespace): The CLI arguments to validate
	'''
	valid = copy.deepcopy(args)
	# Check if output dir (where models are saved) exists, if not create it
	os.makedirs(valid.output_dir, exist_ok=True)
	# Make sure metric dir (where metrics are saved) exists, if not create it
	os.makedirs(valid.metric_dir, exist_ok=True)
	# Update file names to use the valid paths
	# Also, strip the last extension (if it exists) and add a valid extension
	valid.model_file = join(
		valid.output_dir,
		splitext(valid.model_file)[0] + '.pt',
	)
	valid.scores_file = join(
		valid.metric_dir,
		valid.metric_pfx + splitext(valid.scores_file)[0] + '.txt',
	)
	valid.conf_mat_file = os.path.join(
		valid.metric_dir,
		valid.metric_pfx + splitext(valid.conf_mat_file)[0] + '.png',
	)
	return valid
