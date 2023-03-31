'''
File to house CLI-related utility functions used throughout the project.
See train.py for an example
'''
import argparse
import torch
import numpy as np


def parse_args():
	'''
	This function creates argument parser and parses the scrip input arguments.
	This is the most common way to define input arguments in python.
	Used by train.py

	To change the parameters, pass them to the script, for example:

	python3 train.py --output_dir=outputs --weight_decay=0.01
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	'''
	parser = argparse.ArgumentParser(description="Fine-tune image classification model")

	# Required arguments
	parser.add_argument(
		"--dataset_dir",
		type=str,
		default="dataset",
		help="Root directory where the dataset is stored.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="outputs",
		help="Where to store the final model.",
	)
	parser.add_argument(
		"--pretrained_model_name",
		type=str,
		default="microsoft/resnet-18",
		help="Name of pretrained model to be used.",
	)
	parser.add_argument(
		"--dataset_size",
		type=int,
		default=None,
		help="Total number of (train/val/test) samples. If None, full dataset is used.",
	)
	parser.add_argument(
		"--percent_train",
		type=float,
		default=0.7,
		help="Percentage of the data to use for training (train_val_size * percent_train).",
	)
	parser.add_argument(
		"--percent_val",
		type=float,
		default=0.1,
		help="Percentage of the data to use for validation (train_val_size * percent_train).",
	)
	parser.add_argument(
		"--percent_test",
		type=float,
		default=0.2,
		help="Percentage of the data to use for testing (train_val_size * percent_train).",
	)
	parser.add_argument(
		"--weigh_classes",
		default=False,
		action="store_true",
		help="Whether to weigh classes during sampling in the training phase.",
	)
	parser.add_argument(
		"--debug",
		default=False,
		action="store_true",
		help="Whether to use a small subset of the dataset for debugging.",
	)

	# Training arguments
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device (cuda or cpu) on which the code should run",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=128,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=5e-4,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight_decay",
		type=float,
		default=0.0,
		help="Weight decay to use.",
	)
	parser.add_argument(
		"--num_train_epochs",
		type=int,
		default=15,
		help="Total number of training epochs to perform.",
	)
	parser.add_argument(
		"--eval_every_steps",
		type=int,
		default=40,
		help="Perform evaluation every n network updates.",
	)
	parser.add_argument(
		"--logging_steps",
		type=int,
		default=20,
		help="Compute and log training batch metrics every n steps.",
	)
	parser.add_argument(
		"--checkpoint_every_steps",
		type=int,
		default=500,
		help="Save model checkpoint every n steps.",
	)
	parser.add_argument(
		"--max_train_steps",
		type=int,
		default=None,
		help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--lr_scheduler_type",
		type=str,
		default="linear",
		help="The scheduler type to use.",
		choices=["no_scheduler", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="A seed for reproducible training.",
	)
	
	# Weights and biases (wandb) arguments
	parser.add_argument(
		"--use_wandb",
		default=False,
		action="store_true",
		help="Whether to enable usage/logging for the wandb_project.",
	)
	parser.add_argument(
		"--wandb_project", 
		default="medical_mnist",
		help="wandb project name to log metrics to"
	)

	args = parser.parse_args()
	return args
