'''
Script to train a model via CLI arguments.
For all available CLI arguments, see medical_mnist/cli_utils.py.
'''
import math
import copy
import torch
import transformers
# import evaluate

from datasets import load_metric
from tqdm import tqdm
from transformers import Adafactor
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import dataset_from_args
from medical_mnist.model_utils import init_model, save_model, evaluate_model
from medical_mnist.logging_utils import TrainLogger


# Setup logging
logger = TrainLogger()
logger.setup()
# Accuracy metric and loss function to use during training/evalutaion
# accuracy = evaluate.load('accuracy')
accuracy = load_metric('accuracy')


def train_model(model, train_data, val_data, optimizer, loss_f, scheduler, args):
	'''Trains a base model and returns the trained version

	Args:
		model (torchvision model): The base model to finetune/train
		train_data (torch.utils.data.DataLoader): Training DataLoader
		val_data (torch.utils.data.DataLoader): Validation DataLoader
		optimizer (torch.optim.Optimizer): Optimzer to use (from transformers)
		scheduler (SchedulerType): Learning rate scheduler to use
		args (Namespace): CLI arguments provided to this script

	Returns:
		torchvision model, float: Trained/finetuned model and best val loss
	'''
	# Track weights that yield the best val loss
	best_weights = copy.deepcopy(model.state_dict())
	best_loss = float('inf')
	# Progress bar and global step increment by one after each training batch
	progress_bar = tqdm(range(args.max_steps))
	step = 0
	# Train for specified number of training epochs
	for epoch in range(args.num_epochs):
		# Set the model to training mode and process a batch of training data
		model.train()
		for inputs, labels in train_data:
			# Move inputs/labels to the specified device and get model outputs
			inputs = inputs.to(args.device, non_blocking=args.non_blocking)
			labels = labels.to(args.device, non_blocking=args.non_blocking)
			model_output = model(inputs)
			preds = model_output.argmax(-1)
			accuracy.add_batch(predictions=preds, references=labels)
			# Zero out/clear gradients and perform backpropagation
			optimizer.zero_grad()
			train_loss = loss_f(model_output, labels)
			train_loss.backward()
			train_acc = accuracy.compute()['accuracy']
			# Advance the optimizer and, if provided, learning rate scheduler
			optimizer.step()
			if scheduler is not None:
				scheduler.step()
			# Advance our progress bar, global training step, and logger
			progress_bar.update(1)
			step += 1
			logger.progress(step=step, epoch=epoch)
			# Log training metrics
			logger.log_train_metrics(
				{'train_loss': train_loss.item(), 'Train Accuracy': train_acc}
			)
			# Evaluate our model, depending on the step
			if step % args.eval_every == 0 or step == args.max_steps:
				# Get validation metrics and check if best loss has improved
				val_metrics, _ = evaluate_model(
					model=model,
					data=val_data,
					loss_f=loss_f,
					device=args.device,
					non_blocking=args.non_blocking
				)
				if val_metrics['val_loss'] < best_loss:
					best_loss = val_metrics['val_loss']
					best_weights = copy.deepcopy(model.state_dict())
					logger.log_val_metrics(val_metrics)
			# Save our model checkpoint, based on the step
			if step % args.checkpoint_every == 0:
				logger.log_model_checkpoint()
				save_model(args.model_file, model)
			# Check if we are done training, based on the step
			if step >= args.max_steps:
				break
	# Load and return the model with the best weights
	model.load_state_dict(best_weights)
	return model, best_loss


def main():
	'''Main function to run training, using the user-provided CLI args
	'''
	# Parse the cli arguments
	args = parse_args()
	# Begin logging ASAP to log all stdout to the cloud
	logger.start(train_args=args)
	# Get the dataset, based on the script arguments
	dataset = dataset_from_args(args)
	dataloaders = dataset.dataloaders
	# Initialize the model to train
	model = init_model(
		args.model_architecture,
		num_classes=dataset.num_classes,
		pretrained=args.use_pretrained,
	)
	# Move the model to the specified device for training
	model.to(args.device)
	# Watch the model (on wandb) to log its gradients
	logger.watch_model(model)
	# Scheduler and math around the number of training steps
	steps_per_epoch = len(dataloaders['train'])
	# If there is no max # of training steps (None by default), set it based on
	# the # of training epochs and update steps per epoch. Otherwise, use the
	# max # of training steps to set the # of training epochs.
	if args.max_steps is None:
		args.max_steps = args.num_epochs * steps_per_epoch
	else:
		args.num_epochs = math.ceil(args.max_steps / steps_per_epoch)
	# Optimizer to use during training
	optimizer = Adafactor(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
		scale_parameter=False,
		relative_step=False,
	)
	# Define the loss function, using class weights if specified
	if args.weighted_loss:
		class_weights = dataset.class_weights.to(args.device)
		loss_f = torch.nn.CrossEntropyLoss(weight=class_weights)
	else:
		loss_f = torch.nn.CrossEntropyLoss()
	# Get the learning rate scheduler to use (if provided)
	scheduler = transformers.get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps,
		num_training_steps=args.max_steps
	)
	# Train a model
	logger.training_overview(steps_per_epoch=steps_per_epoch)
	trained_model, best_loss = train_model(
		model=model,
		train_data=dataloaders['train'],
		val_data=dataloaders['val'],
		optimizer=optimizer,
		loss_f=loss_f,
		scheduler=scheduler,
		args=args,
	)
	logger.training_summary(best_loss=best_loss)
	# Locally save the fine-tuned model with the best weights
	save_model(args.model_file, trained_model)
	# If specified, save the same model to wandb
	if args.upload_model:
		logger.upload_model()
	logger.finish()


if __name__ == '__main__':
	main()
