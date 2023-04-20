'''
Script to train a model via CLI arguments.
For all available CLI arguments, see medical_mnist/cli_utils.py.
'''
import math
import copy
import torch
import transformers

from tqdm import tqdm
from datasets import load_metric
from transformers import Adafactor
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import init_dataset
from medical_mnist.model_utils import init_model, save_model
from medical_mnist.logging_utils import TrainLogger


# Setup logging
logger = TrainLogger()
logger.setup()
# Accuracy metric and loss function to use during training/evalutaion
accuracy = load_metric('accuracy')
loss_func = torch.nn.CrossEntropyLoss()


def train(model, train_data, val_data, optimizer, scheduler, args):
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
	for epoch in range(args.num_train_epochs):
		# Set the model to training mode and process a batch of training data
		model.train()
		for inputs, labels in train_data:
			# Move inputs/labels to the specified device and get model outputs
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			model_output = model(inputs).to(args.device)
			preds = model_output.argmax(-1).to(args.device)
			accuracy.add_batch(predictions=preds, references=labels)
			# Zero out/clear gradients and perform backpropagation
			optimizer.zero_grad()
			train_loss = loss_func(model_output, labels).to(args.device)
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
				metrics={'Train Loss': train_loss.item(), 'Train Accuracy': train_acc}
			)
			# Evaluate our model, depending on the step
			if step % args.eval_every == 0 or step == args.max_steps:
				# Get validation metrics, check if best loss has improved
				val_loss, _, _ = evaluate(model, val_data, args)
				if val_loss < best_loss:
					best_loss = val_loss
					best_weights = copy.deepcopy(model.state_dict())
			# Save our model checkpoint, based on the step
			if step % args.checkpoint_every_steps == 0:
				logger.log_model_checkpoint()
				save_model(args.model_file, model)
			# Check if we are done training, based on the step
			if step >= args.max_steps:
				break
	# Load and return the model with the best weights
	model.load_state_dict(best_weights)
	return model, best_loss


def evaluate(model, data, args):
	'''Evaluates the provided model on the provided DataLoader

	Args:
		model (torchvision model): The model to evaluate
		data (torch.utils.data.DataLoader): The evaluation data's DataLoader
		args (Namespace): CLI arguments provided to this script

	Returns:
		float, float, tuple: Val loss, accuracy, and (label, prediction) pairs
	'''
	# Set the model to evaluation mode
	model.eval()
	# Keep track of running loss and model outputs (labels and their preds)
	running_loss = 0.0
	val_labels, val_preds = [], []
	for inputs, labels in tqdm(data, desc='Evaluation'):
		# Switch to inference mode b/c we're not using autograd
		with torch.inference_mode():
			# Move inputs and labels to the specified device
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			# Get model outputs/predictions and update running loss/accuracy
			model_output = model(inputs).to(args.device)
			preds = model_output.argmax(-1).to(args.device)
			accuracy.add_batch(predictions=preds, references=labels)
			running_loss += loss_func(model_output, labels).to(args.device)
			# Save labels/preds (useful for metrics like confusion matrices)
			val_labels += labels.tolist()
			val_preds += preds.tolist()
	# Set the model back to training mode
	model.train()
	# Compute the total (i.e. the batch) loss and accuracy
	val_loss = running_loss / len(data.dataset)
	val_acc = accuracy.compute()['accuracy']
	logger.log_val_metrics(
		metrics={'Val Loss': val_loss.item(), 'Val Accuracy': val_acc}
	)
	return val_loss, val_acc, (val_labels, val_preds)


def main():
	'''Main function to run training, using the user-provided CLI args
	'''
	# Parse the cli arguments
	args = parse_args()
	# Begin logging ASAP to log all stdout to the cloud
	logger.start(train_args=args)
	# Initialize the model to train
	model = init_model(args.model_architecture, pretrained=args.use_pretrained)
	# Move the model to the specified device for training
	model = model.to(device=args.device)
	# Watch the model (on wandb) to log its gradients
	logger.watch_model(model)

	# Get the dataset, based on the provided dataset name
	dataset = init_dataset(
		dataset_type=args.dataset_type,
		root=args.dataset_dir,
		val_size=args.percent_val,
		test_size=args.percent_test,
		batch_size=args.batch_size,
	)
	dataloaders = dataset.dataloaders

	# Scheduler and math around the number of training steps
	steps_per_epoch = len(dataloaders['train'])
	# If there is no max # of training steps (None by default), set it based on
	# the # of training epochs and update steps per epoch. Otherwise, use the
	# max # of training steps to set the # of training epochs.
	if args.max_steps is None:
		args.max_steps = args.num_train_epochs * steps_per_epoch
	else:
		args.num_train_epochs = math.ceil(args.max_steps / steps_per_epoch)
	# Optimizer to use during training
	optimizer = Adafactor(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
		scale_parameter=False,
		relative_step=False,
	)
	# Get the learning rate scheduler to use (if provided)
	scheduler = transformers.get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps,
		num_training_steps=args.max_steps
	)
	# Train a model
	logger.training_overview(steps_per_epoch=steps_per_epoch)
	trained_model, best_loss = train(
		model=model,
		train_data=dataloaders['train'],
		val_data=dataloaders['val'],
		optimizer=optimizer,
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
