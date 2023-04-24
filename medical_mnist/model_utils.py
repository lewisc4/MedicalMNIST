'''
File to house model-related utility functions used throughout the project.
See cli/train.py for example usage
'''
import torch
import torchvision

from datasets import load_metric
from tqdm import tqdm
from torch.nn import Linear


# Available pretrained model architectures
PRETRAINED_MODELS = {
	'resnet-18': torchvision.models.resnet18,
	'resnet-50': torchvision.models.resnet50,
	'vgg-16': torchvision.models.vgg16,
	'alexnet': torchvision.models.alexnet,
}
# Available pretrained model weights
PRETRAINED_WEIGHTS = {
	'resnet-18': torchvision.models.ResNet18_Weights.DEFAULT,
	'resnet-50': torchvision.models.ResNet50_Weights.DEFAULT,
	'vgg-16': torchvision.models.VGG16_Weights.DEFAULT,
	'alexnet': torchvision.models.AlexNet_Weights.DEFAULT,
}
# Accuracy metric and loss function to use during training/evalutaion
# accuracy = evaluate.load('accuracy')
accuracy = load_metric('accuracy')


def init_model(model_type, num_classes=1000, pretrained=True):
	'''Initialize a model based on name (i.e., architecture)

	Args:
		model_type (str): The model architecture to use
		num_classes (int, optional): # of classes to use for output features
		pretrained (bool, optional): Whether to use pretrained weights or not

	Returns:
		torchvision model: The initialized torchvision model
	'''
	# Get the model architecture, input features, and weights by name
	# Defaults to ResNet-18, with 1000 classes (ImageNet) and no weights
	architecture = PRETRAINED_MODELS.get(model_type, torchvision.models.resnet18)
	weights = PRETRAINED_WEIGHTS.get(model_type, None) if pretrained else None
	model = architecture(weights=weights)
	# Update the model's final layer with the new number of classes
	if model_type in ('resnet-18', 'resnet-50'):
		model.fc = Linear(model.fc.in_features, num_classes)
	elif model_type in ('vgg-16', 'alexnet'):
		model.classifier[6] = Linear(model.classifier[6].in_features, num_classes)
	return model


def save_model(path, model):
	'''Saves a model to the provided path (including filename)

	Args:
		path (str): Full path to the model's save file (should be .pt or .pth)
		model (torchvision model): The model to save
	'''
	torch.save({'model_state_dict': model.state_dict()}, path)


def load_model(path, model_type, num_classes):
	'''Loads a model from the provided path and model shell

	Args:
		path (str): Full path to the model's save file (should be .pt or .pth)
		model (torchvision model): The model shell to load weights into

	Returns:
		torchvision model: The model with loaded weights
	'''
	# Create the model "shell". Here we don't want to use pretrained weights,
	# because we are loading our own saved model weights.
	model = init_model(model_type, num_classes=num_classes, pretrained=False)
	# Load the model weights (stored in state_dict) into the shell
	saved_state = torch.load(path)
	model.load_state_dict(saved_state['model_state_dict'])
	return model


def evaluate_model(model, data, loss_f, device, non_blocking=False):
	'''Evaluates the provided model on the provided DataLoader

	Args:
		model (torchvision model): The model to evaluate
		data (torch.utils.data.DataLoader): The evaluation data's DataLoader
		loss_f (torch.nn.modules.loss): The loss function to use
		device (torch.cuda.device): The device to evaluate the model on
		non_blocking (bool, optional): Whether to use non_blocking memory transfers.

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
			inputs = inputs.to(device, non_blocking=non_blocking)
			labels = labels.to(device, non_blocking=non_blocking)
			# Get model outputs/predictions and update running loss/accuracy
			model_output = model(inputs)
			preds = model_output.argmax(-1)
			accuracy.add_batch(predictions=preds, references=labels)
			running_loss += loss_f(model_output, labels)
			# Save labels/preds (useful for metrics like confusion matrices)
			val_labels += labels.tolist()
			val_preds += preds.tolist()
	# Set the model back to training mode
	model.train()
	# Compute the total (i.e. the batch) loss and accuracy
	val_loss = running_loss / len(data.dataset)
	val_acc = accuracy.compute()['accuracy']
	# Format and return the validation metrics and (true/pred) label ids
	val_metrics = {'val_loss': val_loss.item(), 'val_accuracy': val_acc}
	val_label_ids = {'actual': val_labels, 'predicted': val_preds}
	return val_metrics, val_label_ids
