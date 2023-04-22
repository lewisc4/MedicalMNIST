'''
File to house model-related utility functions used throughout the project.
See cli/train.py for example usage
'''
import torch
import torchvision

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


def init_model(model_name, num_classes=1000, pretrained=True):
	'''Initialize a model based on name (i.e., architecture)

	Args:
		model_name (str): The model architecture to use
		num_classes (int, optional): # of classes to use for output features
		pretrained (bool, optional): Whether to use pretrained weights or not

	Returns:
		torchvision model: The initialized torchvision model
	'''
	# Get the model architecture, input features, and weights by name
	# Defaults to ResNet-18, with 1000 classes (ImageNet) and no weights
	architecture = PRETRAINED_MODELS.get(model_name, torchvision.models.resnet18)
	weights = PRETRAINED_WEIGHTS.get(model_name, None) if pretrained else None
	model = architecture(weights=weights)
	# Update the model's final layer with the new number of classes
	if model_name in ('resnet-18', 'resnet-50'):
		model.fc = Linear(model.fc.in_features, num_classes)
	elif model_name in ('vgg-16', 'alexnet'):
		model.classifier[6] = Linear(model.classifier[6].in_features, num_classes)
	return model

def save_model(path, model):
	'''Saves a model to the provided path (including filename)

	Args:
		path (str): Full path to the model's save file (should be .pt or .pth)
		model (torchvision model): The model to save
	'''
	torch.save({'model_state_dict': model.state_dict()}, path)


def load_model(path, model):
	'''Loads a model from the provided path and model shell

	Args:
		path (str): Full path to the model's save file (should be .pt or .pth)
		model (torchvision model): The model shell to load weights into

	Returns:
		torchvision model: The model with loaded weights
	'''
	saved_state = torch.load(path)
	model.load_state_dict(saved_state['model_state_dict'])
	return model
