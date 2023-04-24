'''
Script (run via CLI arguments) to demo a trained model.
'''
import torch
import numpy as np
import gradio as gr

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import dataset_from_args
from medical_mnist.model_utils import load_model


# Target layer mappings to use with GradCAM for each mode architecture
TARGET_LAYERS = {
	'resnet-18': lambda net: net.layer4[-1],
	'resnet-50': lambda net: net.layer4[-1],
	'vgg-16': lambda net: net.features[-1],
	'alexnet': lambda net: net.features[-1],
}
# Parse the script arguments
args = parse_args()
# Create the dataset, based on the script arguments. We need the dataset to get
# the number of classes and the (class id -> class label name) mappings
dataset = dataset_from_args(args)
# Load the specified model file (the weights from the model we trained/saved)
# based on the specified model architecture and number of output classes.
model = load_model(
	path=args.model_file,
	model_type=args.model_architecture,
	num_classes=dataset.num_classes,
)
# Move the model to the specified device and switch to eval() mode
model.to(args.device)
model.eval()
# Set the GradCAM object if this is a GradCAM demo
if args.demo_gradcam:
	target_layers = [TARGET_LAYERS[args.model_architecture](model)]
	cam = GradCAM(model=model, target_layers=target_layers)


def gradcam_demo(model_in):
	'''Given model input (i.e., an image), gets its GradCAM visualization

	Args:
		model_in (gradio.Image): The input image to feed to the model

	Returns:
		gradio.Image: The GradCAM visualization
	'''
	# Convert input to an RGB image
	rgb_image = np.float32(model_in) / 255
	model_in = transforms.ToTensor()(model_in).unsqueeze(0)
	# Get GradCAM output
	grayscale_cam = cam(input_tensor=model_in)
	grayscale_cam = grayscale_cam[0, :]
	# Overlay GradCAM image with the RGB image
	vis = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
	return vis


def standard_demo(model_in):
	'''Given model input (i.e., an image), gets confidence for each class

	Args:
		model_in (gradio.Image): The input image to feed to the model

	Returns:
		dict[Any, float]: Confidence for each class label id
	'''
	model_in = transforms.ToTensor()(model_in).unsqueeze(0)
	idx_to_label = dataset.get_label_id_maps()['id_to_label']
	indices = range(len(idx_to_label.keys()))
	with torch.no_grad():
		model_out = model(model_in).to(args.device)
		class_preds = torch.nn.functional.softmax(model_out[0], dim=0)
		confidences = {idx_to_label[i]: float(class_preds[i]) for i in indices}
	return confidences


def main():
	'''Runs the Gradio demo, based on the provided script args.
	The arguments, dataset, and model are defined at the top of this file.
	'''
	if args.demo_gradcam:
			gr.Interface(
			fn=gradcam_demo,
			inputs=gr.Image(type='pil'),
			outputs=gr.Image(type='pil'),
		).launch()
	else:
		gr.Interface(
			fn=standard_demo,
			inputs=gr.Image(type='pil'),
			outputs=gr.Label(num_top_classes=3),
		).launch()


if __name__ == '__main__':
	main()
