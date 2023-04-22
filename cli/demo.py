'''
Script (run via CLI arguments) to demo a trained model.
'''
import torch
import gradio as gr

from torchvision import transforms
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import dataset_from_args
from medical_mnist.model_utils import init_model, load_model


# Parse the script arguments
args = parse_args()
# Create the dataset, based on the script arguments. We need the dataset to get
# the number of classes and the (class id -> class label name) mappings
dataset = dataset_from_args(args)
# Create the model shell. Here we don't want to use pretrained weights from
# PyTorch, because we are loading our own saved model into the shell to demo.
model_shell = init_model(
	args.model_architecture,
	num_classes=dataset.num_classes,
	pretrained=False
)
# Load the specified model file (the weights from the model we trained/saved)
# into the above model shell/architecture, making sure to switch to eval mode.
model = load_model(args.model_file, model_shell)
model.to(args.device)
model.eval()


def predict(model_in):
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
	gr.Interface(
		fn=predict,
		inputs=gr.Image(type='pil'),
		outputs=gr.Label(num_top_classes=3),
	).launch()


if __name__ == '__main__':
	main()
