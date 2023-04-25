'''
Script to perform model evaluations via CLI arguments.
For all available CLI arguments, see medical_mnist/cli_utils.py.
'''
import os
import json
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
from medical_mnist.cli_utils import parse_args
from medical_mnist.dataset_utils import dataset_from_args
from medical_mnist.model_utils import load_model, evaluate_model


# Score names to use/return from  the precision_recall_fscore_support function
SCORE_NAMES = ['Precision', 'Recall', 'F1_Score']


def get_scores(true_labels, pred_labels, class_labels):
	'''Get per-class precision, recall, and F1-scores from predictions.

	Args:
		true_labels (list): The true (i.e., actual) labels
		pred_labels (list): The predicted labels
		class_labels (list): The (unique) class names

	Returns:
		dict: Dictionary containing the scores for each class
	'''
	scores = score(true_labels, pred_labels, labels=class_labels)
	label_scores = [dict(zip(class_labels, metric)) for metric in scores]
	return dict(zip(SCORE_NAMES, label_scores))


def get_confusion_matrix(true_labels, pred_labels, class_labels, pfx=None):
	'''Gets a confusion matrix (matplotlib figure) from predictions.

	Args:
		true_labels (list): The true (i.e., actual) labels
		pred_labels (list): The predicted labels
		class_labels (list): The (unique) class names
		title_pfx (str, optional): The figure's title's prefix.

	Returns:
		matplotlib.figure.Figure: The figure containing the confusion_matrix
	'''
	_, ax = plt.subplots(figsize=(10, 10))
	pfx = '' if pfx is None else pfx
	plt.title(pfx + 'Confusion Matrix')
	conf_mat = confusion_matrix(
		true_labels,
		pred_labels,
		labels=class_labels,
		normalize='pred'
	)
	disp = ConfusionMatrixDisplay(
		confusion_matrix=conf_mat,
		display_labels=class_labels
	)
	disp.plot(ax=ax, cmap=plt.cm.Blues)
	# Return the current pyplot Figure
	return plt.gcf()


def main():
	'''Main function to run model evaluations, using the user-provided CLI args
	'''
	# Parse the cli arguments
	args = parse_args()
	# Get the dataset, based on the script arguments
	dataset = dataset_from_args(args)
	# Get the dataset's class id -> class name map (dictionary)
	id_to_label = dataset.get_label_id_maps()['id_to_label']
	class_labels = list(id_to_label.values())
	# Load the specified model file (the weights from the model we trained/saved)
	# based on the specified model architecture and number of output classes.
	model = load_model(
		path=args.model_file,
		model_type=args.model_architecture,
		num_classes=dataset.num_classes,
		device=args.device,
	)
	model.to(args.device)
	# Define the loss function, using class weights if specified
	# Class weights are from the TRAINING data, not testing data
	if args.weighted_loss:
		class_weights = dataset.class_weights.to(args.device)
		loss_f = torch.nn.CrossEntropyLoss(weight=class_weights)
	else:
		loss_f = torch.nn.CrossEntropyLoss()
	# Evaluate the model on the test dataset
	test_loss_acc, label_ids = evaluate_model(
		model=model,
		data=dataset.dataloaders['test'],
		loss_f=loss_f,
		device=args.device,
		non_blocking=args.non_blocking,
	)
	# Map the true (actual) and predicted label ids to their corresponding name
	true_labels = [id_to_label[lbl_id] for lbl_id in label_ids['actual']]
	pred_labels = [id_to_label[lbl_id] for lbl_id in label_ids['predicted']]
	# Get the metrics using the model outputs
	scores = get_scores(true_labels, pred_labels, class_labels)
	conf_mat = get_confusion_matrix(
		true_labels,
		pred_labels,
		class_labels,
		pfx=args.metric_pfx,
	)
	# Write the metrics to their respective files
	with open(args.scores_file, 'wt') as scores_file:
		scores_file.write(json.dumps(test_loss_acc))
		scores_file.write(json.dumps(scores))
	conf_mat.savefig(args.conf_mat_file)


if __name__ == '__main__':
    main()
