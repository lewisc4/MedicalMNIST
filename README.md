# Medical Image Classification Using Deep Learning

## Index
 1. [Project Overview](https://github.com/lewisc4/MedicalMNIST#project-overview)  
 2. [Environment Setup](https://github.com/lewisc4/MedicalMNIST#environment-setup)  
  2.1 [Package Installation](https://github.com/lewisc4/MedicalMNIST#package-installation)  
  2.2 [Downloading the Datasets](https://github.com/lewisc4/MedicalMNIST#downloading-the-datasets)  
  2.3 [GPU-related Requirements/Installation](https://github.com/lewisc4/MedicalMNIST#gpu-related-requirementsinstallations)
 3. [Training](https://github.com/lewisc4/MedicalMNIST#training)  
  3.1 [Hyperparameters](https://github.com/lewisc4/MedicalMNIST#hyperparameters)  
  3.2 [Example Usage](https://github.com/lewisc4/MedicalMNIST#hyperparameters)
 4. [Evaluation](https://github.com/lewisc4/MedicalMNIST#evaluation)  
  4.1 [Hyperparameters](https://github.com/lewisc4/MedicalMNIST#hyperparameters-1)  
  4.2 [Example Usage](https://github.com/lewisc4/MedicalMNIST#example-usage-1)
 5. [Demonstration](https://github.com/lewisc4/MedicalMNIST#demonstration)  
  5.1 [Hyperparameters](https://github.com/lewisc4/MedicalMNIST#hyperparameters-2)  
  5.2 [Example Usage](https://github.com/lewisc4/MedicalMNIST#example-usage-2)
  

## Project Overview
This project provides the functionality to train multiple CNN architectures to perform medical image classification on two datasets. The first is the  the [Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) dataset, which contains 84,495 grayscale OCT images of human retinas, each belonging to one of four disease classes: CNV, DNE, DRUSEN, and NORMAL (i.e., no disease). The second dataset is the [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) dataset, which consists of 58,954 grayscale medical images, each belonging to one of six classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, and HeadCT.

Results and notebook examples are only reported on the [Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) dataset. For more information on downloading the datasets, please see the [Downloading the Datasets](https://github.com/lewisc4/MedicalMNIST#downloading-the-datasets) section. For more information regarding the datasets themselves, please see their linked Kaggle pages.

## Environment Setup
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules, follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the [root directory](https://github.com/lewisc4/MedicalMNIST), where the [`setup.py`](/setup.py) file is located.
3. Install the [`medical_mnist`](/medical_mnist) module and all dependencies by running: `pip install -e .` (required python modules are in [`requirements.txt`](/requirements.txt)).

### Downloading the Datasets
Both the [Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) and [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) datasets can be downloaded from Kaggle. For the [Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018), Version 2 should be downloaded. Once downloaded, they should be unzipped (you can delete the `.zip` files after). Then, the path to the unzipped folder's root (it can be a relative path) can then be provided to the `dataset_dir` argument for scripts run from the [`cli/`](/cli) directory (by default, it is assumed that the datasets will be downloaded under [`cli/`](/cli).). See the [**Training**](https://github.com/lewisc4/MedicalMNIST/blob/main/README.md#training) section for examples. The expected directory structure for each dataset is (where `root` is the name of each dataset's root folder):

<table align="center">
<tr>
<th>Retinal OCT Structure</th>
<th>Mecical MNIST Structure</th>
</tr>
<tr>
<td valign="top">

```python
root
  ├── test
  │   ├── CNV
  │   ├── DME
  │   ├── DRUSEN
  │   └── NORMAL
  ├── train
  │   ├── CNV
  │   ├── DME
  │   ├── DRUSEN
  │   └── NORMAL
  └── val
      ├── CNV
      ├── DME
      ├── DRUSEN
      └── NORMAL
```

</td>
<td valign="top">

```python
root
  ├── AbdomenCT
  ├── BreastMRI
  ├── CXR
  ├── ChestCT
  ├── Hand
  └── HeadCT
```

</td>
</tr>
</table>

### GPU-related Requirements/Installations
Follow the steps below to ensure your GPU and all relevant libraries are up to date and in good standing.

1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
2. If your CUDA version is below 10.2, don't use this server
3. If your CUDA version is below 11, run `pip install torch`
4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above commands (usually this helps).
6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above.


## Training
The [`train.py`](/cli/train.py) script is used to train a model via CLI arguments.

### Hyperparameters
All available script arguments can be found in [cli_utils.py](/medical_mnist/cli_utils.py#L13). Some useful parameters to change/test with are: 

| Argument/Parameter     | Description                                                                                                                                                               |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset_dir`        | Dataset's root folder, `cli/dataset/` by default (see [**Downloading The Dataset**](https://github.com/lewisc4/MedicalMNIST/blob/main/README.md#downloading-the-dataset)) |
| `--dataset_type`       | Type of dataset being used [`retinal-oct` (default), `medical-mnist`]                                                                                                     |
| `--output_dir`         | Directory to save the trained model to (created if it doesn't exist)                                                                                                      |
| `--model_file`         | The name of the `.pt` model file to save in `output_dir`                                                                                                                  |
| `--model_architecture` | Type of model to train [`resnet-18` (default), `resnet-50`, `vgg-16`, `alexnet`]                                                                                          |
| `--use_pretrained`     | Train using pre-trained weights (default)                                                                                                                                 |
| `--from_scratch`       | Train from scratch (no pre-trained weights)                                                                                                                               |
| `--percent_val`        | % of data to use for validation                                                                                                                                           |
| `--percent_test`       | % of data to use for testing (only for `medical-mnist`, as `retinal-oct` has a fixed test set)                                                                            |
| `--num_workers`        | Number of workers to use in DataLoader(s)                                                                                                                                 |
| `--weighted_sampling`  | Whether to use weighted sampling in the training DataLoader or not                                                                                                        |
| `--weighted_loss`      | Whether to provide class weights to the loss function or not                                                                                                              |
| `--learning_rate`      | External learning rate used by the optimizer                                                                                                                              |
| `--device`             | Device to train on, defaults to `cuda` if available, otherwise `cpu`                                                                                                      |
| `--batch_size`         | Batch size to use in DataLoader(s)                                                                                                                                        |
| `--weight_decay`       | External weight decay used by the optimizer                                                                                                                               |
| `--eval_every`         | How often, in number of training steps, to evaluate the model                                                                                                             |
| `--num_epochs`         | Number of training epochs                                                                                                                                                 |
| `--wandb_project`      | Weights & Biases project name (account not required)                                                                                                                      |
| `--upload_model`       | Whether to upload the model to Weights & Biases or not                                                                                                                    |

### Example Usage
For the below examples, assume we have downloaded the Retinal OCT dataset with a root folder named `oct_data`, as described in the [Downloading the Datasets](https://github.com/lewisc4/MedicalMNIST#downloading-the-datasets) section. Also assume the commands are run from the [`cli/`](/cli) directory.

```bash
# To train a VGG-16 model on the Retinal OCT dataset, saving it to a file named `vgg_model.pt`:
$ python3 train.py --model_architecture=vgg-16 --model_file=vgg_model --dataset_type=retinal-oct --dataset_dir=oct_data

# To do the same as above, but with a ResNet-50 model:
$ python3 train.py --model_architecture=resnet-50 --model_file=resnet_model --dataset_type=retinal-oct --dataset_dir=oct_data

# To train a VGG-16 model for 50 epochs, with a batch size of 256 and learning rate of 0.0005:
python3 train.py --model_architecture=vgg-16 --num_epochs=50 --batch_size=256 --learning_rate=5e-4  --dataset_type=retinal-oct --dataset_dir=oct_data

# To train a VGG-16 model from scratch, using a class-weighted loss function:
python3 train.py --model_architecture=vgg-16 --from_scratch --weighted_loss --dataset_type=retinal-oct --dataset_dir=oct_data
```



## Evaluation
The [`evaluations.py`](/cli/evaluations.py) script is used to evaluate a model on test data via CLI arguments.

### Hyperparameters
All available script arguments can be found in [cli_utils.py](/medical_mnist/cli_utils.py#L13). Some useful parameters to change/test with are: 

| Argument/Parameter     | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| `--dataset_dir`        | Root folder of the dataset to evaluate                          |
| `--dataset_type`       | Type of dataset to evaluate with                                |
| `--output_dir`         | Directory storing the model to evaluate                         |
| `--model_file`         | Filename of the model we want to evaluate                       |
| `--model_architecture` | Architecture of the trained model we want to evaluate           |
| `--metric_dir`         | Directory to save metric files to (created if it doesn't exist) |
| `--scores_file`        | The filename to save scores to (accuracy, recall, etc.)         |
| `--conf_mat_file`      | The filename of the confusion matrix figure                     |


### Example Usage
Assume we are evaluating models that were trained on the Retinal OCT dataset, which has a root folder named `oct_data`, as described in the [Downloading the Datasets](https://github.com/lewisc4/MedicalMNIST#downloading-the-datasets) section. Also assume the commands are run from the [`cli/`](/cli) directory.

```bash
# To evaluate a VGG-16 model that was saved to `vgg_model.pt`
python3 evaluations.py --model_architecture=vgg-16 --model_file=vgg_model --scores_file=vgg_scores --conf_mat_file=vgg_conf_mat --dataset_type=retinal-oct --dataset_dir=oct_data

# To do the same as above, but using a ResNet-50 model that was saved to `resnet_model.pt`
python3 evaluations.py --model_architecture=resnet-50 --model_file=resnet_model --scores_file=resnet_scores --conf_mat_file=resnet_conf_mat --dataset_type=retinal-oct --dataset_dir=oct_data
```



## Demonstration
The [`demo.py`](/cli/demo.py) script is used to demonstrate a model (using [gradio](https://gradio.app/)) via CLI arguments.

### Hyperparameters
All available script arguments can be found in [cli_utils.py](/medical_mnist/cli_utils.py#L13). Some useful parameters to change/test with are: 

| Argument/Parameter     | Description                                                                      |
|------------------------|----------------------------------------------------------------------------------|
| `--dataset_dir`        | Root folder of the dataset the model was trained on                              |
| `--dataset_type`       | Type of dataset the model was trained on                                         |
| `--output_dir`         | Directory storing the model to demo                                              |
| `--model_file`         | Filename of the model we want to demo                                            |
| `--model_architecture` | Architecture of the trained model we want to demo                                |
| `--demo_gradcam`       | Outputs Grad-CAM results, instead of the model's top-3 predictions (the default) |


### Example Usage
Assume we are evaluating models that were trained on the Retinal OCT dataset, which has a root folder named `oct_data`, as described in the [Downloading the Datasets](https://github.com/lewisc4/MedicalMNIST#downloading-the-datasets) section. Also assume the commands are run from the [`cli/`](/cli) directory.

```bash
# To demo a VGG-16 model that was saved to `vgg_model.pt`
# Here, the input is an image and the output is the top-3 classes it most likely belongs to
python3 demo.py --model_architecture=vgg-16 --model_file=vgg_model --dataset_type=retinal-oct --dataset_dir=oct_data

# To do the same as above, but using a ResNet-50 model that was saved to `resnet_model.pt`
python3 evaluations.py --model_architecture=resnet-50 --model_file=resnet_model --dataset_type=retinal-oct --dataset_dir=oct_data

# To demo a VGG-16 model that was saved to `vgg_model.pt`, using Grad-CAM
# Here, the input is an image and the output is Grad-CAM's output, overlayed on the input image
python3 demo.py --model_architecture=vgg-16 --model_file=vgg_model --demo_gradcam --dataset_type=retinal-oct --dataset_dir=oct_data
```
