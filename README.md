# Medical Image Classification Using Deep Learning


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

<details>
  <summary>Retinal OCT Structure</summary>

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

</details>

<details>
  <summary>Medical MNIST Structure</summary>
  
```python
root
  ├── AbdomenCT
  ├── BreastMRI
  ├── CXR
  ├── ChestCT
  ├── Hand
  └── HeadCT
```

</details>

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
All available script arguments can be found in [cli_utils.py](/medical_mnist/cli_utils.py#L10). Some useful parameters to change/test with are: 

* `dataset_dir` <- Folder where the dataset is stored (`cli/dataset/` by default, see [**Downloading The Dataset**](https://github.com/lewisc4/MedicalMNIST/blob/main/README.md#downloading-the-dataset))
* `dataset_type` <- The type of dataset to use (`"retinal-oct"` (default) or `"medical-mnist"`)
* `output_dir` <- Directory to save the trained model to (created if it doesn't exist)
* `model_file` <- The name of the actual model `.pt` file being saved in `output_dir`
* `model_architecture` <- Model architecture to use for training (`"resnet-18"` (default), `"resnet-50"`, `"vgg-16"`, or `"alexnet"`)
* `use_pretrained`, `from_scratch` <- Dictates if pre-trained weights are used or not, respectively
* `percent_val` <- Percentage of the dataset to use as validation data
* `percent_test` <- Percentage of the dataset to use as test data (only for Medical MNIST, as Retinal OCT has a predefined test set)
* `num_workers` <- The number of workers to use in each (train/val/test) DataLoader
* `weighted_sampling` <- Whether to use weighted sampling in the training DataLoader
* `weighted_loss` <- Whether to provide class weights to the loss function or not
* `learning_rate` <- The external learning rate (used by the optimizer)
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay (used by the optimizer)
* `eval_every` <- How often (in number of training steps) to evaluate the model on validation data
* `num_epochs` <- Number of training epochs to use
* `wandb_project` <- The Weights & Biases project name (account not required)


### Example Usage
**To train a model, using a specified dataset folder named "dataset":**
- `python3 train.py --dataset_dir=dataset`

