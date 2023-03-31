# Medical Image Classification Using Deep Learning


## Project Overview
In this project, multiple CNN variants are trained to classify medical images from the [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) dataset. Each image is a 64 $\times$ 64 grayscale image and belongs to one of six classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, and HeadCT.


## Environment Setup
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules, follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the [root directory](https://github.com/lewisc4/MedicalMNIST), where the [`setup.py`](/setup.py) file is located.
3. Install the [`medical_mnist`](/medical_mnist) module and all dependencies by running the following command: `pip install -e .` (required python modules are in [`requirements.txt`](/requirements.txt)).

### Downloading the Dataset
The dataset can be downloaded from [here](https://www.kaggle.com/datasets/andrewmvd/medical-mnist).

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
The available hyperparameters are stored in [`utils.py`](/medical_mnist/utils.py).

### Example Usage
***To train a model using the defaul arguments:***
- `python3 train.py`

