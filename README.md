# celebA GAN

CelebA GAN: A Generative Adversarial Network trained on CelebA dataset.

## Description

This project implements a Generative Adversarial Network (GAN) trained on the CelebA dataset. The GAN is used to generate high-resolution images of human faces.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- TensorFlow
- Keras
- PIL
- matplotlib
- A GPU (recommended)

### Dependency installation

It is recommended to install these dependencies inside a python venv.

You may do this using conda:

```powershell
# Create a new virtual environment with Conda
conda create --name myenv python=3.9

# Activate the virtual environment
conda activate myenv
```

### 1. Install Tensorflow & Keras
```powershell
# Install TensorFlow and Keras
conda install tensorflow keras
```

### 2. Install Pillow
```powershell
# Install Pillow
conda install Pillow
```

### 3. Install matplotlib
```powershell
# Install Matplotlib
conda install matplotlib
```

### Usage

The program expects you to have the dataset already - you may download it from the [CelebA publishers](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing).

Use the *config.py* to set the hyperparameters, epochs, and path to the image dataset.

Images generated from a fixed 16 dimensional noise vector will be saved in the *images* folder.

Set *save_interval* in the *Trainer* class to specify how many epochs to wait until model parameters are saved in the *model_param* folder.
### Training Progress

<img src="https://github.com/bukhtiar-haider/dcgan-celeba/blob/master/images/output.gif" width="400" height="400" />

### Interpolated Images

<table>
  <tr>
    <td><img src="https://github.com/bukhtiar-haider/dcgan-celeba/blob/master/images/solo1.gif" width="200"></td>
    <td><img src="https://github.com/bukhtiar-haider/dcgan-celeba/blob/master/images/solo2.gif" width="200"></td>
    <td><img src="https://github.com/bukhtiar-haider/dcgan-celeba/blob/master/images/solo3.gif" width="200"></td>
    <td><img src="https://github.com/bukhtiar-haider/dcgan-celeba/blob/master/images/solo4.gif" width="200"></td>
  </tr>
</table>
