import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def denormalize_images(images):
    """
    Denormalize images from [-1, 1] to [0, 255] and convert to RGB mode.

    Args:
        images (list): A list of images in the range [-1, 1].

    Returns:
        list: A list of denormalized and converted images.
    """
    denormalized_images = []
    for img in images:
        # Remove channel dimension and scale pixel values to [0, 255]
        np_img = np.squeeze((img + 1.0) * 127.5)
        # Convert to PIL Image and convert to RGB mode
        pil_img = Image.fromarray(np_img.astype(np.uint8))
        rgb_img = pil_img.convert('RGB')
        denormalized_images.append(rgb_img)
    return denormalized_images

def generate_and_save_images(generator, epoch, fixed_noise, output_dir):
    """
    Generate and save a grid of images from a fixed noise vector using the given generator.

    Parameters:
    generator: The generator model.
    epoch: The current epoch number.
    fixed_noise: A fixed noise vector for generating the images.
    output_dir: The directory to save the generated images.
    """
    # Generate images
    generated_images = generator(fixed_noise, training=False)
    generated_images = denormalize_images(generated_images)

    # Create a grid of images and save it
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap=None)
        ax.axis('off')
    cwd = os.getcwd()  # get current working directory
    parent_dir = os.path.dirname(os.path.abspath(cwd))  # get parent directory

    # create output directory in parent directory
    os.makedirs(os.path.join(parent_dir, output_dir), exist_ok=True)

    plt.savefig(os.path.join(parent_dir, output_dir, 'generated_images_{:04d}.jpg'.format(epoch)))
    plt.close(fig)

def save_model_parameters(generator, discriminator, epoch, output_dir):
    """
    Save the parameters of the generator and discriminator models.

    Parameters:
    generator: The generator model.
    discriminator: The discriminator model.
    epoch: The current epoch number.
    output_dir: The directory to save the model parameters.
    """
    cwd = os.getcwd()  # get current working directory
    parent_dir = os.path.dirname(os.path.abspath(cwd))  # get parent directory

    # create output directory in parent directory
    os.makedirs(os.path.join(parent_dir, output_dir), exist_ok=True)

    generator.save(os.path.join(parent_dir, output_dir, 'generator_epoch_{}.h5'.format(epoch + 1)))
    discriminator.save(os.path.join(parent_dir, output_dir, 'discriminator_epoch_{}.h5'.format(epoch + 1)))

def check_gpu():
    """
    Check if GPU is available and being used for computation.

    Returns:
    A boolean indicating whether a GPU is available and being used for computation.
    """
    print('Checking GPU:')
    devices = tf.config.list_physical_devices()
    gpu_available = False
    for device in devices:
        print(device.name)
        if 'GPU' in device.name:
            gpu_available = True
            print('GPU found', device.name)
            print('Is device available:', tf.config.list_physical_devices('GPU'))
            print('Is device built with CUDA:', tf.test.is_built_with_cuda())