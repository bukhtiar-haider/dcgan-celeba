import sys
import os

# Set the logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the directory of the current file
dir_path = os.path.dirname(os.path.realpath(__file__))
# Append the directory to the system path
sys.path.append(dir_path)

import logging
from config import *
from utils.utils import *
from utils.losses import *
from Trainer import Trainer
from keras import mixed_precision
from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.DataLoader import DataLoader
from utils.optimizers import gen_optimizer, disc_optimizer

logging.getLogger('tensorflow').setLevel(logging.ERROR)
mixed_precision.set_global_policy('mixed_float16')
check_gpu()

# Load Data
data_loader = DataLoader(data_path, img_dims, batch_size, split)
data_loader.load_data(num_images)
data_dict = data_loader.get_data()
train_data = data_dict['train']
val_data = data_dict['val']

# Instantiate Generator
G = Generator(latent_dim=gen_latent_dim)
G = G.build_model()

#Instantiate Discriminator
D = Discriminator(input_shape=disc_image_shape)
D = D.build_model()

trainer = Trainer((G, D), train_data, val_data, batch_size, epochs, gen)
trainer.train()