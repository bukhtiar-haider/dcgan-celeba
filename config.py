import tensorflow as tf

# DataLoader config
data_path = 'path\to\dir'
img_dims = (64,64)
batch_size = 128
split = 0.9
num_images = 1500

# Generator config
gen_latent_dim = 100
gen_learning_rate = 0.0002
gen_momentum = 0.5
gen_decay = 1e-4

# Discriminator config
disc_image_shape = (64,64,3)
disc_learning_rate = 0.0002
disc_momentum = 0.5
disc_decay = 1e-3

# Misc
gen = tf.random.Generator.from_seed(42)
epochs = 10

