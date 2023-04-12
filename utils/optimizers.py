import tensorflow as tf
from config import gen_learning_rate, gen_momentum, gen_decay, disc_learning_rate, disc_momentum, disc_decay

# Define optimizer for the generator
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_learning_rate, beta_1 = gen_momentum, decay=gen_decay)

# Define the optimizer for the discriminator
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_learning_rate, beta_1 = disc_momentum, decay=disc_decay)