import tensorflow as tf


# Define the loss functions for the generator and discriminator models
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define BCE loss function for the Discriminator
@tf.function(reduce_retracing=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss   
    return total_loss, real_loss, fake_loss

# Define Generative Adverserial loss for the Generator
@tf.function(reduce_retracing=True)
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the Wasserstein loss for the Discriminator
@tf.function(reduce_retracing=True)
def discriminator_loss_wasserstein(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = fake_loss - real_loss
    
    return total_loss, real_loss, fake_loss

# Define the Wasserstein loss for the Generator
@tf.function(reduce_retracing=True)
def generator_loss_wasserstein(fake_output):
    return -tf.reduce_mean(fake_output)
