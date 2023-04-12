from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.losses import generator_loss, discriminator_loss
from utils.optimizers import gen_optimizer, disc_optimizer
from utils.utils import denormalize_images, generate_and_save_images, save_model_parameters


class Trainer:
    def __init__(self, gan, train_data, val_data, batch_size, epochs, rng):
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = rng
        self.save_interval = 1
        self.generator, self.discriminator = gan
        self.generator_optimizer = gen_optimizer
        self.discriminator_optimizer = disc_optimizer
        self.fixed_noise = tf.random.normal(shape=(16, 100))

    def validate_gan(self, dataset) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        strategy = tf.distribute.get_strategy()  # default strategy
        with strategy.scope():
            disc_losses, real_losses, fake_losses = tf.cast(0, tf.float16), tf.cast(0, tf.float16), tf.cast(0, tf.float16)
            for real_images in dataset:
                noise = self.rng.normal(shape=(self.batch_size, 100))
                fake_images = self.generator.call(noise, training=False)

                real_output = self.discriminator.call(real_images, training=False)
                fake_output = self.discriminator.call(fake_images, training=False)
                disc_loss, real_loss, fake_loss = discriminator_loss(real_output, fake_output)

                disc_losses += disc_loss
                real_losses += real_loss
                fake_losses += fake_loss

            num_total = tf.cast(len(dataset), tf.float16)
            avg_disc_loss = disc_losses / num_total
            avg_real_loss = real_losses / num_total
            avg_fake_loss = fake_losses / num_total

            return tf.cast(avg_disc_loss, tf.float16), tf.cast(avg_real_loss, tf.float16), tf.cast(avg_fake_loss, tf.float16)

    def train_step(self, real_images):
        noise = self.rng.normal(shape=(self.batch_size, 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator.call(noise, training=True)

            real_output = self.discriminator.call(real_images, training=True)
            fake_output = self.discriminator.call(fake_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss, real_loss, fake_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss, real_loss, fake_loss, gen_loss

    def train(self):
        
        # Enable mixed precision training
        strategy = tf.distribute.get_strategy()  # default strategy
        self.train_data = strategy.experimental_distribute_dataset(self.train_data)
        self.val_data = strategy.experimental_distribute_dataset(self.val_data)

        for epoch in range(self.epochs):
            disc_losses, real_losses, fake_losses, gen_losses = tf.cast(0, tf.float16), tf.cast(0, tf.float16), tf.cast(0, tf.float16), tf.cast(0, tf.float16)

            for real_images in self.train_data:
                disc_loss, real_loss, fake_loss, gen_loss = self.train_step(real_images)
                disc_losses += disc_loss
                real_losses += real_loss
                fake_losses += fake_loss
                gen_losses += gen_loss

            num_total = tf.cast(len(self.train_data), tf.float16)
            avg_disc_loss = tf.cast((disc_losses / num_total), tf.float16)
            avg_real_loss = tf.cast((real_losses / num_total), tf.float16)
            avg_fake_loss = tf.cast((fake_losses / num_total), tf.float16)
            avg_gen_loss = tf.cast((gen_losses / num_total), tf.float16)

            disc_val_loss, real_val_loss, fake_val_loss = self.validate_gan(self.val_data)

            print(f"Epoch {epoch+1}/{self.epochs} - G: {avg_gen_loss:.4f} - D: {avg_disc_loss:.4f} - DVal: {disc_val_loss:.4f} - Real: {avg_real_loss:.4f} - RVal: {real_val_loss:.4f} - Fake: {avg_fake_loss:.4f} - FVal: {fake_val_loss:.4f}")
        
            generate_and_save_images(self.generator, epoch, self.fixed_noise, 'images')

            # Save the model
            if (epoch + 1) % self.save_interval == 0:
                save_model_parameters(self.generator, self.discriminator, epoch, 'model_params')
