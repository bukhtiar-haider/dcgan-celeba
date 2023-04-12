import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    """
    Generator model for GAN

    Parameters:
    latent_dim: dimension of the input noise vector
    num_filters: Multiplier to scale the number of filters in each layer | default = 64
    dropout_rate: dropout rate for dropout layers in each layer of the model | default = 0.4
    """
    def __init__(self, latent_dim, num_filters=64, dropout=0.4):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # Set up the layers in the model
        self.model = tf.keras.Sequential()

        # Add dense layer to start
        self.model.add(layers.Dense(4 * 4 * (num_filters * 16), use_bias=False, input_shape=(latent_dim,)))
        self.model.add(layers.Reshape((4, 4, 1024)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        # Add convolutional transpose layers to increase spatial dimensions
        self.model.add(layers.Conv2DTranspose(num_filters * 16, 7, strides=(2, 2), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2DTranspose(num_filters * 8, 5, strides=(2, 2), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2DTranspose(num_filters * 4, 4, strides=(2, 2), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2DTranspose(num_filters * 2, 3, strides=(2, 2), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        # Add final convolutional layer to output generated image
        self.model.add(layers.Conv2D(3, 3, padding='same', activation='tanh'))

    def build_model(self):
        """
        Builds the generator model.

        Parameters:
        latent_dim: dimension of the input noise vector
        """
        self.build((None, self.latent_dim))
        return self.model

    def call(self, inputs):
        """
        Passes input noise vector through the generator model.

        Parameters:
        inputs: input noise vector

        Returns:
        Generated image as output of the model
        """
        return self.model(inputs)
