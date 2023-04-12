import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class Discriminator(tf.keras.Model):
    """
    Discriminator model for GAN

    Parameters:
    input_shape: shape of input image
    filters_multiplier: Multiplier to scale the number of filters in each layer | default = 64
    dropout_rate: dropout rate for dropout layers in each layer of the model | default = 0.5
    """
    def __init__(self, input_shape, num_filters=64, dropout=0.5):
        super(Discriminator, self).__init__()

        self.in_shape = input_shape

        # Set up the layers in the model
        self.model = tf.keras.Sequential()

        # Add convolutional layers to reduce spatial dimensions
        self.model.add(layers.Conv2D(num_filters, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape,
                                     kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2D(num_filters*2, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                                     kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2D(num_filters*4, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                                     kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2D(num_filters*8, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                                     kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Conv2D(num_filters*8, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                                     kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(dropout))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))

    def build_model(self):
        """
        Builds the generator model.

        Parameters:
        latent_dim: dimension of the input noise vector
        """
        self.build((None,self.in_shape[0],self.in_shape[1],self.in_shape[2]))
        return self.model

    def call(self, inputs):
        """
        Passes an image into the Discriminator.

        Parameters:
        inputs: input image

        Returns:
        Unactivated output for image being fake or real
        """
        return self.model(inputs)
