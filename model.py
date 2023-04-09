import numpy as np
import tensorflow as tf

#
#   data overview
#       (x, y, slices) MRI input
#           want to run a 3d convolution on this structure
#       (n_features, timesteps) EEG input
#
#   architecture outline:
#
#       MRI CNN:
#           3d conv     ()
#           maxpooling  ()
#           3d conv     ()
#           maxpooling  ()
#           flatten     ()
#
#       EEG RNN
#           LSTM        ()
#           linear      ()
#
#       *combine in latent space*
#       linear
#       output linear
#

class Model(tf.keras.Model):

    """
    central model (combined CNN and RNN)
    """

    def __init__(self):

        super.__init__()

        # hyperparameters
        self.learning_rate = 1e-2
        self.batch_size = 2

        # CNN
        conv_filters = 48
        conv_shape = (3, 3, 3)
        conv_stride = (1, 1, 1)
        conv_padding = 'valid'
        pool_shape = (2, 2, 2)
        self.conv1 = tf.keras.layers.conv_3d(conv_shape, stride=conv_stride, padding=conv_padding)
        self.maxpool1 = tf.keras.layers.maxpool_3d(pool_shape, stride=conv_stride, padding=conv_padding)

        self.conv2 = tf.keras.layers.conv3d(conv_shape, stride=conv_stride, padding=conv_padding)
        self.maxpool2 = tf.keras.layers.maxpool_3d(pool_shape, stride=conv_stride, padding=conv_padding)

        self.flatten = tf.keras.layers.flatten()

    def call():
        pass


def main():
    pass


if __name__ == "__main__":
    main()