import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from EEGModels import EEGNet

"""
EEG layout:
EEGNet tail (frozen, loaded from .h5), 
linear layers as classifier

"""

def EEGModel():

    #replace below with number of classes, channels, samples from data once preprocessing begins working
    # n_classes = 0
    # n_channels = 0
    # n_samples = 204
    # inputs = None
    # labels = None


    """
    Model is already configured for variable number of classes, channels, and samples
    If needed, will need to copy and replace parts of EEGModels.py to change classification head
    Below code is derived from comments in the original EEGNet.py file
    """
    # model = EEGNet(nb_classes = n_classes, Chans = n_channels, Samples = n_samples)
    # model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy, optimizer = tf.keras.optimizers.Adam, metrics = tf.keras.metrics.AUC)
    # fitted = model.fit(x = inputs, y = labels, epochs = 30)
    # predicted = model.predict(x = inputs)

    """
    Load model from .h5, build independent classification head and train as appropriate
    """
    #load model weights and freeze all tail layers
    eegnet = tf.keras.models.load_model('../EEGNet-8-2-weights.h5')
    for layer in eegnet.layers:
        layer.trainable = False

    model = tf.keras.models.Sequential([
        eegnet,
        tf.keras.layers.Dense(128, activation="leaky_relu"),
        tf.keras.layers.Dense(64, activation="leaky_relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation="leaky_relu"),
        tf.keras.layers.Dense(12, activation="softmax"),
    ])

    return model

if __name__ == "__main__":
    EEGModel()