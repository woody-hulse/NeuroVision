import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from EEGModels import EEGNet

class VGG3DModel(tf.keras.Model):
    def __init__(self,
            input_shape=(256, 256), 	# input shape
            output_units=2, 			# number of units in output dense layer
            name="vgg3d"		    	# name of model
        ):

        super().__init__()

        self.conv1_1 = tf.keras.layers.Conv3D(64, 3)
        self.conv1_2 = tf.keras.layers.Conv3D(64, 3)
        self.pool1_1 = tf.keras.layers.MaxPool3D()

        self.conv2_1 = tf.keras.layers.Conv3D(128, 3)
        self.conv2_2 = tf.keras.layers.Conv3D(128, 3)
        self.pool2_1 = tf.keras.layers.MaxPool3D()
        
        self.conv3_1 = tf.keras.layers.Conv3D(256, 3)
        self.conv3_2 = tf.keras.layers.Conv3D(256, 3)
        self.conv3_3 = tf.keras.layers.Conv3D(256, 3)
        self.pool3_1 = tf.keras.layers.MaxPool3D()

        self.conv4_1 = tf.keras.layers.Conv3D(512, 3)
        self.conv4_2 = tf.keras.layers.Conv3D(512, 3)
        self.conv4_3 = tf.keras.layers.Conv3D(512, 3)
        self.pool4_1 = tf.keras.layers.MaxPool3D()

        self.conv5_1 = tf.keras.layers.Conv3D(512, 3)
        self.conv5_2 = tf.keras.layers.Conv3D(512, 3)
        self.conv5_3 = tf.keras.layers.Conv3D(512, 3)
        self.pool5_1 = tf.keras.layers.MaxPool3D()

        block1 = [self.conv1_1, self.conv1_2, self.pool1_1]
        block2 = [self.conv2_1, self.conv2_2, self.pool2_1]
        block3 = [self.conv3_1, self.conv3_2, self.conv3_3, self.pool3_1]
        block4 = [self.conv4_1, self.conv4_2, self.conv4_3, self.pool4_1]
        block5 = [self.conv5_1, self.conv5_2, self.conv5_3, self.pool5_1]

        self.blocks = [block1, block2, block3, block4, block5]

        self.head = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(output_units)
        ]

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

    
    def call(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        
        for layer in self.head:
            x = layer(x)
        
        return x


class VGGSlicedModel(tf.keras.Model):
    def __init__(self,
            input_shape=(224, 224, 3), 	# input shape
            output_units=2, 			# number of units in output dense layer
            freeze_vgg=True, 			# freeze vgg weights
            name="vgg_layered"		    # name of model
        ):

        """
        not functional
        """

        super().__init__()

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )
        self.flatten_layer = tf.keras.layers.Flatten(name=f"{name}_flatten")
        self.head = [
            tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense_1"),
            tf.keras.layers.Dropout(0.2, name=f"{name}_dropout_1"),
            tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense_2"),
            tf.keras.layers.Dropout(0.2, name=f"{name}_dropout_2")
        ]
        self.output_layer = tf.keras.layers.Dense(
            output_units, name=f"{name}_output_dense")

        if freeze_vgg:
            self.vgg.trainable = False

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)


    def call(self, input):
        slice_outputs = []
        for slice in input:
            slice_outputs.append(self.vgg(tf.keras.applications.vgg19.preprocess_input(slice)))
        slice_outputs = tf.stack(slice_outputs)
        
        x = self.flatten_layer(slice_outputs)
        for layer in self.head:
            x = layer(x)
        x = self.output_layer(x)

        return x
    


class VGGACSModel(tf.keras.Model):

    def __init__(self, input_shape=(256, 256, 3), output_units=1, freeze_vgg=True, output_activation="sigmoid", name="vgg_acs"):

        super().__init__()

        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )
        self.flatten_layer = tf.keras.layers.Flatten(name=f"{name}_flatten")
        self.head = [
            tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense_1"),
            tf.keras.layers.Dropout(0.2, name=f"{name}_dropout_1"),
            tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense_2"),
            tf.keras.layers.Dropout(0.2, name=f"{name}_dropout_2")
        ]

        self.output_layer = tf.keras.layers.Dense(output_units, activation=output_activation, name=f"{name}_output")

        if freeze_vgg:
            self.vgg.trainable = False

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)

    def call(self, input):
        x = self.vgg(input)
        x = self.flatten_layer(x)
        for layer in self.head:
            x = layer(x)
        x = self.output_layer(x)

        return x
    

class CenterModel():
    
    """
    Guesses 0.5
    """
    
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
    
    def call(self, data):
        return np.zeros((data.shape[0], self.shape[1])) + 0.5
    

class MeanModel():

    """
    Guesses the mean of the training data
    """

    def __init__(self, name, train_labels):
        self.name = name
        self.means = np.mean(train_labels, axis=0)

    def call(self, data):
        return np.stack([self.means for example in data])


class MedianModel():

    """
    Guesses the median of the training data
    """

    def __init__(self, name, train_labels):
        self.name = name
        self.medians = np.median(train_labels, axis=0)

    def call(self, data):
        return np.stack([self.medians for example in data])
    

class SimpleNN(tf.keras.Model):

    """
    dumb neural network
    """

    def __init__(self, output_units, name="control (1layerNN)"):

        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_units, activation='softmax')

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, data):
        return self.dense(self.flatten(data))
    

class EEGModel(tf.keras.Model):

    """
    Model is already configured for variable number of classes, channels, and samples
    If needed, will need to copy and replace parts of EEGModels.py to change classification head
    Below code is derived from comments in the original EEGNet.py file
    """

    def __init__(self, output_units=2, output_activation="sigmoid", name="eegnet"):

        super().__init__()
        
        eegnet = tf.keras.models.load_model('data/weights/EEGNet-8-2-weights.h5').layers
        self.tail = []
        for layer in eegnet:
            if type(layer) == type(tf.keras.layers.Dense(10)):
                break
            '''
            if type(layer) == type(tf.keras.layers.AveragePooling2D()):
                continue
                shape = layer.pool_size
                layer = tf.keras.layers.MaxPooling2D((2, 1))
                print(shape)
            if type(layer) == type(tf.keras.layers.Conv2D(1, (1, 1))):
                filters = layer.filters
                kernel_size = (layer.kernel_size[1], layer.kernel_size[0])
                padding = layer.padding
                use_bias = layer.use_bias
                layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=padding, use_bias=use_bias)
            '''
            layer.trainable = False
            self.tail.append(layer)

        '''
        self.tail = [
            tf.keras.layers.Conv2D(8, (1, 64), padding = 'same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D((input_shape[1], 1), use_bias=False, depth_multiplier=2, 
                                            depthwise_constraint=tf.keras.constraints.max_norm(1.)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('elu'),
            tf.keras.layers.AveragePooling2D((1, 4)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.SeparableConv2D(16, (1, 16),
                                        use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('elu'),
            tf.keras.layers.AveragePooling2D((1, 8)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
        ]
        '''

        self.head = [
            tf.keras.layers.Dense(20, kernel_regularizer='l2', activation="leaky_relu", name=f"{name}_dense_1"),
            tf.keras.layers.Dropout(0.2),
        ]

        self.output_layer = tf.keras.layers.Dense(output_units, activation=output_activation, name=f"{name}_output")

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, x):
        for layer in self.tail:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        x = self.output_layer(x)
        
        return x



class NeuroVisionModel(tf.keras.Model):

    def __init__(self, mri_input_shape=(), output_units=2, name="NeuroVision"):

        super().__init__(name=name)

        self.eegmodel = EEGModel(output_units=20, output_activation="softmax")
        self.mrimodel = VGGACSModel(input_shape=mri_input_shape, output_units=20, output_activation="softmax")

        self.head = [
            tf.keras.layers.Dense(output_units)
        ]

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, data):
        eeg_data, mri_data = data
        eegmodel_out = self.eegmodel(eeg_data)
        mrimodel_out = self.mrimodel(mri_data)
        
        x = tf.concat([eegmodel_out, mrimodel_out], axis=1)
        for layer in self.head:
            x = layer(x)
        
        return x
