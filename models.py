import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from EEGModels import EEGNet


class InceptionACSModel(tf.keras.Model):

    """
    Inception-based MRI model
    """

    def __init__(self, input_shape=(256, 256, 3), output_units=1, output_activation="sigmoid", name="inception_acs"):

        super().__init__()

        self.inception = createInceptionModel(input_shape=input_shape, aux_output_units=output_units)

        self.head = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu", name=f"{name}_dense_1"),
            tf.keras.layers.Dropout(0.7, name=f"{name}_dropout_1"),
            tf.keras.layers.Dense(256, activation="relu", name=f"{name}_dense_2"),
            tf.keras.layers.Dropout(0.7, name=f"{name}_dropout_2"),
            tf.keras.layers.Dense(output_units, activation=output_activation, name=f"{name}_output")
        ]

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
      
    def call(self, x):
        x, aux_1, aux_2 = self.inception(x)
        for layer in self.head:
          x = layer(x)
        return x, aux_1, aux_2


class VGGACSModel(tf.keras.Model):
    
    """
    VGG-based MRI model
    """

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
    

class EEGModel(tf.keras.Model):

    """
    EEGNet-based EEG model
    """

    def __init__(self, output_units=2, output_activation="sigmoid", name="eegnet"):

        super().__init__()
        
        eegnet = tf.keras.models.load_model('data/weights/EEGNet-8-2-weights.h5').layers
        self.tail = []
        for layer in eegnet:
            if type(layer) == type(tf.keras.layers.Dense(10)):
                break
            layer.trainable = False
            self.tail.append(layer)

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


class NeuroVisionModel2(tf.keras.Model):
  
  """
  Inception-based NeuroVision model
  """

  def __init__(self, mri_input_shape=(), eeg_input_shape=(), output_units=2, concat_units=32, name="NeuroVision2"):

      super().__init__(name=name)

      self.model = createNeuroVision2(
        mri_input_shape=mri_input_shape, 
        output_units=output_units, 
        concat_units=concat_units, 
        eeg_model=EEGModel(
          output_units=concat_units,
          output_activation="sigmoid"
        ),
        eeg_input_shape=eeg_input_shape
        )

      self.loss = tf.keras.losses.MeanSquaredError()
      self.optimizer = tf.keras.optimizers.Adam(0.001)
    
  def call(self, x):
      return self.model(x)



class NeuroVisionModel(tf.keras.Model):

    """
    VGG-based NeuroVision model
    """

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

        super().__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_units, activation='softmax')

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.01)
    
    def call(self, data):
        return self.dense(self.flatten(data))
    

class VGG3DModel(tf.keras.Model):

    """
    *unused* 3D VGG-based model (see report for details)
    """

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

    """
    *unused* VGG sliced model attempt (see report for details)
    """

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


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    """
    adapted from google

    Szegedy, Christian, et al. “Going Deeper with Convolutions.” arXiv.Org, 17 Sept. 2014, arxiv.org/abs/1409.4842. 

    create a single inception module
    """

    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)
    
    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.concat([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

    
def createInceptionModel(input_shape, aux_output_units):

    """
    based on original Inception architecture, Google

    Szegedy, Christian, et al. “Going Deeper with Convolutions.” arXiv.Org, 17 Sept. 2014, arxiv.org/abs/1409.4842. 

    creates inception-based MRI model
    """
        
    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)

    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name='inception_4a')


    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(aux_output_units, activation='sigmoid', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4d')


    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(aux_output_units, activation='sigmoid', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5b')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    return tf.keras.Model(input_layer, [x, x1, x2], name='inception')


def createNeuroVision2(mri_input_shape, output_units, concat_units, eeg_model, eeg_input_shape):

    """
    based on original Inception architecture, Google

    Szegedy, Christian, et al. “Going Deeper with Convolutions.” arXiv.Org, 17 Sept. 2014, arxiv.org/abs/1409.4842. 

    creates inception-based joint MRI-EEG model
    """
        
    kernel_init = tf.keras.initializers.glorot_uniform()
    bias_init = tf.keras.initializers.Constant(value=0.2)

    eeg_input_layer = tf.keras.layers.Input(shape=eeg_input_shape)
    mri_input_layer = tf.keras.layers.Input(shape=mri_input_shape)

    eeg_out = eeg_model(eeg_input_layer)

    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(mri_input_layer)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name='inception_4a')


    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(concat_units, activation='sigmoid')(x1)
    x1 = tf.keras.layers.Concatenate(axis=1)([x1, eeg_out])
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
    x1 = tf.keras.layers.Dense(output_units, activation="sigmoid", name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4d')


    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(concat_units, activation="sigmoid")(x2)
    x2 = tf.keras.layers.Concatenate(axis=1)([x2, eeg_out])
    x2 = tf.keras.layers.Dense(64, activation='relu')(x2)
    x2 = tf.keras.layers.Dense(output_units, activation="sigmoid", name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')

    x = tf.keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5b')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(concat_units, activation="relu")(x)
    x = tf.keras.layers.Concatenate(axis=1)([x, eeg_out])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(output_units, activation="sigmoid", name="output")(x)

    return tf.keras.Model([eeg_input_layer, mri_input_layer], [x, x1, x2], name='inception_neurovision')

