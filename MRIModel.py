import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

import preprocessing
import random



def applyVGG(mri_data, patientIDs, downsampling_factor=2, save=True, path="../data/mri_vgg/"):
	"""
	applies pretrained vgg to input
	mri_data		: input of shape (res, res, res, 3)
	patientIDs		: list of patientIDs
	"""

	inputX = tf.transpose(mri_data, [1, 0, 2, 3, 4])
	inputY = tf.transpose(mri_data, [3, 0, 1, 2, 4])
	inputZ = tf.transpose(mri_data, [2, 0, 3, 1, 4])

	print("collecting VGG data ...")

	vgg = tf.keras.applications.VGG19(
			include_top=False,
			weights="imagenet",
			input_tensor=None,
			input_shape=mri_data.shape[2:],
			pooling=None,
		)
	
	print("passing MRI data through VGG ...")

	outputX, outputY, outputZ = [], [], []
	for i in tqdm(range(0, inputX.shape[0], downsampling_factor)):
		outputX.append(vgg(inputX[i]))
		outputY.append(vgg(inputY[i]))
		outputZ.append(vgg(inputZ[i]))
	outputX = tf.stack(outputX)
	outputY = tf.stack(outputY)
	outputZ = tf.stack(outputZ)

	output = tf.concat([outputX, outputY, outputZ], axis=0)
	output = tf.transpose(output, [1, 0, 2, 3, 4])

	if save:
		print("saving VGG output to", path, "...")
		with tqdm(total=len(patientIDs)) as pbar:
			for patientID, patient_output in zip(patientIDs, tf.unstack(output)):
				np.save(path + patientID, patient_output)
				pbar.update(1)

	return output


def loadVGG(path="../data/mri_vgg/"):
	"""
	loads saved vgg data
	"""
	patientIDs = os.listdir(path)
	if ".DS_Store" in patientIDs:
		patientIDs.remove(".DS_Store")

	print("loading VGG data from", path, "...")

	vgg_data = []
	for patientID in tqdm(patientIDs):
		vgg_data.append(np.load(path + patientID))
	vgg_data = np.stack(vgg_data)

	patientIDs = [patientID.replace(".npy", "") for patientID in patientIDs]

	return patientIDs, vgg_data
	



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