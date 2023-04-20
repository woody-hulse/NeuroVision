import model
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

import preprocessing
from preprocessing import get_behavioral_test, preprocess_behavioral_dict
import MRIModel
from MRIModel import VGG3DModel, VGGSlicedModel


def get_patientIDs(filepath, mri_dir, eeg_dir):
    """
    retrieve shared list of patientIDs
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    mri_dir         : mri dir name
    """

    mri_patientIDs = set(os.listdir(filepath + mri_dir))
    eeg_patientIDs = set(os.listdir(filepath + eeg_dir))
    if ".DS_Store" in mri_patientIDs:
        mri_patientIDs.remove(".DS_Store")
    patientIDs = list(mri_patientIDs.intersection(eeg_patientIDs))
    patientIDs = [patientID.replace(".npy", "") for patientID in patientIDs]
    patientIDs.sort()

    return patientIDs


def load_mri_data(filepath, mri_dir, patientIDs):
    """
    load preprocessed mri data
    filepath        : path to data directory
    mri_dir         : mri dir name
    patientIDs      : patientID list
    """

    print("loading mri data from", filepath, "...")
    mri_data = []
    for patientID in tqdm(patientIDs):
        patient_mri = np.load(filepath + mri_dir + patientID + ".npy")
        mri_data.append(patient_mri)
    mri_data = np.stack(mri_data)

    return mri_data


def load_eeg_data(filepath, eeg_dir, patientIDs):
    """
    load preprocessed eeg data
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    patientIDs      : patientID list
    """

    print("loading eeg data from", filepath, "...")
    eeg_data = []
    for patientID in tqdm(patientIDs):
        patient_eeg = np.load(filepath + eeg_dir + patientID + ".npy")
        eeg_data.append(patient_eeg)
    eeg_data = np.stack(eeg_data)

    return eeg_data


def load_behavioral_data(filepath, behavioral_dir, patientIDs):
    """
    load preprocessed behavioral data
    filepath        : path to data directory
    behavioral_dir  : behavioral dir name
    patientIDs      : patientID list
    """

    behavioral_path = filepath + behavioral_dir
    behavioral_test = ['cvlt', 'lps']
    behavioral_test_data = [preprocess_behavioral_dict(get_behavioral_test(behavioral_path, test)) for test in behavioral_test]

    print("loading behavioral data from", filepath, "...")
    behavioral_data = []
    for patientID in tqdm(patientIDs):
        behavioral_data.append([])
        for test_data in behavioral_test_data:
            behavioral_data[-1] += test_data[patientID]
        behavioral_data[-1] = np.array(behavioral_data[-1])
    behavioral_data = np.stack(behavioral_data)

    return behavioral_data


def load_data(filepath, mri_dir, eeg_dir, behavioral_dir, patientIDs):
    """
    load preprocessed data
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    mri_dir         : mri dir name
    behavioral_dir  : behavioral dir name
    return          : mri_data, eeg_data, behavioral_data
    """

    print()
    mri_data = load_mri_data(filepath, mri_dir, patientIDs)
    eeg_data = load_eeg_data(filepath, eeg_dir, patientIDs)
    behavioral_data = load_behavioral_data(filepath, behavioral_dir, patientIDs)
    print()

    return  mri_data, eeg_data, behavioral_data


def main(train=False, preprocess=True):
    if preprocess:
        preprocessing.preprocess(preprocessing.DATA_PATH, sync=False)
    
    patientIDs = get_patientIDs(preprocessing.DATA_PATH, preprocessing.MRI_RESULT_DIR, preprocessing.EEG_RESULT_DIR)
    mri_data, eeg_data, behavioral_data = load_data(
        preprocessing.DATA_PATH, 
        preprocessing.MRI_RESULT_DIR, 
        preprocessing.EEG_RESULT_DIR,
        preprocessing.BEHAVIORAL_DIR,
        patientIDs)
    mri_data = preprocessing.add_colorchannels(mri_data)

    num_samples = mri_data.shape[0]
    train_prop = 0.7
    num_train_samples = int(train_prop * num_samples)
    train_mri_data, test_mri_data = mri_data[:num_train_samples], mri_data[num_train_samples:]
    train_behavioral_data, test_behavioral_data = behavioral_data[:num_train_samples], behavioral_data[num_train_samples:]

    MRIModel.applyVGG(mri_data, patientIDs, downsampling_factor=4)

    """
    model = VGG3DModel(output_units=behavioral_data.shape[1])
    model.compile(
		optimizer=model.optimizer,
		loss=model.loss,
		metrics=[],
	)
    model.build(mri_data.shape)
    model.summary()
    model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=4, validation_data=(test_mri_data, test_behavioral_data))
    """


    """
    model = VGGSlicedModel(output_units=behavioral_data.shape[1])
    model.compile(
		optimizer=model.optimizer,
		loss=model.loss,
		metrics=[],
	)
    model.build(mri_data.shape)
    model.summary()
    model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=4, validation_data=(test_mri_data, test_behavioral_data))
    """


if __name__ == "__main__":
    os.system("clear")
    main(
        train       = True,
        preprocess  = True
        )