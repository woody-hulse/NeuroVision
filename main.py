import model
import numpy as np
import os

import preprocessing


def load_data(filepath, mri_dir, eeg_dir):
    """
    load preprocessed data
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    mri_dir         : mri dir name
    return          : mri_data, eeg_data
    """

    print("loading data from", filepath, "...")
    patientIDs = os.listdir(filepath + mri_dir)
    if ".DS_Store" in patientIDs:
        patientIDs.remove(".DS_Store")

    mri_data = []
    eeg_data = []
    for patientID in patientIDs:
        patient_mri = np.load(filepath + mri_dir + patientID)
        patient_eeg = np.load(filepath + eeg_dir + patientID)
        print(patientID, ":", "mri shape :", patient_mri.shape, "eeg shape :", patient_eeg.shape)
        mri_data.append(patient_mri)
        eeg_data.append(patient_eeg)

    return np.array(mri_data), np.array(eeg_data)


def main(preprocess=True):
    if preprocess:
        preprocessing.preprocess(preprocessing.DATA_PATH)
    
    mri_data, eeg_data = load_data(
        preprocessing.DATA_PATH, 
        preprocessing.MRI_RESULT_DIR, 
        preprocessing.EEG_RESULT_DIR)

    print(mri_data.shape, eeg_data.shape)


if __name__ == "__main__":
    main(
        preprocess  = False
        )