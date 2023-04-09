import nibabel as nib
import numpy as np
import os
import mne
import gzip
import shutil
from tqdm import tqdm

"""

create data directory outside of the NeuroVision repo:
    behavioral  - behavroral data from source
    eeg         - preprocessed data from source
    mri         - preprocessed data from mri
"""

DATA_PATH = "../data/"
MRI_DIR, MRI_RESULT_DIR = "mri/", "mri_preprocessed/"
EEG_DIR, EEG_RESULT_DIR = "eeg/", "eeg_preprocessed/"

def compress_MRI(filepath, patientID):
    """
    compress a patient MRI into a single 3d numpy vector, save to MRI_RESULT_DIR

    filepath    : path to data folder
    patientid   : id of patient
    return      : none
    """

    MRI_PATH = "/anat/"
    nii_files = [f for f in os.listdir(filepath + patientID + MRI_PATH) if f.endswith('.nii')]
    gz_file = [f for f in os.listdir(filepath + patientID + MRI_PATH) if f.endswith('.gz')][0]
    if len(nii_files) == 0:
        with gzip.open(filepath + patientID + MRI_PATH + gz_file, 'rb') as f_in:
            with open(filepath + patientID + MRI_PATH + gz_file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        nii_file = gz_file[:-3]
    else:
        nii_file = nii_files[0]

    data = nib.load(filepath + patientID + MRI_PATH + nii_file).get_fdata()
    if not os.path.exists(filepath + "../" + MRI_RESULT_DIR):
        os.mkdir(filepath + "../" + MRI_RESULT_DIR)
    np.save(filepath + "../" + MRI_RESULT_DIR + patientID + ".npy", data)


def reconfigure_VHDR(filepath, patientID):
    """
    replaces the real patient ID with the patient ID embedding

    filepath    : path to data folder
    patientID   : embedded patient ID
    """

    with open(filepath + patientID + "/RSEEG/tmp", "w") as outfile:
        with open(filepath + patientID + "/RSEEG/" + patientID + ".vhdr", "r") as vhdr:
            for line in vhdr.readlines():
                tokens = line.split("=")
                if tokens[0] == "DataFile":
                    outfile.write("DataFile=" + patientID + ".eeg")
                elif tokens[0] == "MarkerFile":
                    outfile.write("MarkerFile=" + patientID + ".vmrk")
                else:
                    outfile.write(line)


def compress_raw_EEG(filepath, patientID):
    """
    compress a patient EEG from raw data into a single numpy vector, save to EEG_RESULT_DIR

    filepath    : path to data folder
    patientid   : id of patient
    return      : none
    """

    reconfigure_VHDR(filepath, patientID)
    raw_mne = mne.io.read_raw_brainvision(filepath + patientID + "/RSEEG/" + patientID + ".vhdr")
    meg_eeg = np.array(raw_mne[0][0][0])
    if not os.path.exists(filepath + "../" + EEG_RESULT_DIR):
        os.mkdir(filepath + "../" + EEG_RESULT_DIR)
    np.save(filepath + "../" + EEG_RESULT_DIR + patientID + ".npy", meg_eeg)


def compress_preprocessed_EEG(filepath, patientID):
    """
    compress a patient EEG from .set into a single numpy vector, save to EEG_RESULT_DIR

    filepath    : path to data folder
    patientid   : id of patient
    return      : none
    """
    raw_mne = mne.io.read_raw_eeglab(filepath + patientID + "/" + patientID + "_EC.set")
    print(raw_mne[:].shape)
    eeg = np.array(raw_mne[0][0][0])

    trim = 10000
    border = int((len(eeg) - trim) / 2)
    if not os.path.exists(filepath + "../" + EEG_RESULT_DIR):
        os.mkdir(filepath + "../" + EEG_RESULT_DIR)
    np.save(filepath + "../" + EEG_RESULT_DIR + patientID + ".npy", eeg[border:border + trim])

def preprocess(filepath):
    """
    preprocess and link all MRI and EEG data 

    filepath    : path to data folder
    return      : none
    """

    print("preprocessing data from", filepath, "...")
    
    mri_patientIDs = set(os.listdir(filepath + MRI_DIR))
    eeg_patientIDs = set(os.listdir(filepath + EEG_DIR))
    patientIDs = mri_patientIDs.intersection(eeg_patientIDs)
    patientIDs.remove(".DS_Store")
    for patientID in patientIDs:
        compress_MRI(filepath + MRI_DIR, patientID)
        compress_preprocessed_EEG(filepath + EEG_DIR, patientID)

def main():
    """
    for testing
    """
    preprocess(DATA_PATH)

if __name__ == "__main__":
    main()