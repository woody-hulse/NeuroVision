import tensorflow as tf
import nibabel as nib
import numpy as np
import os
import sys
import mne
import gzip
import shutil
import csv
import skimage
from tqdm import tqdm

"""

full dataset at:
http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html

create data directory outside of the NeuroVision repo:
    behavioral  - behavroral data from source
    eeg         - preprocessed data from source
    mri         - preprocessed data from mri
"""

DATA_PATH = "../data/"
MRI_DIR, MRI_RESULT_DIR = "mri/", "mri_preprocessed/"
EEG_DIR, EEG_RESULT_DIR = "eeg/", "eeg_preprocessed/"
BEHAVIORAL_DIR = "behavioral/"
VGG_DIR = "mri_vgg/"


BEHAVIORAL_TESTS = ["cvlt", "lps", "rwt", "tap-alertness", "tap-incompatibility", "tap-working", "tmt", "wst", "bisbas",
                    "cerq", "cope", "f-sozu", "fev", "lot-r", "mspss", "neo-ffi", "psq", "tas", "teique", "upps"]

BEHAVIORAL_COLUMNS = {
    "cvlt"                  : [4, 10, 11, 13],
    "lps"                   : [1],
    "rwt"                   : [8, 20],
    "tap-alertness"         : [5],
    "tap-incompatibility"   : [15],
    "tap-working"           : [1],
    "tmt"                   : [2],
    "wst"                   : [3],
    "bisbas"                : [1, 2, 3, 4],
    "cerq"                  : [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "cope"                  : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "f-sozu"                : [1, 2, 3, 4, 5],
    "fev"                   : [3],
    "lot-r"                 : [3],
    "mspss"                 : [4],
    "neo-ffi"               : [1, 2, 3, 4, 5],
    "psq"                   : [5],
    "tas"                   : [4],
    "teique"                : [1, 2, 3, 4, 5],
    "upps"                  : [1, 2, 3, 4],
    "yfas"                  : [9]
}

BEHAVIORAL_FILENAMES = {
    "cvlt"                  : "Cognitive_Test_Battery_LEMON/CVLT/CVLT.csv",
    "lps"                   : "Cognitive_Test_Battery_LEMON/LPS/LPS.csv",
    "rwt"                   : "Cognitive_Test_Battery_LEMON/RWT/RWT.csv",
    "tap-alertness"         : "Cognitive_Test_Battery_LEMON/TAP_Alertness/TAP-Alertness.csv",
    "tap-incompatibility"   : "Cognitive_Test_Battery_LEMON/TAP_Incompatibility/TAP-Incompatibility.csv",
    "tap-working"           : "Cognitive_Test_Battery_LEMON/TAP_Working_Memory/TAP-Working Memory.csv",
    "tmt"                   : "Cognitive_Test_Battery_LEMON/TMT/TMT.csv",
    "wst"                   : "Cognitive_Test_Battery_LEMON/WST/WST.csv",
    "bisbas"                : "Emotion_and_Personality_Test_Battery_LEMON/BISBAS.csv",
    "cerq"                  : "Emotion_and_Personality_Test_Battery_LEMON/CERQ.csv",
    "cope"                  : "Emotion_and_Personality_Test_Battery_LEMON/COPE.csv",
    "f-sozu"                : "Emotion_and_Personality_Test_Battery_LEMON/F-SozU_K-22.csv",
    "fev"                   : "Emotion_and_Personality_Test_Battery_LEMON/FEV.csv",
    "lot-r"                 : "Emotion_and_Personality_Test_Battery_LEMON/LOT-R.csv",
    "mspss"                 : "Emotion_and_Personality_Test_Battery_LEMON/MSPSS.csv",
    "neo-ffi"               : "Emotion_and_Personality_Test_Battery_LEMON/NEO_FFI.csv",
    "psq"                   : "Emotion_and_Personality_Test_Battery_LEMON/PSQ.csv",
    "tas"                   : "Emotion_and_Personality_Test_Battery_LEMON/TAS.csv",
    "teique"                : "Emotion_and_Personality_Test_Battery_LEMON/TEIQue-SF.csv",
    "upps"                  : "Emotion_and_Personality_Test_Battery_LEMON/UPPS.csv",
    "yfas"                  : "Emotion_and_Personality_Test_Battery_LEMON/YFAS.csv"
}


def applyACS(mri_data, patientIDs, save=True, path="../data/mri_acs/", downsampling_factor=2):
    """
    acs preprocesses mri data
    mri_data		: input of shape (num_patients, res, res, res)
    patientIDs		: list of patientIDs
    return			: acs data of shape (num_patients, res, res, 3)
    """

    print("computing ACS (axial-coronal-sagaittal) data")

    num_patients, resX, resY, resZ = mri_data.shape
    resX = int(resX / downsampling_factor)
    resY = int(resY / downsampling_factor)
    output = np.empty((num_patients, resX * resY, resZ, 3))

    for patient in tqdm(range(num_patients)):
        for i, x in enumerate(range(0, resX, downsampling_factor)):
            for j, y in enumerate(range(0, resY, downsampling_factor)):
                output[patient][i*resX + j, :, 0] = mri_data[patient, x, y, :]
                output[patient][i*resX + j, :, 1] = mri_data[patient, :, x, y]
                output[patient][i*resX + j, :, 2] = mri_data[patient, x, :, y]

    if save:
        print("saving ACS output to", path, "...")
        if not os.path.exists(path):
            os.mkdir(path)
        with tqdm(total=len(patientIDs)) as pbar:
            for patientID, patient_output in zip(patientIDs, tf.unstack(output)):
                np.save(path + patientID, patient_output)
                pbar.update(1)

    return output


def applyVGG(mri_data, patientIDs, downsampling_factor=2, save=True, path="../data/mri_vgg/"):
    """
    applies pretrained vgg to input
    mri_data		: input of shape (num_patients, res, res, res, 3)
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


def load_preprocessing(path="../data/mri_vgg/"):
    """
    loads saved data
    """
    patientIDs = os.listdir(path)
    if ".DS_Store" in patientIDs:
        patientIDs.remove(".DS_Store")

    print("loading data from", path, "...")

    data = []
    for patientID in tqdm(patientIDs):
        data.append(np.load(path + patientID))
    data = np.stack(data)

    patientIDs = [patientID.replace(".npy", "") for patientID in patientIDs]

    return patientIDs, data


def train_test_split(data, prop=0.7):
    """
    splits data into training and testing segments
    """
    cutoff = int(len(data) * prop)

    return data[:cutoff], data[cutoff:]


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


def resize_MRI(arr, shape=(224, 224)):
    """
    resizes MRI data

    arr             : preprocessed MRI numpy array
    shape           : compressed shape (224 x 224 for VGG compatibility)
    """

    new_MRI_arr = np.empty((arr.shape[0], shape[0], shape[1]))
    for i in range(arr.shape[1]):
        new_MRI_arr[i] = skimage.transform.resize(arr[i], shape, preserve_range=True)
    
    return new_MRI_arr


def add_colorchannels(mri_data):
    """
    adds color channels to mri data

    mri_data        : preprocessed MRI numpy array
    return          : expanded mri data
    """
    axes = list(range(len(mri_data.shape) + 1))
    return np.transpose(np.stack((mri_data, mri_data, mri_data)), axes=axes[1:] + [0])


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


def compress_preprocessed_EEG(filepath, patientID, num_channels=60, timesteps=1000):
    """
    compress a patient EEG from .set into a single numpy vector, save to EEG_RESULT_DIR

    filepath    : path to data folder
    patientid   : id of patient
    return      : none
    """

    with HiddenPrints():
        raw_mne = mne.io.read_raw_eeglab(filepath + patientID + "/" + patientID + "_EC.set", preload=True)
        raw_mne.set_eeg_reference(ref_channels='average', projection=False)
        
        raw_mne.filter(l_freq=0.5, h_freq=50, picks='eeg') # bandpass filter
        raw_data = raw_mne.get_data()
        
        # remove channels with zero standard deviation
        std_channels = np.std(raw_data, axis=1)
        nonzero_std_channels = np.where(std_channels > 0)[0]
        raw_data = raw_data[nonzero_std_channels]

        if len(nonzero_std_channels) < 60:
            return
        
        border = 1000
        data = np.empty((1, num_channels, timesteps))
        for i in range(num_channels):
            data[0, i, :] = raw_data[i, border : border + timesteps]

    if not os.path.exists(filepath + "../" + EEG_RESULT_DIR):
        os.mkdir(filepath + "../" + EEG_RESULT_DIR)
    np.save(filepath + "../" + EEG_RESULT_DIR + patientID + ".npy", data)


def get_behavioral_test(filepath, test):
    """
    extracts the behavioral information from patients

    filepath    : path to data folder
    sync        : preprocess only paired patient info
    return      : dictionary of behavioral scores for given metric
    """

    test = test.lower()
    filename = filepath + BEHAVIORAL_FILENAMES[test]
    with open(filename, 'r') as f:
        data = csv.reader(f)
        next(data, None)  # skip header
        cols = BEHAVIORAL_COLUMNS[test]
        behavioral_dict = {rows[0]:[rows[col] for col in cols] for rows in data}
    return behavioral_dict


def get_behavioral_column_names(filepath, tests):
    """
    gets the columns selected for behavioral data
    """

    colnames = []
    for test in tests:
        test = test.lower()
        filename = filepath + BEHAVIORAL_FILENAMES[test]
        with open(filename, 'r') as f:
            data = csv.reader(f)
            cols = np.array(next(data, None))
            colnames += list(cols[BEHAVIORAL_COLUMNS[test]])
    
    return colnames


def preprocess_behavioral_dict(behavioral_dict):
    """
    convert dict values (string) to standardized (percentile) floats
    filter unfilled rows (set to 0.5?)

    behavioral_dict     : dictionary to preprocess
    return              : processed behavioral dict
    """

    num_patients = len(list(behavioral_dict.values()))
    num_cols = len(list(behavioral_dict.values())[0])
    
    minima = [np.inf] * num_cols
    maxima = [-np.inf] * num_cols

    # find min and max
    empty_cells = {}
    for patient in behavioral_dict:
        for col in range(num_cols):
            try:
                behavioral_dict[patient][col] = float(behavioral_dict[patient][col])
                minima[col] = min(behavioral_dict[patient][col], minima[col])
                maxima[col] = max(behavioral_dict[patient][col], maxima[col])
            except ValueError:
                behavioral_dict[patient][col] = 0
                if patient not in empty_cells:
                    empty_cells[patient] = [col]
                else:
                    empty_cells[patient].append(col)
    
    # rescale values
    for patient in behavioral_dict:
        behavioral_dict[patient] = [(behavioral_dict[patient][i] - minima[i]) / (maxima[i] - minima[i]) for i in range(num_cols)]

    # do something with empty values
    for patient in empty_cells:
        for col in empty_cells[patient]:
            behavioral_dict[patient][col] = 0.5

    return behavioral_dict


def preprocess(filepath, sync=False):
    """
    preprocess and link all MRI and EEG data 

    filepath    : path to data folder
    sync        : preprocess only paired patient info
    return      : none
    """

    print("preprocessing data from", filepath, "...")
    
    mri_patientIDs = set(os.listdir(filepath + MRI_DIR))
    if ".DS_Store" in mri_patientIDs:
        mri_patientIDs.remove(".DS_Store")
    eeg_patientIDs = set(os.listdir(filepath + EEG_DIR))
    if ".DS_Store" in eeg_patientIDs:
        eeg_patientIDs.remove(".DS_Store")
        
    if sync:
        patientIDs = mri_patientIDs.intersection(eeg_patientIDs)
        for patientID in tqdm(patientIDs):
            compress_MRI(filepath + MRI_DIR, patientID)
            compress_preprocessed_EEG(filepath + EEG_DIR, patientID)
    else:
        print("preprocessing MRI data ...")
        for mri_patientID in tqdm(mri_patientIDs):
            compress_MRI(filepath + MRI_DIR, mri_patientID)
        print("preprocessing EEG data ...")
        for eeg_patientID in eeg_patientIDs:
            compress_preprocessed_EEG(filepath + EEG_DIR, eeg_patientID)


def main():
    """
    for testing
    """
    preprocess(DATA_PATH)

if __name__ == "__main__":
    main()




class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout