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

    BEHAVIORAL_DIR = filepath + "Cognitive_Test_Battery_LEMON/"

    behavioral_dict = {}
    filename = ""
    cols = []

    test = test.lower()
    if test == "cvlt":
        """
        1= test version of the CVLT
        2= number of correct  recalls after hearing the list for the first time
        3= number of correct  recalls after hearing the list for the fifth time
        4= Proactive Interference (Interference from previosly learned information)
        5= Retroactive Interference
        6= sum of all correct recalls from first until fifth trial
        7= sum of all correct recalls from first until fith trial with an age-gender-education correction
        8= number of correct recalls from the interference task (List B)
        9= number of correct recalls (short delay)
        10= number of correct recalls when category cues are presented (short delay)
        11= number of correct recalls after 20 minutes (long delay recall) 
        12= number of correct recalls when category cues are presented (long delay recall)
        13= delayed recognition memory performance
        14= overall repetitions
        15=overall intrusions
        16= comments
        """

        filename = "CVLT/CVLT.csv"
        cols = [4, 10, 11, 13]
        # notes:

    elif test == "lps":
        """
        LPS_1 = LPS raw data, how many symbol-rows did the participant process correctly
        LPS_2 = comments
        """
        filename = "LPS/LPS.csv"
        cols = [1]
        # notes: 

    elif test == "rwt":
        """
        S-words
        RWT_1 = how many s-words did the participant name during the first minute (according to rules)
        RWT_2 = percentile rank of correct words for the subtest 's-words' (1 minute)
        RWT_3 = how many repetitions during the first minute
        RWT_4 = how many rule breaks during the first minute
        RWT_5 = how many s-words did the participant name during the second minute (according to rules)
        RWT_6 = how many repetitions during the second minute
        RWT_7 = how many rule breaks during the second minute
        RWT_8 = how many s-words did the pp. name in total (according to rules); two minutes
        RWT_9 = percentile rank of correct words for the subtest 's-words' (2 minutes)
        RWT_10 = how many repetitions in total; two minutes
        RWT_11 = how many rule breaks in total; two minutes
        RWT_12 = comments (s-words)

        Animals
        RWT_13 = how many animals did the participant. name in the first minute (according to rules)
        RWT_14 = percentile rank of correct words for the subtest 'animal' (1 minute)
        RWT_15 = how many repetitions during the first minute (animals)
        RWT_16 = how many rule breaks during the first minute (animals)
        RWT_17 = how many animals did the pp. name during the second minute (according to rules)
        RWT_18 = how many repetitions during the second minute (animals)
        RWT_19 = how many rule breaks during the second minute (animals)
        RWT_20 = how many animals did the pp. name in total (according to rules); two minutes
        RWT_21 = percentile rank of correct words for the subtest 'animals' (2 minutes)
        RWT_22 = how many repetitions in total; two minutes (animals)
        RWT_23 = how many rule breaks during the second minute (animals)
        RWT_24 = comments (animals)
        """

        filename = "RWT/RWT.csv"
        cols = [8, 20]
        # notes: add cols 8 and 20?

    elif test == "tap":
        dict1 = get_behavioral_test(filepath, "tap-alertness")
        dict2 = get_behavioral_test(filepath, "tap-incompatibility")
        dict3 = get_behavioral_test(filepath, "tap-working")
        
        # notes: combine dicts instead of returning separately?
        return [dict1, dict2, dict3]
    
    elif test == "tap-alertness":
        """
        TAP_A_1 = reaction time medians for 1. round (no signal)
        TAP_A_2 = reaction time medians for 2. round (signal)
        TAP_A_3 = reaction time medians for 3. round (signal)
        TAP_A_4 = reaction time medians for 4. round (no signal)

        NO signal (aggregate scores)
        TAP_A_5 = mean reaction time (no signal)
        TAP_A_6 = median reaction time (no signal)
        TAP_A_7 = % (no signal)
        TAP_A_8 = standard deviations (no signal)
        TAP_A_9 = % (no signal)

        Signal (aggregate scores)
        TAP_A_10 = mean reaction time (signal)
        TAP_A_11 = median reaction time (signal)
        TAP_A_12 = % (signal)
        TAP_A_13 = standard deviations (signal)
        TAP_A_14 = % (signal)

        Phasic alertness
        TAP_A_15 = phasic alertness
        TAP_A_16 = % (phasic alertness)
        TAP_A_17 = comments
        """

        filename = "TAP_Alertness/TAP-Alertness.csv"
        cols = [5]

        # notes: i think for dependency reasons we only need to choose 1 (with/without signal)

    elif test == "tap-incompatibility":
        """
        Compatible stimuli
        TAP_I_1 = mean time for compatible stimuli
        TAP_I_2 = median time for compatible stimuli
        TAP_I_3 = %
        TAP_I_4 = standard deviations
        TAP_I_5 = %
        TAP_I_6 = how many errors did participant make during compatible stimuli presentation
        TAP_I_7 = %

        Incompatible stimuli
        TAP_I_8 = mean time for incompatible stimuli
        TAP_I_9 = median time for incompatible stimuli
        TAP_I_10 = %
        TAP_I_11 = standard deviations
        TAP_I_12 = %
        TAP_I_13 = how many errors did participant make during incompatible stimuli presentation
        TAP_I_14 = %

        Whole stimuli
        TAP_I_15 = mean time for whole stimuli presentation
        TAP_I_16 = median time for whole stimuli presentation
        TAP_I_17 = %
        TAP_I_18 = standard deviations
        TAP_I_19 = %
        TAP_I_20 = how many errors did participant make during whole stimuli presentation
        TAP_I_21 = %

        F-values
        TAP_I_22 = F-value_visual_field
        TAP_I_23 = %
        TAP_I_24 = F-value_hand
        TAP_I_25 = %
        TAP_I_26 = F-value_visual_field_x_hand
        TAP_I_27 = %
        TAP_I_28 = comments
        """

        filename = "TAP_Incompatibility/TAP-Incompatibility.csv"
        cols = [15]

        # notes:
    
    elif test == "tap-working":
        """
        TAP_WM_1 = mean reaction time for correct pressed bottoms (correct answers)
        TAP_WM_2 = reaction time median
        TAP_WM_3 = %
        TAP_WM_4 = standard deviations
        TAP_WM_5 = %
        TAP_WM_6 = how many correct matches
        TAP_WM_7 = how many incorrect matches (errors)
        TAP_WM_8 = %
        TAP_WM_9 = how many missed matches (omissions)
        TAP_WM_10 = %
        TAP_WM_11 = outliers
        TAP_WM_12 = comments
        """

        filename = "TAP_Working_Memory/TAP-Working Memory.csv"
        cols = [1]

        # notes: should really be wm2 / (wm6 / (wm6 + wm7 + wm9))

    elif test == "tmt":
        """
        TMT_1 = time it took to connect numbers (seconds. milliseconds)
        TMT_2 = brain functions based on performance for Trail A (Reitan & Wolfson, 1988)
        TMT_3 = how many errors did the pp. make
        TMT_4 = comments
        TMT_5 = time it took to connect numbers and letters (seconds. milliseconds)
        TMT_6 = brain functions based on performance for Trail B (Reitan & Wolfson, 1988)
        TMT_7 = how many errors did the pp. make
        TMT_8 = comments
        """ 

        filename = "TMT/TMT.csv"
        cols = [2]

        # notes: should probably sum 2 and 6

    elif test == "wst":
        """
        WST_1 = WST raw data; how many real words did the pp. recognize correctly
        WST_2 = z-scale
        WST_3 = IQ-scale
        WST_4 = Z-scale
        WST_5 = comments
        """

        filename = "WST/WST.csv"
        cols = [3]

        # notes: 

    else:
        print("behavioral test not configured :", test)
        return

    with open(BEHAVIORAL_DIR + filename, 'r') as fin:
        data = csv.reader(fin)
        next(data, None)  # skip header
        behavioral_dict = {rows[0]:[rows[col] for col in cols] for rows in data}
    return behavioral_dict


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