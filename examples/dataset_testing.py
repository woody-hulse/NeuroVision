import numpy as np
import mne

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

#
# data can be found at  : http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html
# paper reference       : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371893/
# 

def make_gif(data):
    for i in tqdm(range(data.shape[3])):
        fig, axs = plt.subplots(8, 8)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for j in range(8):
            for k in range(8):
                axs[j, k].imshow(data[:, :, j * 8 + k, i])
        plt.savefig("images/" + str(i).zfill(3) + ".png")
        plt.close()
    
    gif_name = "brain"
    file_list = glob.glob("images/*.png")
    list.sort(file_list, key=lambda x: int(x[7].split(".png")[0]))
    with open("image_list.txt", "w") as file:
        for item in file_list:
            file.write("%s\n" % item)

    os.system("convert @image_list.txt {}.gif".format(gif_name))


def make_mri_gif(data):
    for i in range(data.shape[0]):
        plt.imshow(data[i])
        plt.savefig("mri_images/" + str(i).zfill(3) + ".png")
        plt.close()

    gif_name = "mri-front"
    file_list = glob.glob("mri_images/*.png")
    list.sort(file_list, key=lambda x: int(x[11:].split(".png")[0]))
    with open("image_list.txt", "w") as file:
        for item in file_list:
            file.write("%s\n" % item)

    os.system("convert @image_list.txt {}.gif".format(gif_name))


def eeg_plot(raw_mne):
    print(raw_mne[0])

    for channel_index in range(2):
        raw_selection = raw_mne[channel_index, :]

        x = raw_selection[1]
        y = raw_selection[0].T
        plt.plot(x, y)
    plt.show()

def main():
    # test = nib.load("../data/mri/sub-032301/ses-01/func/sub-032301_ses-01_task-rest_acq-AP_run-01_bold.nii").get_fdata()
    # make_gif(test)

    mri_path = "../data/mri_preprocessed/sub-032301.npy"
    mri_data = np.load(mri_path)
    mri_data = np.transpose(mri_data, [2, 1, 0])
    make_mri_gif(mri_data)

    # file = "../data/eeg/sub-032301/RSEEG/sub-032301.vhdr"
    # raw_mne = mne.io.read_raw_brainvision(file)
    # eeg_plot(raw_mne)

if __name__ == "__main__":
    main()
