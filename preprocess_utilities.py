# %%
import os
import sys
import numpy as np
import matplotlib
import mne
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import pickle


def save_data(obj, filename):
    """
    Saves data to be loaded from disk using pickle
    :param obj: The object to save
    :param filename: The name of the file in which the object is saved
    :return: None
    """
    print("Saving", filename)
    with open(filename, 'wb') as save_file:
        pickle.dump(obj, save_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    """
    Load data from given file using pickle
    :param filename: The name of the file to load from
    :return: The loaded object
    """
    print("Loading", filename)
    with open(filename, 'rb') as save_file:
        return pickle.load(save_file)


def read_bdf_files():
    """
    Reads bdf file from disk. If there are several files, reads them all to the same raw object.
    GUI will open to choose a file from a folder. Any file in that folder that has the same name of the
    chosen file up to the last _ in the filename will be added to the final raw object, by
    order of names (lexicographic)
    :return: The loaded raw object (preloaded)
    """
    # TODO: don't concatenate, return list
    # get file path using GUI
    Tk().withdraw()
    filename = askopenfilename(title="Please choose ")
    filenames = [filename]
    dir_path = os.path.dirname(filename)
    just_filename = os.path.basename(filename)
    count = 0
    for cur_filename in os.listdir(dir_path):
        if just_filename[:just_filename.rfind('_')] == cur_filename[:cur_filename.rfind('_')] and cur_filename[
                                                                                                  -3:] == "bdf":
            filenames.append(cur_filename)
            count += 1
    print("Found", count, " BDF files.")
    # open file as RAW object, preload data into memory
    ret = []
    for file_name in sorted(filenames):
        if ret is not None:
            ret.append(mne.io.read_raw_bdf(os.path.join(dir_path, file_name), preload=True))
    return ret


def add_bipolar_derivation(raw, ch_1, ch_2):
    """
    adds a channel to the given raw instance that is ch_1-ch_2
    """
    mne.set_bipolar_reference(raw, ch_1, ch_2)


def process_epochs(trigger, epochs, notch_list=[50], highFilter=30, lowFilter=1, samp_rate=2048):
    """
    Gal Chen
    this function is responsible for the processing of existing 'epochs' object and adding the relevant filters
    most parameters are default but can be changed
    notcch list is a list of amplitudes of line noise to be filtered ouy
    obligatory: the specific trigger we epoch by ("short words") and epochs object that was previously created
    """
    curr_epochs = epochs[trigger]
    filt_epochs = curr_epochs.copy()
    filt_epochs = mne.filter.notch_filter(filt_epochs, samp_rate, notch_list)
    filt_epochs.filter(l_freq=lowFilter, h_freq=highFilter)
    return filt_epochs

