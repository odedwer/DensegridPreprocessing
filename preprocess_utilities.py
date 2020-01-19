# %%
import os
import sys
import numpy as np
import matplotlib
import mne
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import pickle

from scipy.io import loadmat


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
    Reads bdf file from disk. If there are several files, reads them all to different raw objects.
    GUI will open to choose a file from a folder. Any file in that folder that has the same name of the
    chosen file up to the last _ in the filename will be added opened as a raw object, by
    order of names (lexicographic)
    :return: List of the raw objects (preloaded)
    """
    # TODO: don't concatenate, return list
    # get file path using GUI
    Tk().withdraw()
    filename = askopenfilename(title="Please choose ")
    filenames = []
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


def load_mat_files_to_raws(mat_filenames, data_variable_names, raws):
    """
    loads EEG data into mne.Raw objects from .mat files
    :param mat_filenames: list, str
        The names of the .mat files from which to load EEG data. Each .mat should have 1 EEG data matrix
    :param data_variable_names: list, str
        The names of the variables in each .mat files containing the EEG data
    :param raws: list, mne.io.Raw
        raw object that holds the correct info for the data. can be obtained by reading the same BDF used in matlab
        into raw object. preload false for this.
    :return: list of raw objects loaded from
    """
    # sanity checks and input checks
    if type(raws) is not list:
        raws = [raws]
    if type(mat_filenames) is not list:
        mat_filenames = [mat_filenames]
    if type(raws) is not list:
        data_variable_names = [data_variable_names]
    if len(mat_filenames) != len(data_variable_names):
        print("Incompatible lengths! mat_filenames and data_variable_names should have the same length!",
              file=sys.stderr)
        return
    if len(mat_filenames) != len(raws):
        print("Incompatible lengths! mat_filenames and raws should have the same length!",
              file=sys.stderr)
        return
    if len(raws) != len(data_variable_names):
        print("Incompatible lengths! raws and data_variable_names should have the same length!",
              file=sys.stderr)
        return

    # actual code starts from here
    ret = list()
    for i, mat_filename in enumerate(mat_filenames):
        if not os.path.exists(mat_filename) and not os.path.exists(mat_filename + ".mat"):
            print("File " + mat_filename + " can't be found. Please enter a relative path to the file.",
                  file=sys.stderr)
            continue
        print("loading", mat_filename + "...")
        mat_file: dict = loadmat(mat_filename)
        if data_variable_names[i] not in mat_file:
            print("File ", mat_filename, " doesn't contain a variable named", data_variable_names[i],
                  ". Please try again and enter the name of the variable containing the EEG data.", file=sys.stderr)
            continue
        print("converting to raw...")
        ret.append(mne.io.RawArray(np.array(mat_file[data_variable_names[i]]).T, raws[i].info))
        print("added to result in index", len(ret) - 1)
    return ret
