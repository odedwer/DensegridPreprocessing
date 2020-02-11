# %%
import os
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pandas import DataFrame
import pandas as pd
import seaborn as sn

import mne
from h5py import File


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


def read_bdf_files(preload=True):
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
            ret.append(mne.io.read_raw_bdf(os.path.join(dir_path, file_name), preload=preload))
    return ret


def add_bipolar_derivation(raw, ch_1, ch_2):
    """
    adds a channel to the given raw instance that is ch_1-ch_2
    :param raw: raw object to derive channel from
    :param ch_1: anode
    :param ch_2: cathode
    """
    raw = mne.set_bipolar_reference(raw, ch_1, ch_2, drop_refs=False)
    return raw


def process_epochs(trigger, epochs, notch_list=None, high_filter=30, low_filter=1, samp_rate=2048):
    """
    Gal Chen
    this function is responsible for the processing of existing 'epochs' object and adding the relevant filters
    most parameters are default but can be changed
    notch list is a list of amplitudes of line noise to be filtered ouy
    obligatory: the specific trigger we epoch by ("short words") and epochs object that was previously created
    """
    if notch_list is None:
        notch_list = [50]
    curr_epochs = epochs[trigger]
    filt_epochs = curr_epochs.copy()
    filt_epochs = mne.filter.notch_filter(filt_epochs, samp_rate, notch_list)
    filt_epochs.filter(l_freq=low_filter, h_freq=high_filter)
    return filt_epochs


def load_raws_from_mat(mat_filename, raws):
    """
    Reads a single .mat file to mne.io.Raw objects
    :param mat_filename: The name of the /mat file to load from.
        This function assumes that the .mat file contains only one variable.
        This variable should be a cell array containing the detrended data for each block
        in each cell.
    :param raws: The original raw objects that correspond to the data in each cell of the cell array in the given .mat file.
        These are needed for the info object in order to turn the arrays to mne.io.Raw objects
    :return: a list of raw objects that contain the data from the .mat file
    """
    print("starting....")
    arrays = list()
    with File(mat_filename) as mat_file:
        print("opened file...")
        for key in mat_file.keys():
            if key == '#refs#':
                continue
            data = mat_file[key]
            for arr in data:
                arrays.append(mat_file[arr[0]])
            for i, arr in enumerate(arrays):
                print("parsing block", str(i) + "...")
                arrays[i] = mne.io.RawArray(arr, raws[i].info)
    return arrays


def plot_correlations(ica, raw, components,
                  picks=['A1', 'Nose', 'RHEOG', 'LHEOG', 'RVEOGS', 'RVEOGI', 'M1', 'M2', 'LVEOGI']):
    """
       Reads ica and raw and prints correlation matrix of all ica components and electrodes listed.
       :param ica: the ica object
       :param raw: the raw data to check correlations with
       :param picks: the electrodes from raw we want to include in the matrix
       prints correlation matrix of all listed channels, and psds of components chosen
       """
    print("correlation matrix of electrodes and components...")
    data = {}
    data_electrodes = {}
    data_ica = {}
    # add raw channels
    for i in picks:
        data[i] = raw.get_data(picks=i)[0]
        data_electrodes[i] = raw.get_data(picks=i)[0]

    ica_raw = ica.get_sources(raw)
    set_type = {i: 'eeg' for i in ica_raw.ch_names}  #setting ica_raw
    ica_raw.set_channel_types(mapping=set_type)
    for i in list(components):
        data[ica_raw.ch_names[i]] = ica_raw.get_data(picks=i)[0]
        data_ica[ica_raw.ch_names[i]] = ica_raw.get_data(picks=i)[0]

    df = DataFrame(data)
    df_electrodes = DataFrame(data_electrodes)
    df["Mean eog"] = (df_electrodes['Nose']+
                                 df_electrodes['RHEOG']+
                                 df_electrodes['LHEOG']+
                                 df_electrodes['RVEOGS']+
                                 df_electrodes['RVEOGI']+
                                 df_electrodes['LVEOGI'])/6
    df_electrodes["Mean eog"] = df["Mean eog"]
    df_ica = DataFrame(data_ica)
    corr_matrix = df.corr().filter(df_electrodes.columns, axis=1).filter(df_ica.columns, axis=0)
    sn.heatmap(corr_matrix)
    ica_raw.plot_psd(fmin=0, fmax=250, picks=components, n_fft=10 * 2048)  # plot all psds of the 5 components


def annotate_breaks(raw, trig=254, samp_rate = 2048):
    """
       Reads raw and  annotates, for every start trigger, the parts from the trigger
       up to 1sec before the next one. RUN BEFORE ICA, and make sure that reject by annotation in ica is True.
       :param raw: the raw data to check correlations with
       :param trig: trigger to remove, default is 254
       :return: raw with annotated breaks
       """
    events = mne.find_events(raw, stim_channel="Status", mask=255)
    event_times = [i[0] / samp_rate for i in events if i[2] == 254]  #time of beginning of record
    next_trig_dur = [(events[i+1][0] / samp_rate - 2 - events[i][0]/samp_rate)
                             for i in range(len(events)-2) if events[i][2] == 254]  ##2 seconds before next (real) trigger after 254
    raw._annotations = mne.Annotations(event_times, next_trig_dur, 'BAD')
    return
