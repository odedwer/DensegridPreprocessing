# written by Oded Wertheimer, 26/08/2020, based on Eden Gerber's Matlab function sync_eyetracker_samples_to_eeg
from copy import deepcopy
import numpy as np
import pandas as pd
from preprocess_utilities import *
from scipy.stats import pearsonr
import matplotlib
import samplerate as smp
from EyeLinkProcessor import *
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType


def find_common_sequential_elements(a, b):
    """
    find the common elements in sequential order
    :param a: 1d vector
    :param b: 1d vector
    :return: common elements, indices of common elements in a,  indices of common elements in b
    """
    a = np.asarray(a)
    b = np.asarray(b)
    min_len_arr, max_len_arr = (a, b) if a.size < b.size else (b, a)
    j = 0
    min_arr_idx, max_arr_idx = [], []
    for i in range(min_len_arr.size):
        if min_len_arr[i] == max_len_arr[j]:  # match on i,j
            min_arr_idx.append(i)
            max_arr_idx.append(j)
            j += 1
            continue
        isin = np.isin(max_len_arr[j:], min_len_arr[i])  # check if the element is in the rest of the array
        if np.sum(isin) > 0:  # if it is
            idx = np.argmax(isin)
            min_arr_idx.append(i)
            max_arr_idx.append(j + idx)
            j += idx + 1  # update to next sequential possible common element
        if j >= max_len_arr.size:
            break
    min_arr_idx = np.asarray(min_arr_idx)
    max_arr_idx = np.asarray(max_arr_idx)
    return (max_len_arr[max_arr_idx], min_arr_idx, max_arr_idx) if a.size < b.size else (
        max_len_arr[max_arr_idx], max_arr_idx, min_arr_idx)


# %% constant definitions
START_RECORD_TRIGGER = 254
STOP_RECORD_TRIGGER = 255
SMOOTHING_FACTOR = 10
ET_SAMPLING_RATE_CONST = 1000
#%% read EEG data
raws = read_bdf_files(preload=True)
raw: mne.io.Raw = mne.concatenate_raws(raws[1:])
# %% read ET data
et_p = EyeLinkProcessor("data/vis_S2.asc", ParserType.MONOCULAR_NO_VELOCITY,
                        SaccadeDetectorType.ENGBERT_AND_MERGENTHALER)
# %%
et_p.sync_to_raw(raw)
# %%
microsaccades = et_p.get_synced_microsaccades()
# %%
eeg_index = et_p._eeg_index
#%%
a = eeg_index[~np.isnan(eeg_index)]