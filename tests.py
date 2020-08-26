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
# %% read EEG data
raws = read_bdf_files(preload=True)
raw: mne.io.Raw = mne.concatenate_raws(raws[1:])
# %% read ET data
et_p = EyeLinkProcessor("data/vis_S2.asc", ParserType.MONOCULAR_NO_VELOCITY,
                        SaccadeDetectorType.ENGBERT_AND_MERGENTHALER)
# %% get et data
et_samples = np.asarray(et_p._samples, dtype=np.int)
et_trigs = np.asarray(et_p._triggers, dtype=np.int)
et_trigs = et_trigs[np.where((et_trigs[:, 1] != 0) & (et_trigs[:, 0] <= np.max(et_samples[:, 0])))]
# %% get sampling rates and time diffs
eeg_sf = raw.info['sfreq']
et_sf = et_p._sf
et_timediff = int(et_samples[1, 0] - et_samples[0, 0])

# %% Reset the first timestamp of the ET data to 1 (or 2 if the sampling rate
first_ts = min(et_samples[0, 0], et_trigs[0, 0])
orig_sample_time = et_samples[:, 0]
orig_trig_time = et_trigs[:, 0]
et_samples[:, 0] += et_timediff - first_ts
et_trigs[:, 0] += et_timediff - first_ts

# %% Generate EEG event time course
eeg_data_trig_ch = np.squeeze((raw['Status'])[0])
eeg_data_trigs = mne.find_events(raw, mask=255, mask_type="and")
eeg_trigs = np.zeros_like(eeg_data_trig_ch, dtype=np.int)
eeg_trigs[eeg_data_trigs[:, 0]] = eeg_data_trigs[:, 2]

# %% split eeg to blocks based on 254 start record value
eeg_block_starts = np.where(eeg_trigs == START_RECORD_TRIGGER)[0]
eeg_blocks = np.split(eeg_trigs, eeg_block_starts)
eeg_blocks = [np.squeeze(block) for block in eeg_blocks if block[0] == START_RECORD_TRIGGER and np.sum(block) > 0]

# %% Generate ET event time course (unlike the EEG event time course, the
# event timestamps are in absolute time (ms) and not sample indexes, so
# this timecourse will also include epochs in which there was no recording).
et_trig_ch = np.zeros(et_samples[-1, 0])
et_trig_ch[et_trigs[:, 0]] = et_trigs[:, 1]

# %% split et to blocks based on 254 start record value and 255 stop
et_block_starts = np.hstack([np.where(et_trig_ch == START_RECORD_TRIGGER)[0], len(et_trig_ch)])
et_block_pause = np.where(et_trig_ch == STOP_RECORD_TRIGGER)[0]
et_recording_stop = np.hstack([np.where(np.diff(et_samples[:, 0]) > et_timediff)[0], et_samples[-1, 0]])
et_blocks = []
for i in range(et_block_starts.size):
    cur_start = et_block_starts[i]
    end_conditions = (et_block_pause > cur_start) & (
        1 if i + 1 == et_block_starts.size else et_block_pause <= et_block_starts[i + 1])
    possible_ends = np.where(end_conditions)[0]
    if possible_ends.size > 0:
        cur_end = et_block_pause[possible_ends[-1]]
        if np.sum(et_trig_ch[cur_start:cur_end]) > 1:
            et_blocks.append(et_trig_ch[cur_start:cur_end])

# %% Re-sample the EEG event blocks to ET sampling rate.
# Although the final result of the function will be the ET data re-sampled
# to the EEG time-course, this is first done by translating the latter to
# the ET time-course - so that each individual ET sample could be mapped to
# its corresponding EEG time point.
for i in range(len(eeg_blocks)):
    trig_sample_idx = np.where(eeg_blocks[i] != 0)[0]
    triggers = eeg_blocks[i][trig_sample_idx]
    new_block = np.zeros_like(eeg_blocks)
    new_trig_idx = np.ceil(trig_sample_idx * (eeg_sf / et_sf), dtype=np.int)
    new_block[new_trig_idx] = triggers
    eeg_blocks[i] = new_block

# %% Find the best match between EEG and ET blocks___
# This function can synchronize recordings where the EEG and ET can have a
# different number of blocks (e.g. if one recording started or stopped
# earlier than the other). This is done by going over all possible overlaps
# of the two block lists, and chosing the one with the highest total
# correlation between each pair of matched blocks.
#
# Create smoothed event time courses to decrease the correlations'
# sensitivity to event timing discrepancies. But first delete the first
# event in the block to prevent spuriously high correlations for blocks
# with only one event.

# smooth with convolution
smoothed_eeg_blocks = list(map(lambda block: np.convolve(block, np.ones(SMOOTHING_FACTOR), 'same'), eeg_blocks))
smoothed_et_blocks = list(map(lambda block: np.convolve(block, np.ones(SMOOTHING_FACTOR), 'same'), et_blocks))


# define function for map - returns correlation of eeg and et block
def get_block_correlation(eeg_block, et_block):
    et = et_block[:, 1]
    min_len = min(eeg_block.size, et.size)
    eeg = eeg_block[:min_len]
    et = et[:min_len]
    return pearsonr(eeg, et)


# finding the highest correlation for the first block gives us the offset between blocks
eeg_block = smoothed_eeg_blocks[0]
correlations = list(map(lambda et_block: get_block_correlation(eeg_block, et_block), smoothed_et_blocks))
offset = np.argmax(correlations)  # et_blocks[offset] corresponds to eeg_blocks[0]

matched_eeg_blocks = np.arange(len(eeg_blocks))
matched_et_blocks = np.arange(offset, len(et_blocks), 1)
eeg_block_starts = np.hstack([eeg_block_starts[matched_eeg_blocks], [eeg_data_trig_ch.size]])
et_block_starts = np.hstack([et_block_starts[matched_et_blocks], [et_samples[-1, 0]]])

# %% Correct sampling-rate discrepancies
# Due to different clocks, sampling rates after the initial re-sampling are
# still not identical, leading to cumulative discrepancies (can be on the
# order of 1 ms discrepancy per 10 seconds of recording).
# This is corrected by comparing the latency of matching events at the end
# of each block and re-sampling the EEG block once again.
resampling_factor = np.zeros(matched_eeg_blocks.size)
for i in range(resampling_factor.size):
    eeg_block = eeg_blocks[matched_eeg_blocks[i]]
    et_block = et_blocks[matched_et_blocks[i]]
    eeg_latencies = np.where(eeg_block != 0)
    et_latencies = np.where(et_block != 0)
    eeg_trigs = eeg_block[eeg_latencies]
    et_trigs = et_block[et_latencies]
    _, eeg_common_idx, et_common_idx = find_common_sequential_elements(eeg_trigs, et_trigs)


# %% resample to EEG freq
np_trigs_resampled_times = smp.resample(et_trigs[:, 0], raw.info['sfreq'] / et_p._sf, 'sinc_best')
np_trigs_resampled_trigs = smp.resample(et_trigs[:, 1], raw.info['sfreq'] / et_p._sf, 'zero_order_hold')
np_trigs_resampled = np.vstack([np_trigs_resampled_times, np_trigs_resampled_trigs]).T
np_trigs_resampled[:, 0] = np.round(np_trigs_resampled[:, 0])
np_trigs_resampled = np_trigs_resampled.astype(np.int)

# %%


# %%
max_len_arr[max_arr_idx]
