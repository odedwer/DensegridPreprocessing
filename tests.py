import numpy as np
import pandas as pd
from preprocess_utilities import *
from scipy.stats import pearsonr
import matplotlib
import samplerate as smp
from EyeLinkProcessor import *
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType

# %% constant definitions
START_RECORD_TRIGGER = 254
STOP_RECORD_TRIGGER = 255
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
eeg_blocks = [block for block in eeg_blocks if block[0] == START_RECORD_TRIGGER and np.sum(block) > 0]

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

# %%
i = 0
cur_start = et_block_starts[i]
end_conditions = (et_block_pause > cur_start) & (et_block_pause <= et_block_starts[i + 1])
cur_end = et_block_pause[np.where(end_conditions)[0][-1]]
# %%


# %%
et_blocks = [block for block in et_blocks if block[0] == START_RECORD_TRIGGER and np.sum(block) > 0]

# %% resample to EEG freq
np_trigs_resampled_times = smp.resample(et_trigs[:, 0], raw.info['sfreq'] / et_p._sf, 'sinc_best')
np_trigs_resampled_trigs = smp.resample(et_trigs[:, 1], raw.info['sfreq'] / et_p._sf, 'zero_order_hold')
np_trigs_resampled = np.vstack([np_trigs_resampled_times, np_trigs_resampled_trigs]).T
np_trigs_resampled[:, 0] = np.round(np_trigs_resampled[:, 0])
np_trigs_resampled = np_trigs_resampled.astype(np.int)

# %% divide into recording blocks


et_blocks = np.split(et_trigs, block_starts)
et_blocks = [block for block in et_blocks if block[0, 1] == START_RECORD_TRIGGER]
# %%
first_time = et_trigs[0, 0]
for i in range(len(et_blocks)):
    et_block = et_blocks[i]
    min_time, max_time = np.min(et_block[:, 0]), np.max(et_block[:, 0])
    new_block = np.zeros((max_time - min_time + 1, 2), dtype=np.int)
    new_block[et_block[:, 0] - min_time, 1] = et_block[:, 1]
    new_block[:, 0] = np.arange(min_time, max_time + 1, 1) - first_time + 1
    block_stop = np.where(new_block[:, 0] == 255)[0]
    if block_stop == []:
        new_block = new_block[:block_stop[0]]
    et_blocks[i] = new_block

# %% divide EEG into recording blocks
eeg_data_trig_ch = np.squeeze((raw['Status'])[0])
eeg_data_trigs = mne.find_events(raw, mask=255, mask_type="and")
trig_array = np.zeros_like(eeg_data_trig_ch, dtype=np.int)
trig_array[eeg_data_trigs[:, 0]] = eeg_data_trigs[:, 2]
eeg_block_starts = np.where(trig_array == START_RECORD_TRIGGER)[0]
eeg_blocks = np.split(trig_array, eeg_block_starts)
eeg_blocks = [block for block in eeg_blocks if block[0] == START_RECORD_TRIGGER]
# %%
et_to_eeg_block_mapping = np.zeros(len(eeg_blocks))
# %%
et_block = et_blocks[3]
eeg_block = eeg_blocks[0]
# %%

pearsonr(eeg_block, et_block[:, 1])
# %%
for i, eeg_block in enumerate(eeg_blocks):
    correlations = np.asarray(map(lambda et_block: pearsonr(eeg_block[:, 1], et_block[:, 1]), et_blocks))
    et_to_eeg_block_mapping[i] = np.argmax(correlations)
# %%
