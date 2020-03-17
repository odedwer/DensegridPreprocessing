# %
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib
matplotlib.use('Qt5Agg')

# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files(preload=True)
# detrended_raws = load_raws_from_mat('detrended_ord10_10s_window.mat', raws)
# concatenate to one raw file
raw = mne.concatenate_raws(raws)
copy_raw = raw.copy()  # make a copy before adding the new channel

# %% in case of existing raw file, like detrended:
raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file: "))
raw.load_data()

# %% drop bad channels, annotate breaks
raw.plot(n_channels=32, duration=30)  #to see data and mark bad channels and segments
#%%
if input("auditory? (Y/N)") == 'Y':
    raw = annotate_breaks(raw)  # only for auditory
#raw.drop_channels(["C26", "D3"])  # bridged/noisy channels we choose to remove ##n
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
raw.set_eeg_reference()

# %%
eog_map_dict = {'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog', 'RVEOGS': 'eog', 'RVEOGI': 'eog', 'M1': 'eog',
                'M2': 'eog', 'LVEOGI': 'eog'}
raw.set_channel_types(mapping=eog_map_dict)
# %%
# reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=250e-6, eog=300e-6)  # 150 μV
rej_step = .1  # in seconds

# fit ica
# %%
ica = mne.preprocessing.ICA(n_components=.90, random_state=97, max_iter=800)
# ica.fit(raw, reject_by_annotation=True, reject=reject_criteria)
ica.fit(epochs, reject_by_annotation=True)
ica.save('SavedResults/S3/visual-detrended-s3-ica.fif')
#raw.save('visual-detrended-s2-rejected100-raw.fif')

# %%
ica = mne.preprocessing.read_ica(input())
# checking components is in running_script.py
ica.exclude = [2,10]

# find which ICs match the EOG pattern

# %%

eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.plot_scores(eog_scores, title="Nose correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="LHEOG-RHEOG", threshold=2.5)
ica.plot_scores(eog_scores, title="Horizontal eye correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="RVEOGS-RVEOGI")
ica.plot_scores(eog_scores, title="Vertical eye correlations")

# barplot of ICA component "EOG match" scores
#%%
ica.apply(raw)

# %% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers

# event names
# NO_RESP_TRIAL_START_CODE = 200
# RESP_TRIAL_START_CODE = 201
# SHORT_FACE_STIM_ONSET_CODE = 10
# LONG_FACE_STIM_ONSET_CODE = 20
# SHORT_ANIMAL_STIM_ONSET_CODE = 12
# LONG_ANIMAL_STIM_ONSET_CODE = 22
# SHORT_OBJECT_STIM_ONSET_CODE = 14
# LONG_OBJECT_STIM_ONSET_CODE = 24
# SHORT_BODY_STIM_ONSET_CODE = 16
# LONG_BODY_STIM_ONSET_CODE = 26
# SHORT_FACE_STIM_OFFSET_CODE = 11
# LONG_FACE_STIM_OFFSET_CODE = 21
# SHORT_ANIMAL_STIM_OFFSET_CODE = 13
# LONG_ANIMAL_STIM_OFFSET_CODE = 23
# SHORT_OBJECT_STIM_OFFSET_CODE = 15
# LONG_OBJECT_STIM_OFFSET_CODE = 25
# SHORT_BODY_STIM_OFFSET_CODE = 17
# LONG_BODY_STIM_OFFSET_CODE = 27

events = mne.find_events(raw, stim_channel="Status", mask=255,min_duration=2/2048)
event_dict_aud = {'short_word': 12, 'long_word': 22}
event_dict_vis = {'short_face': 10, 'long_face': 20,
                  'short_anim': 12, 'long_anim': 22,
                  'short_obj': 14, 'long_obj': 24,
                  'short_body': 16, 'long_body': 26}
raw.notch_filter([50, 100, 150])  # notch filter
# raw_filt = raw.copy().filter(l_freq=1, h_freq=30)  # performing fikltering on copy of raw data, not on raw itself or epochs
# epoch raw data without filtering for TF analysis
epochs = mne.Epochs(raw, events, event_id=event_dict_vis,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject=reject_criteria,
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)
# filt_epochs = mne.Epochs(raw_filt, events, event_id=event_dict_aud, tmin=-0.2, tmax=1.6,
#                    reject=reject_criteria, preload=True, reject_by_annotation=True)

# %%
ds_epochs = epochs.copy().resample(512)

raw.plot()
