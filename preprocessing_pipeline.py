# %
# ##import
import numpy as np
from preprocess_utilities import *

# matplotlib.use('TkAgg')
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
# raw._data = raw._data/10**6
# good_status_ch_raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file that is not detrended: "))
# good_status_ch_raw.load_data()
# raw._data[262] = good_status_ch_raw._data[272]
# del good_status_ch_raw


# %% filter, drop bad channels, annotate breaks
# raw.load_data().filter(l_freq=.1, h_freq=None)  ##n
if input("auditory? (Y/N)") == 'Y':
    raw = annotate_breaks(raw)  # only for auditory
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
raw.drop_channels(["C26", "D3"])  # bridged/noisy channels we choose to remove ##n

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# %%
eog_map_dict = {'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog', 'RVEOGS': 'eog', 'RVEOGI': 'eog', 'M1': 'eog',
                'M2': 'eog', 'LVEOGI': 'eog'}
raw.set_channel_types(mapping=eog_map_dict)
# %%
# fit ica
# reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=150e-6, eog=300e-6)  # 150 μV
rej_step = .1  # in seconds
raw.filter(l_freq=1, h_freq=None)
# %%
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, max_iter=800)
# ica.fit(raw, reject_by_annotation=True, reject=reject_criteria)
ica.fit(raw, reject_by_annotation=True, tstep=rej_step, reject=reject_criteria)
ica.save('visual-hpf1-rejected100-ica.fif')
raw.save('visual-hpf1-rejected100-raw.fif')
# %%
# checking components is in running_script.py
ica.exclude = [0, 12, 22, 23, 46]  ####n all

# find which ICs match the EOG pattern

# %%

eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.plot_scores(eog_scores, title="Nose correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="LHEOG-RHEOG", threshold=2.5)
ica.plot_scores(eog_scores, title="Horizontal eye correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="RVEOGS-RVEOGI")
ica.plot_scores(eog_scores, title="Vertical eye correlations")

# barplot of ICA component "EOG match" scores

# %%
# exclude components
# raw = copy_raw  ##n
# ica.exclude = [0, 6, 7, 14, 16, 20, 11, 9]  # components we delete
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

events = mne.find_events(raw, stim_channel="Status", mask=255)
event_dict_aud = {'short_word': 12, 'long_word': 22}
event_dict_vis = {'short_face': 10, 'long_face': 20,
                  'short_anim': 12, 'long_anim': 22,
                  'short_obj': 14, 'long_obj': 24}  # ,
#                  'short_body': 16, 'long_body': 26}
raw.notch_filter([50, 100, 150])  # notch filter
filt_raw = raw.copy()  # save unfiltered copy for TF analysis ##n
filt_raw.load_data().filter(l_freq=1, h_freq=30)  ## epoch for ERPs ##n
filt_epochs = mne.Epochs(filt_raw, events, event_id=event_dict_vis, tmin=-0.2, tmax=1.6,
                         reject=reject_criteria, preload=True, reject_by_annotation=True)
# epoch raw data without filtering for TF analysis
epochs = mne.Epochs(raw, events, event_id=event_dict_vis, tmin=-0.2, tmax=1.6,
                    reject=reject_criteria, preload=True, reject_by_annotation=True)
# %%
ds_raw: mne.io.Raw = raw.copy()
ds_raw.resample(512)
chosen_electrode_epochs = mne.Epochs(ds_raw, events, event_id=event_dict_vis, tmin=-0.2, tmax=1.6,
                                     reject=reject_criteria, preload=True, reject_by_annotation=True)

