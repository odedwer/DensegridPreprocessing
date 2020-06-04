# %
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib

matplotlib.use('Qt5Agg')

# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files(preload=True)
# concatenate to one raw file
raw = mne.concatenate_raws(raws)
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
copy_raw = raw.copy()  # make a copy before adding the new channel
raw = raw.resample(512,n_jobs=4)

raw.filter(h_freq=None, l_freq=1, n_jobs=4)

# %%
raw = set_reg_eog(raw)
# %% drop bad channels, annotate bad intervals
plot_all_channels_var(raw, max_val=1e-5, threshold=5e-10)  # max value for visualization in case of large value
raw.plot(n_channels=30, duration=30)  # to see data and mark bad channels
raw.info['bads'] = ["C26","D3","E13"]

# %%
raw = annotate_bads_auto(raw, reject_criteria=150e-6, jump_criteria=1e-4)
# %% plot again to see annotations and mark missed noise/jumps
raw.plot(n_channels=30, duration=30)  # to see data and mark bad  segments

# %%
print("total time annotated as bad: ", round(sum(raw._annotations.duration), 2))
# %%
if input("auditory? (Y/N)") == 'Y':
    raw = annotate_breaks(raw)  # only for auditory

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
raw.set_eeg_reference()
# %%
# reject bad intervals based on pek to peak in ICA** - make sure its in the right place!
reject_criteria = dict(eeg=250e-6, eog=200e-5)  # 200 μV and only extremeeog events
rej_step = .1  # in seconds


# %%
ica = mne.preprocessing.read_ica(input("file?"))

# %%fit ica
ica = mne.preprocessing.ICA(n_components=.90, method='infomax', random_state=97, max_iter=800, fit_params=dict(extended=True))
ica.fit(raw, reject_by_annotation=True, reject=reject_criteria)
ica.save(
    "SavedResults/S"+input("subject number?")+"/"+input("name?")+"-ica.fif")  # raw.save('visual-detrended-s2-rejected100-raw.fif')
# %%
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=2 / 2048)
event_dict_aud = {'short_word': 12, 'long_word': 22}
event_dict_vis = {'short_face': 10, 'long_face': 20,
                  'short_anim': 12, 'long_anim': 22,
                  'short_obj': 14, 'long_obj': 24,
                  'blink':2, 'saccade':3}#,
# %%
stimuli = ['short_anim','long_anim','short_face','long_face','short_obj','long_obj']
plot_ica_component(raw, ica, events, event_dict_vis, stimuli)
# %%
#plot components function here
matplotlib.use('Qt5Agg')



# %%
ica = mne.preprocessing.read_ica(input())
# checking components is in running_script.py
ica.exclude = [2, 10]

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

#                  'short_body': 16, 'long_body': 26}
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
evoked = epochs.average()
# %%
ds_epochs = epochs.copy().resample(512)

raw.plot()
