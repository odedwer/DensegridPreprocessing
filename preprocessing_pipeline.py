# %
# ##import
import numpy as np
from preprocess_utilities import *

# matplotlib.use('TkAgg')
# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files(preload=True)
#detrended_raws = load_raws_from_mat('detrended_ord10_10s_window.mat', raws)
# concatenate to one raw file
raw = mne.concatenate_raws(raws)
copy_raw = raw.copy() #make a copy before adding the new channel

# %% in case of existing raw file, like detrended:
raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file: "))
raw.load_data()
# raw._data = raw._data/10**6
# good_status_ch_raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file that is not detrended: "))
# good_status_ch_raw.load_data()
# raw._data[262] = good_status_ch_raw._data[272]
# del good_status_ch_raw



# %% filter, drop bad channels, annotate breaks
#raw.load_data().filter(l_freq=.1, h_freq=None)  ##n
raw = annotate_breaks(raw)
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
raw.drop_channels(["C26", "D3"])  # bridged/noisy channels we choose to remove ##n

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# %%
eog_map_dict = {'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog', 'RVEOGS': 'eog', 'RVEOGI': 'eog', 'M1': 'eog',
                'M2': 'eog', 'LVEOGI': 'eog'}
raw.set_channel_types(mapping=eog_map_dict)
# %%
# fit ica
# reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=250e-6)  # 250 μV
rej_step = .15  # in seconds
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, max_iter=800)
# ica.fit(raw, reject_by_annotation=True, reject=reject_criteria)
ica.fit(raw, reject_by_annotation=True, tstep=rej_step, reject=reject_criteria)

# %%##n
ica.save('det_ord10_s10_w_rejected150_breaks-ica.fif')
raw.save('hpf01_rejected150_breaks-raw.fif')
# %%
# checking components is in running_script.py
ica.exclude = [0,3,7,12,16,17,18]  ####n all

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
#raw = copy_raw  ##n
ica.exclude = [0, 6, 7, 14, 16, 20, 11, 9]  # components we delete
ica.apply(raw)



# %% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers
events = mne.find_events(raw, stim_channel="Status", mask=255)
event_dict = {'short_word': 12,  'long_word': 22}
raw.notch_filter([50, 150])  # notch filter
filt_raw = raw.copy()  # save unfiltered copy for TF analysis ##n
filt_raw.load_data().filter(l_freq=1, h_freq=30)  ## epoch for ERPs ##n
filt_epochs = mne.Epochs(filt_raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                          reject=reject_criteria, preload=True, reject_by_annotation=True)
# epoch raw data without filtering for TF analysis
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                    reject=reject_criteria, preload=True, reject_by_annotation=True)
