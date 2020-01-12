# %%
import os
import numpy as np
import matplotlib
import mne
import seaborn as sns
import math
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from tkinter.filedialog import askopenfilename

# %%
# load data

# get file path using GUI
filename = askopenfilename()
# open file as RAW object, preload data into memory
raw = mne.io.read_raw_bdf(filename, preload=True)
# drop analog channels - in the meanwhile
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
# drop bridged (noisy) channels
#raw.drop_channels('A24')
# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
# raw.plot_psd(fmax=400)
# raw.plot(duration=5, n_channels=8)
# %%
##do here robustic trending
copy_raw = raw.copy()
raw.load_data().filter(l_freq=1., h_freq=None)
# %%
##ica
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, max_iter=800)
ica.fit(raw)
# %%
ica.plot_sources(raw)
ica.plot_components()

#%%
ica.plot_properties(raw, picks=[0, 1])
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
#ica.plot_properties(raw, picks=ica.exclude)

# # %%
# ##check for triggers
# events = mne.find_events(raw, stim_channel="Status", mask=255)
# print(events[:5])  # show the first 5
# event_dict = {'short_word': 12, 'short_resp': 13, 'long_word': 22, 'start_recording': 254, 'long_resp': 23}
# #fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
# #fig.subplots_adjust(right=0.7)  # make room for the legend
#
# # %% reject bad intervals
# reject_criteria = dict(eeg=150e-6)  # 150 μV
#
# # %% epoching
# epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
#                     reject=reject_criteria, preload=True)
# conds_we_care_about = ['short_word', 'long_word']
# epochs.equalize_event_counts(conds_we_care_about)  # this operates in-place
# short_epochs = epochs['short_word']
# long_epochs = epochs['long_word']
# # del raw, epochs  # free up memory
# short_epochs.plot_image(picks=['A1'])
# long_epochs.plot_image(picks=['A1'])
# # %%
# frequencies = np.arange(7, 30, 3)
# power = mne.time_frequency.tfr_morlet(aud_epochs, n_cycles=2, return_itc=False,
#                                       freqs=frequencies, decim=3)
# power.plot(['A1'])
