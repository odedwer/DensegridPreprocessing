# %%
# ##import
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
from tkinter import Tk
from preprocess_utilities import *

## upload raw files AFTER robust detrending
raw = read_bdf_files()
load_from_mat()

## concatenate to one raw file

copy_raw = raw.copy()

## drop bad channels
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
# raw.drop_channels('A24') ##bridged/noisy channels we choose to remove

## set the montage of the electrodes - position on head
raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)

## fit ica
ica = mne.preprocessing.ICA(n_components=.95, random_state=97, max_iter=800)
ica.fit(raw)

#%%
# plot components topography
ica.plot_components(outlines='skirt',picks=range(27))
# ica.plot_sources(raw, range(27))

#%%
# plot properties of component by demand
ica.plot_properties(raw, picks=[0, 1])

#%% # plot correlation of all components with eog channel
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)



#%%
# exclude components
ica.exclude = [0]  # components we delete
ica.apply(raw)

## reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=150e-6)  # 150 μV

#%% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers
events = mne.find_events(raw, stim_channel="Status", mask=255)
event_dict = {'short_word': 12, 'short_resp': 13, 'long_word': 22,
              'start_recording': 254, 'long_resp': 23}
raw.set_eeg_reference(ref_channels=['M1', 'M2'])
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                    reject=reject_criteria, preload=True)


