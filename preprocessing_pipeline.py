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

## upload raw files AFTER robust detrending

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

## plot components topography
comp_to_plot = 27 ## number of components to plot


## plot properties of component by demand
ica.plot_properties(raw, picks=[0, 1])

## plot correlation of alll components with eog channel

## exclude components
ica.exclude = [0]  # components we delete
ica.apply(copy_raw)

## reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=150e-6)  # 150 μV

## epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                    reject=reject_criteria, preload=True)
conds_we_care_about = ['short_word', 'long_word']
epochs.equalize_event_counts(conds_we_care_about)
short_epochs = epochs['short_word'];
long_epochs = epochs['long_word']

## filter triggers for ERP - set parameters, filter bandpass+notch, set reference
filt_epochs = process_epochs(short_epochs)
filt_epochs_plot = filt_epochs.plot_image(picks=['A1'])

## visualize ERP by electrode

## time frequency analysis



