# %%
# ##import
import os
import os.path as op
import numpy as np
import matplotlib
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import seaborn as sns
import math
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from preprocess_utilities import *


###starts with an epoch object after artifact removal
###all further analysis will happen here


#%% parameters
freq_range = [1,200]
base_correction = (-0.25,0); correction_mode= 'logratio'
alpha = [8,12]; beta = [13,30]; narrowgamma = [31,60]; high_gamma = [80,200]

#%% # filter triggers for ERP - choose triggers, set parameters, filter bandpass+notch, set reference
filt_epochs = process_epochs('short_word',short_epochs) #filters included by default! choose the triggers you want to process
evoked = filt_epochs.average()
evoked.plot_topomap(times=[0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.4, 0.5,0.7,0.9])

#%% # visualize ERP by electrode
filt_epochs_plot = filt_epochs.plot_image(picks=['A1'])
######or this?
filt_epochs_plot = evoked.plot_image(picks=['A1'])

#%% # time frequency analysis
# define frequencies of interest (log-spaced)
freqs = np.logspace(*np.log10(freq_range), num=100)
power = tfr_morlet(epochs, freqs=freqs, n_cycles=freqs/2, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)

power.plot_topo(baseline=base_correction, mode=correction_mode, title='Average power')
power.plot([3], baseline=base_correction, mode=correction_mode, title=power.ch_names[3])

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=alpha[0], fmax=alpha[1],
                   baseline=base_correction, mode=correction_mode, axes=axis[0],
                   title='Alpha', show=False)
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=beta[0], fmax=beta[1],
                   baseline=base_correction, mode=correction_mode, axes=axis[1],
                   title='Beta', show=False)
mne.viz.tight_layout()
plt.show()

