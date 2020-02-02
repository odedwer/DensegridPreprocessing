# %%
# ##import
import math
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import os.path as op
import seaborn as sns
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from preprocess_utilities import *

###starts with an epoch object after artifact removal
###all further analysis will happen here


# %% parameters
freq_range = [1, 200]
base_correction = (-0.25, 0)
correction_mode = 'mean'
alpha = [8, 12]
beta = [13, 30]
narrowgamma = [31, 60]
high_gamma = [80, 200]

# %% # triggers for ERP - choose triggers, create topomap and ERP graph per chosen electrode ##n all
curr_epochs = filt_epochs['short_word']  # choose the triggers you want to process
evoked = curr_epochs.average()
evoked.plot_topomap(outlines='skirt', times=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9])

# %% # visualize ERP by electrode
filt_epochs_plot = curr_epochs.plot_image(picks=['A1'])

# %% # time frequency analysis
# define frequencies of interest (log-spaced)
freqs = (np.arange(2., 200., 3.))
curr_epochs = epochs['short_word'][1:50]  # choose the triggers you want to process

power = tfr_morlet(curr_epochs, freqs=freqs, n_cycles=freqs / 2, use_fft=True,
                   return_itc=False, decim=3, n_jobs=1)
# %%
power.save('tfr_onlyfirst50.fif', True)  ##n
# %%
power.plot_topo(baseline=base_correction, mode=correction_mode, title='Average power')
power.plot([3], baseline=base_correction, mode=correction_mode, title=power.ch_names[3])
# %%
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=alpha[0], fmax=alpha[1],
                   baseline=base_correction, outlines='skirt', mode=correction_mode, axes=axis[0],
                   title='Alpha', show=False)
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=beta[0], fmax=beta[1],
                   baseline=base_correction, outlines='skirt', mode=correction_mode, axes=axis[1],
                   title='Beta', show=False)
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=80, fmax=150,
                   baseline=base_correction, outlines='skirt', mode=correction_mode, axes=axis[1],
                   title='Beta', show=False)
mne.viz.tight_layout()
plt.show()