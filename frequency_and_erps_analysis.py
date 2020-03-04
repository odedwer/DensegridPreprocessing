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
from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper, psd_welch, tfr_stockwell
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from preprocess_utilities import *

###starts with an epoch object after artifact removal
###all further analysis will happen here


# %% parameters
freq_range = [1, 200]
base_correction = (-0.25, 0)
correction_mode = 'logratio'
alpha = [8, 12]
beta = [13, 30]
narrowgamma = [31, 60]
high_gamma = [80, 200]

# %% # triggers for ERP - choose triggers, create topomap and ERP graph per chosen electrode ##n all
curr_epochs = filt_epochs['long_obj']  # choose the triggers you want to process
evoked = curr_epochs.average()
evoked.plot_topomap(outlines='skirt', times=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9])

# %% # visualize ERP by electrode
filt_epochs_plot = curr_epochs.plot_image(picks=['A19'])

# %% # time frequency analysis
# define frequencies of interest (log-spaced)
freqs = (np.arange(25., 200., 3.))
curr_epochs = epochs  # choose the triggers you want to process

power_long = tfr_morlet(curr_epochs, freqs=freqs, n_cycles=freqs / 2, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)
curr_epochs = epochs['short_word']  # choose the triggers you want to process

power_short = tfr_morlet(curr_epochs, freqs=freqs, n_cycles=freqs / 2, use_fft=True,
                         return_itc=False, decim=3, n_jobs=1)
# %%
freqs = np.arange(5., 200., 3.)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
power_list = []
widths = [0.3, 0.5, 0.7]
for width in widths:
    power_list.append(tfr_multitaper(chosen_electrode_epochs[["long_anim", "long_obj", "long_face"]], freqs=freqs,
                                     n_cycles=freqs * width, verbose=True, average=False, return_itc=False))
# %%

# %%
evoked = chosen_electrode_epochs.average()
evoked.crop(0., 1.6)
evoked_data = evoked.data
times = evoked.times
# %%
for i in range(len(power_list)):
    power_list[i].apply_baseline(mode='logratio', baseline=(-.200, 0))
    power_list[i].crop(0., 1.6)
# %%
n_perm = 500

# %%
for j in range(18, 145):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for power, ax, width in zip(power_list, axs, widths):
        epochs_power = power.data[:, j, :, :]  # take the 1 channel
        threshold = 2.5
        T_obs, clusters, cluster_p_values, H0 = \
            mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_perm, tail=0)
        T_obs_plot = np.nan * np.ones_like(T_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= 0.05:
                T_obs_plot[c] = T_obs[c]
        vmax = np.max(np.abs(T_obs))
        vmin = -vmax
        print("vmax", vmax)
        print("vmin", -vmax)
        ax.imshow(T_obs, cmap=plt.cm.RdBu_r,
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        # ax.title('Induced power (%s)' % j)
        # power.plot([j], baseline=(0., 0.2), mode='logratio', axes=ax, colorbar=True if width == widths[-1] else False,
        #            show=False, vmin=-.6, vmax=.6)
        ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
    plt.show()
    if input():
        break

# %%
for j in range(197,200):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for power, ax, width in zip(power_list, axs, widths):
        avg = power.average()
        vmax = np.max(np.abs(avg.data))
        vmin = -vmax
        avg.plot([j], axes=ax, colorbar=True if width == widths[-1] else False,
                 show=False, vmin=vmin, vmax=vmax)
        ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
    plt.show()
    if input():
        break
# plt.tight_layout()
# %%
# power.save('tfr_onlyfirst50.fif', True)  ##n
# %%
power.plot_topo(baseline=base_correction, mode=correction_mode, title='Average power')
power.plot([1], baseline=base_correction, mode=correction_mode, title=power.ch_names[1])
# %%
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=narrowgamma[0], fmax=narrowgamma[1],
                   baseline=base_correction, outlines='skirt', mode=correction_mode, axes=axis[0],
                   title='gamma', show=False)
power.plot_topomap(ch_type='eeg', tmin=0, tmax=1.8, fmin=high_gamma[0], fmax=high_gamma[1],
                   baseline=base_correction, outlines='skirt', mode=correction_mode, axes=axis[1],
                   title='BB_gamma', show=False)
mne.viz.tight_layout()
plt.show()
