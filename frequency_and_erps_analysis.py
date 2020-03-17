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
from scipy import signal
from preprocess_utilities import *
import gc
###starts with an epoch object after artifact removal
###all further analysis will happen here


# %% parameters
chosen_s_trigs = ['short_word']
chosen_l_trigs = ['long_word']
chosen_s_trigs = ['short_anim', 'short_obj', 'short_face']
chosen_l_trigs = ['long_anim', 'long_obj', 'long_face']
# freq_range = [5, 200]
base_correction = (-0.25, -.10)  # when epoch starts at -0.400
correction_mode = 'logratio'
# alpha = [8, 12]
# beta = [13, 30]
# narrowgamma = [31, 60]
# high_gamma = [80, 200]
freqsH = np.logspace(5, 7.6, 30, base=2)
freqsL = np.logspace(1, 5.5, 8, base=2)
n_perm = 500
save_path = 'SavedResults/S3'
# %% # triggers for ERP - choose triggers, create topomap and ERP graph per chosen electrode ##n all
curr_epochs = filt_epochs  # choose the triggers you want to process
evoked = curr_epochs.average()
evoked.plot_topomap(outlines='skirt', times=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9])

# %% # visualize ERP by electrode
filt_epochs_plot = curr_epochs.plot_image(picks=['A1'])
#%% calculate power and save
power_long_H = tfr_morlet(epochs['long_anim', 'long_obj', 'long_face', 'long_body'], freqs=freqsH, average=False,
                          n_cycles=freqsH / 10, use_fft=True,
                          return_itc=False, decim=3, n_jobs=5)
power_long_H.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_long_trials_high_freq-pow.fif'))
del power_long_H
gc.collect()
#%%
power_short_H = tfr_morlet(epochs['short_anim', 'short_obj', 'short_face', 'short_body'], freqs=freqsH, average=False,
                           n_cycles=freqsH / 10, use_fft=True,
                           return_itc=False, decim=3, n_jobs=5)
power_short_H.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_short_trials_high_freq-pow.fif'))
del power_short_H
gc.collect()
#%% low freqs tfr
power_long_L = tfr_morlet(epochs['long_anim', 'long_obj', 'long_face', 'long_body'], freqs=freqsL, average=False,
                          n_cycles=freqsL / 10, use_fft=True,
                          return_itc=False, decim=3, n_jobs=5)
power_long_L.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_long_trials_low_freq-pow.fif'))
del power_long_L; gc.collect()
power_short_L = tfr_morlet(epochs['short_anim', 'short_obj', 'short_face', 'short_body'], freqs=freqsL, average=False,
                           n_cycles=freqsL / 10, use_fft=True,
                           return_itc=False, decim=3, n_jobs=5)
power_short_L.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_short_trials_low_freq-pow.fif'))
del power_short_L
gc.collect()
# %% # time frequency analysis - high freqs tfrs

# %%
power = mne.time_frequency.read_tfrs("SavedResults/S3/S3_vis_detrended_ord10_10s_windows_long_trials_high_freq-pow.fif")
power=power[0].average() # get onlyaverage induced response
power.plot_topo(baseline=base_correction, mode=correction_mode, title='Average power',
                       tmin=-0.3, tmax=1.8,vmin=-.25, vmax=.25, layout_scale=.5 )

# %% show ERP after hilbert
## filter raw data for 60-80, 80-100, 100-120,120-140 using iir butterwoth filter of order 3
raw_hilb = []
nbands = 4  # starting from 60, jumping by 20
for i in range(4):
    curr_l_freq = (i + 1) * 20 + 40
    curr_h_freq = curr_l_freq + 20
    raw_bp = raw.copy().filter(l_freq=curr_l_freq, h_freq=curr_h_freq, method='iir',
                               iir_params=dict(order=3, ftype='butter'))
    raw_hilb.append(raw_bp.apply_hilbert(envelope=True), base=10)  # compute envelope of analytic signal
    raw_hilb[i]._data[:] = 10 * np.log(raw_bp._data ** 2, base=10)  # change to dB
    raw_hilb[i] = raw_hilb[i]._data[:] - raw_hilb[i]._data[:].mean(1)  # demean to apply correction
    print("finished band of " + str(curr_l_freq) + " - " + str(curr_h_freq))

## compute mean of all filter bands
raw_hilb[0]._data = (raw_hilb[0]._data + raw_hilb[1]._data + raw_hilb[2]._data + raw_hilb[3]._data) / nbands
epochs_hilb = mne.Epochs(raw_hilb[0], events, event_id=event_dict_vis,
                         tmin=-0.4, tmax=1.9, baseline=None,
                         reject=reject_criteria,
                         reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                         preload=True, reject_by_annotation=True)
del raw_hilb
# apply hilbert
epochs_hilb.apply_baseline((-.3, -.1), verbose=True)

# %% show ERP after hilbert
check_electrode = "B8"
epochs_hilb['short_anim', 'short_obj', 'short_face', 'short_body'].plot_image(picks=[check_electrode])

# # %%
# power_list = []
# # widths = [0.2, 0.4]
# widths = [0.2]
# for width in widths:
#     power_list.append(tfr_morlet(ds_epochs[["long_word"]], freqs=freqs, use_fft=True,
#                                  n_cycles=np.concatenate([3 * np.ones(9), 12 * np.ones(29)]), verbose=True, average=False, return_itc=False))
# # n_cycles = freqs * width
#
# # %%
# power_list_baseline_corrected = []
# for i in range(len(power_list)):
#     power_list_baseline_corrected.append(power_list[i].copy().apply_baseline(mode='logratio', baseline=(-.200, 0)))
# # %%
# power_avg = [power.average() for power in power_list_baseline_corrected]
# # %%
# power_avg[0].plot_topo()
#
# # %%
# for i in range(len(power_list)):
#     power_list[i].crop(0., 1.6)
#
# evoked = ds_epochs.average()
# evoked.crop(0., 1.6)
# times = evoked.times
#
# del evoked
#
# # %%
# for j in range(18, 145):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     for power, ax, width in zip(power_list, axs, widths):
#         epochs_power = power.data[:, j, :, :]  # take the 1 channel
#         threshold = 2.5
#         T_obs, clusters, cluster_p_values, H0 = \
#             mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_perm, tail=0)
#         T_obs_plot = np.nan * np.ones_like(T_obs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= 0.05:
#                 T_obs_plot[c] = T_obs[c]
#         vmax = np.max(np.abs(T_obs))
#         vmin = -vmax
#         ax.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                   aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#         ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
#                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                   aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#         ax.set_xlabel('Time (ms)')
#         ax.set_ylabel('Frequency (Hz)')
#         # ax.title('Induced power (%s)' % j)
#         # power.plot([j], baseline=(0., 0.2), mode='logratio', axes=ax, colorbar=True if width == widths[-1] else False,
#         #            show=False, vmin=-.6, vmax=.6)
#         ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
#     plt.show()
#     if input():
#         break
#
# # %%
# for j in range(197, 200):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     for power, ax, width in zip(power_list, axs, widths):
#         avg = power.average()
#         vmax = np.max(np.abs(avg.data))
#         vmin = -vmax
#         avg.plot([j], axes=ax, colorbar=True if width == widths[-1] else False,
#                  show=False, vmin=vmin, vmax=vmax)
#         ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
#     plt.show()
#     if input():
#         break
#
# # %% # running with low frequencies to ensure we see ERP
# power_low = tfr_morlet(epochs, freqs=[3, 6, 8, 10, 12, 14, 16], average=False,
#                         n_cycles=np.concatenate([3 * np.ones(7)]), use_fft=True,
#                         return_itc=False, decim=3, n_jobs=1)
# power_low = power_low.average()
