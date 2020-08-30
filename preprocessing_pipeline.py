# %
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib
from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType
# matplotlib.use('Qt5Agg')

# %%
raw=mne.io.read_raw_fif("SavedResults/S4/S4_vis_artifact_rej-raw.fif",preload=True)
# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files(preload=True)
# concatenate to one raw file
raw = mne.concatenate_raws(raws)
raw.drop_channels(['ET_RX', 'ET_RY', 'ET_R_PUPIL', 'ET_LX', 'ET_LY',
                   'ET_L_PUPIL', 'Photodiode', 'ResponseBox'])
copy_raw = raw.copy()  # make a copy before adding the new channel
raw = raw.resample(512,n_jobs=12)

raw.filter(h_freq=None, l_freq=1, n_jobs=12)
raw.plot(n_channels=30, duration=30)  # exclude electrodes from artifact rejection

# %%
raw = set_reg_eog(raw)
# %%
raw = annotate_bads_auto(raw, reject_criteria=200e-6, jump_criteria=100e-6) #by threshold and jump
# %% plot again to see annotations and mark missed noise/jumps
raw.plot(n_channels=30, duration=30)  # to see data and mark bad  segments

# %%
print("total time annotated as bad: ", round(sum(raw._annotations.duration), 2))
# %% drop bad channels, annotate bad intervals
plot_all_channels_var(raw, max_val=4e-7, threshold=500e-10)  # max value for visualization in case of large value
raw.plot(n_channels=30, duration=30)  # to see data and mark bad channels
# %% set bads
raw.info['bads']


# %%
if input("auditory? (Y/N)") == 'Y':
    raw = annotate_breaks(raw)  # only for auditory

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
raw.set_eeg_reference(['Nose'])
# %%
# reject bad intervals based on peak to peak in ICA
reject_criteria = dict(eeg=450e-6, eog=300e-5)  # 200 μV and only extreme eog events
rej_step = .1  # in seconds
# %% set events
et_processor = EyeLinkProcessor("SavedResults/S4/S4_visual.asc",ParserType.MONOCULAR_NO_VELOCITY,
                                SaccadeDetectorType.ENGBERT_AND_MERGENTHALER)
#%%
et_processor.sync_to_raw(raw)

#%%
saccade_times = et_processor.get_synced_microsaccades()
#%%
blink_times  =et_processor.get_synced_blinks()
#check sync
#plt.plot((raw.get_data(264).astype(np.int) & 255)[0]==254) # block start
plt.plot(np.in1d(np.arange(len(raw.get_data(1)[0])),saccade_times-88789+449)) #blink triggers
#plt.plot((raw.get_data(259))[0]>0.0001) # EOG channel
plt.plot((raw.get_data(258))[0]/max((raw.get_data(258))[0])) # EOG channel
#%%
raw._data[raw.ch_names.index("Status")][blink_times] = 99  # set blinks
raw._data[raw.ch_names.index("Status")][saccade_times] = 98  # set saccades
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=2 / 2048)
event_dict_aud = {'blink':2, 'saccade':3,
                  'short_word': 12, 'long_word': 22}
event_dict_vis = {'blink':2, 'saccade':3,
                  'short_scrambled': 110, 'long_scrambled': 112,
                  'short_face': 120, 'long_face': 122,
                  'short_obj': 130, 'long_obj': 132,
                  'short_body': 140, 'long_body': 142}
                  #'blink':2, 'saccade':3}#,#
                  # 'short_body': 16, 'long_body': 26}
stim_onset_events = list(event_dict_vis.values())
raw_for_ica = multiply_event(raw,stim_onset_events = list(event_dict_vis.values()), event_id=2)
# %%
ica = mne.preprocessing.read_ica(input("file?"))

# %%fit ica
ica = mne.preprocessing.ICA(n_components=.99, method='infomax',
                            random_state=97, max_iter=800, fit_params=dict(extended=True))
ica.fit(raw_for_ica, reject_by_annotation=True, reject=reject_criteria)
ica.save(
    "SavedResults/S"+input("subject number?")+"/"+input("name?")+"-ica.fif")
# example: raw.save('visual-detrended-s2-rejected100-raw.fif')

# %%
stimuli = ['short_scrambled', 'long_scrambled','short_face', 'long_face',
           'short_obj', 'long_obj','short_body', 'long_body']
comp_start = 0  # from which component to start showing
ica.exclude = plot_ica_component(raw, ica, events, event_dict_vis, stimuli, comp_start)

# %%
ica.apply(copy_raw)

# %% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers

raw.notch_filter([50, 100, 150])  # notch filter
# raw_filt = raw.copy().filter(l_freq=1, h_freq=40)  # performing filtering on copy of raw data, not on raw itself or epochs
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
# %%
evoked.plot_topomap()

# %%
def plt_plot(series):
