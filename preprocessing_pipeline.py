# %
# ##import

from preprocess_utilities import *

# matplotlib.use('TkAgg')
# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files()

# concatenate to one raw file
raw = mne.concatenate_raws(raws)
raw = add_bipolar_derivation(raw, 'LHEOG', 'RHEOG')
raw = add_bipolar_derivation(raw, 'RVEOGS', 'RVEOGI')
# drop bad channels
# raw = add_bipolar_derivation(raw, "LHEOG", "RHEOG")##n?
raw.drop_channels(['Ana' + str(i) for i in range(1, 9)])
raw.drop_channels(["C26", "D3"])  # bridged/noisy channels we choose to remove ##n

raw.notch_filter([50, 150])  # notch filter before copying ##n

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)

# %%
copy_raw = raw.copy()
raw.load_data().filter(l_freq=1., h_freq=None)  ##n

# %%
# fit ica
# ica = mne.preprocessing.ICA(n_components=.99, random_state=97, max_iter=800)
# ica.fit(raw)
# %%##n
# ica.save('ica_fits/0202-ica.fif')##n

# %%##n
ica = mne.preprocessing.read_ica('ica_fits/0202-ica.fif')  ##n
# %%
# plot components topography
ica.plot_components(outlines='skirt', picks=range(20))
# %%
ica.plot_sources(raw, range(10, 20))

# %%
# plot properties of component by demand
ica.plot_properties(raw, picks=range(11))

# %% # plot correlation of all components with eog channel
ica.exclude = []  ####n all
# find which ICs match the EOG pattern
eog_map_dict = {'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog', 'RVEOGS': 'eog', 'RVEOGI': 'eog', 'M1': 'eog',
                'M2': 'eog', 'LVEOGI': 'eog'}
raw.set_channel_types(mapping=eog_map_dict)
# %%
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.plot_scores(eog_scores, title="Nose correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="LHEOG", threshold=2.5)
ica.plot_scores(eog_scores, title="Left horizontal correlations")
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="RVEOGS")
ica.plot_scores(eog_scores, title="Right vertical correlations")

# barplot of ICA component "EOG match" scores

# %%
# exclude components
raw = copy_raw  ##n
ica.exclude = [0, 3, 7, 11, 12, 13, 14, 18]  # components we delete
ica.apply(raw)

# reject bad intervals** - make sure its in the right place!
reject_criteria = dict(eeg=150e-6)  # 150 μV

# %% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers
events = mne.find_events(raw, stim_channel="Status", mask=255)
event_dict = {'short_word': 12, 'short_resp': 13, 'long_word': 22,
              'start_recording': 254, 'long_resp': 23}
raw.set_eeg_reference(ref_channels=['M1', 'M2'])
filt_raw = raw.copy()  # save unfiltered copy for TF analysis ##n
filt_raw.load_data().filter(l_freq=1, h_freq=30)  ## epoch for ERPs ##n
filt_epochs = mne.Epochs(filt_raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                         reject=reject_criteria, preload=True)
# epoch raw data without filtering fot TF analysis
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=1.6,
                    reject=reject_criteria, preload=True)
