# %%
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib
matplotlib.use('Qt5Agg')

files_path = "SavedResults/S2/visual-hpf1-rejected100-ica.fif"
raw = mne.io.read_raw_fif("SavedResults/S2/detrended_visual_s2-raw.fif") #filtered_not raw epochs
#%%
ica = mne.preprocessing.read_ica("SavedResults/S2/visual-detrended-s2-ica.fif")
ica.plot_properties(raw, picks=range(5), show=False, psd_args={'fmax': 100})  # plot component properties
ica.plot_sources(raw, show=False)  # plot sources
#%%
#raw = mne.io.read_raw_fif("SavedResults/S3/detrended_aud_s3-raw.fif", preload=True) #raw unfiltered
ica.apply(raw)
raw.save("SavedResults/S2CHangeThis.fif")