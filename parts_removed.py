### this script will contain parts I removed from the original pipeline


##
raw.filter(h_freq=None, l_freq=1)


# %% in case of existing raw file, like detrended:
raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file: "))
raw.load_data()
