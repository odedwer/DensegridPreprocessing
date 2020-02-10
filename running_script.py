from preprocess_utilities import *
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
components = range(20) # number of components to show

if __name__ == "__main__":
    mpl.use("TkAgg")
    raw = mne.io.read_raw_fif(sys.argv[1])
    raw.load_data()
    raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
    print("plotting full psd...")
    raw.plot_psd(fmin=0,fmax=300,picks=range(20),n_fft=10*2048)
    plt.show()
    # print("plotting short psd...")
    # raw.plot_psd(fmin=0,fmax=30,picks=range(20),n_fft=10*2048)
    # plt.show()
    ica = mne.preprocessing.read_ica(sys.argv[2])
    print("plotting components...")
    ica.plot_overlay(raw)
    ica.plot_components(outlines='skirt', picks=components, show=False)
    plt.show()
    print("plotting properties...")
    comp_jumps = np.linspace(0, ica.n_components_, int(ica.n_components_ / 10) + 1) ##the begining of each components group to be shown
    for i in range(len(comp_jumps)): # go over the components and show 10 each time
        if input("stop plotting? (Y/N)") == "Y":
            break
        comps = range(int(comp_jumps[i]), int(comp_jumps[i + 1]))
        print("plotting from component "+str(comps))
        ica.plot_properties(raw, picks=comps, show=False)
        ica.plot_sources(raw, picks=comps, show=False)
        #raw.plot(duration=10, n_channels=15, order=range(256))
        plt.show()