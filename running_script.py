from preprocess_utilities import *
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
components = range(20)  # number of components to show

if __name__ == "__main__":
    mpl.use("TkAgg")
    root = Tk()
    root.withdraw()
    raw = mne.io.read_raw_fif(askopenfilename(title="Please choose raw file"))
    raw=mne.read_epochs("SavedResults/S2/det_epochs-epo.fif")
    raw.load_data()
    raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
    print("plotting full psd...")
    #raw.plot_psd(fmin=0, fmax=300, picks=range(20), n_fft=10 * 2048)
    #plt.show()
    ica = mne.preprocessing.read_ica(askopenfilename(title="Please choose ICA file"))
    print("plotting components...")
    ica.plot_components(picks=components, show=False)
    plt.show()
    root.destroy()

    print("plotting properties...")
    # the beginning of each components group to be shown
    comp_jumps = np.linspace(0, ica.n_components_, int(ica.n_components_ / 8) + 1)
    for i in range(len(comp_jumps)):  # go over the components and show 8 each time
        if input("keep plotting? (Y/N)") == "N":
            break
        comps = range(int(comp_jumps[i]), int(comp_jumps[i + 1]))
        print("plotting from component " + str(comps))
        plot_correlations(ica, raw, components=comps,
                          picks=['A19', 'Nose', 'RHEOG', 'LHEOG', 'RVEOGS', 'RVEOGI', 'M1', 'M2', 'LVEOGI'])
        ica.plot_properties(raw, picks=comps, show=False)  # plot component properties
        ica.plot_sources(raw, picks=comps, show=False)  # plot sources
        plt.show()
