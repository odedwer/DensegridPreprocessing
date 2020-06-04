import numpy as np
import matplotlib.pyplot as plt

def remove_line_noise(signal, frequency=50, fs=512, win_size=1):
    """
    Removes line noise from a signal (channels * samples!) by filtering out windowed sine waves
    of the line frequency and its harmonics. The wider the window the more the filter ignores
    transient fluctuations and looks for consistent oscillations.
    Written By Gal Vishne June, 2020, Based on a Matlab function by Alon Keren April, 2008
    :param signal: the signal you want to de-noise - important: it must be channels * samples!
    :param frequency: frequency of line noise in Hz (default: 50Hz)
    :param fs: sampling frequency in Hz (default: 512Hz)
    :param win_size: filter window in seconds (default 1sec)
    :return: denoised\filtered signal (channels * samples)
    """
    win_size_samples = np.floor(win_size * fs)
    if win_size_samples < 2:
        raise Exception("Window size must be at least two samples long")

    # time_points of filter window (centered around 0)
    filter_time_points = (np.arange(-win_size_samples / 2, win_size_samples / 2) + 1) / fs
    num_time_points = filter_time_points.size

    # create noise filter
    num_harmonics = np.floor(fs / frequency / 2)
    freqs_harmonics = np.arange(1, num_harmonics + 1) * frequency
    harmonics_mat = np.cos(2 * np.pi * np.outer(freqs_harmonics, filter_time_points))
    harmonics_mat = harmonics_mat / (num_time_points / 2)
    cosine_win = np.cos(2 * np.pi * filter_time_points / win_size) + 1
    noise_filter = harmonics_mat.sum(axis=0) * cosine_win

    # create denoising impulse response
    impulse = np.zeros(num_time_points)
    impulse[int(np.floor(num_time_points / 2))] = 1
    denoise_filter = impulse - noise_filter

    # do the actual removal of line noise
    denoised = np.zeros(signal.shape)
    for ch in range(signal.shape[0]):
        denoised[ch, ] = np.convolve(signal[ch, ], denoise_filter, 'same')

    return denoised


fake_data = np.random.normal(size=(1, 15000)) + np.sin(2*np.pi*50*np.arange(15000)/512)
denoised = remove_line_noise(fake_data, win_size=3)

plt.subplot(211)
plt.psd(fake_data[0,], Fs=512)
plt.subplot(212)
plt.psd(denoised[0,], Fs=512)

plt.show()