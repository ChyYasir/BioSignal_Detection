import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def load_signals(dat_file, n_channels, n_samples):
    # Assuming 16-bit signed integer data
    data = np.fromfile(dat_file, dtype=np.int16)
    # Reshape data into n_channels rows
    signals = data.reshape((n_samples, n_channels)).T
    return signals

def read_header_file(header_file):
    with open(header_file, 'r') as file:
        header_content = file.readlines()


    record_name, n_channels, sampling_frequency, n_samples = header_content[0].split()
    print("Record Name = ", record_name)
    return int(sampling_frequency), int(n_samples)



def plot_signals(signals, sampling_frequency):
    time_axis = np.arange(signals.shape[1]) / sampling_frequency
    fig, axs = plt.subplots(nrows=signals.shape[0], ncols=1, figsize=(15, 20), sharex=True)

    for i in range(signals.shape[0]):
        axs[i].plot(time_axis, signals[i])
        axs[i].set_title(f'Signal {i+1}')
        axs[i].set_ylabel('Amplitude')

    plt.xlabel('Time (seconds)')
    plt.show()


def plot_single_signal(signal, sampling_frequency, signal_number=1):

    time_axis = np.arange(signal.size) / sampling_frequency
    plt.figure(figsize=(15, 4))
    plt.plot(time_axis, signal)
    plt.title(f'Signal {signal_number}')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    plt.show()


def plot_filtered_signal_with_original(signal, sampling_frequency, lowcut=0.8, highcut=4, signal_number=1):
    # Exclude initial 500 and final 500 samples for the original signal to match the filtered version
    signal = signal * 1000
    original_signal_excluded = signal[500:-1000]

    # Apply Butterworth bandpass filter
    filtered_signal = butter_bandpass_filter(signal[500:-1000], lowcut, highcut, sampling_frequency)

    # Prepare time axis, adjusted for the excluded samples
    time_axis = np.arange(500, signal.size-1000) / sampling_frequency

    plt.figure(figsize=(15, 4))

    # Plot original signal
    # plt.plot(time_axis, original_signal_excluded, color='blue', label='Original Signal')

    # Plot filtered signal
    plt.plot(time_axis, filtered_signal, color='red', label='Filtered Signal')

    plt.title(f'Signal {signal_number}: Original and Filtered (Butterworth Bandpass)')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.show()
header_file = 'F:/signal/files/ice002_p_1of3.hea'
dat_file = 'F:/signal/files/ice002_p_1of3.dat'

sampling_frequency, n_samples = read_header_file(header_file)
signals = load_signals(dat_file, 16, n_samples)
# plot_signals(signals, sampling_frequency)

# plot_single_signal(signals[0], sampling_frequency, signal_number=1)
plot_filtered_signal_with_original(signals[0], sampling_frequency, signal_number=1)