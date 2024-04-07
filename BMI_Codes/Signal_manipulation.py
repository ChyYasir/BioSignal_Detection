import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, welch


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
    print("Sampling Frequency = ", sampling_frequency)

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
    signal_mV = signal / adc_resolution
    time_axis = np.arange(signal.size) / sampling_frequency
    plt.figure(figsize=(15, 4))
    plt.plot(time_axis, signal_mV )
    plt.title(f'Signal {signal_number}')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    plt.show()


def plot_filtered_signal_with_original(signal, sampling_frequency, lowcut=0.8, highcut=4, signal_number=1):
    # Exclude initial 500 and final 500 samples for the original signal to match the filtered version
    # signal = signal * 1000
    signal_mV = signal / adc_resolution
    original_signal_excluded = signal_mV[500:-1000]

    # Apply Butterworth bandpass filter
    filtered_signal = butter_bandpass_filter(signal_mV[500:-1000], lowcut, highcut, sampling_frequency)

    # Prepare time axis, adjusted for the excluded samples
    time_axis = np.arange(500, signal.size-1000) / sampling_frequency

    plt.figure(figsize=(15, 4))

    # Plot original signal
    plt.plot(time_axis, original_signal_excluded, color='blue', label='Original Signal')

    # Plot filtered signal
    # plt.plot(time_axis, filtered_signal, color='red', label='Filtered Signal')

    plt.title(f'Signal {signal_number}: Original and Filtered (Butterworth Bandpass)')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.show()


def plot_filtered_signal_for_duration(signal, sampling_frequency, lowcut, highcut, duration=10, signal_number=1):
    # Convert the signal to millivolts
    signal_mV = signal / adc_resolution

    # Apply Butterworth bandpass filter to the signal
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, sampling_frequency)

    # Calculate the number of samples corresponding to the desired duration (e.g., 10 seconds)
    n_samples_duration = int(duration * sampling_frequency)
    original_signal_excluded = signal[:n_samples_duration]
    # Slice the time axis and filtered signal to the desired duration
    time_axis = np.arange(0, n_samples_duration) / sampling_frequency
    filtered_signal_duration = filtered_signal[:n_samples_duration]

    # Plotting
    plt.figure(figsize=(15, 4))
    plt.plot(time_axis, original_signal_excluded, color='blue', label='Original Signal')
    # plt.plot(time_axis, filtered_signal_duration, color='red', label='Filtered Signal')
    plt.title(f'Signal {signal_number}: Filtered (Butterworth Bandpass) for First {duration} Seconds')
    plt.ylabel('Amplitude (mV)')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

def power_density_fft(signal):
    # Compute the FFT
    n = len(signal)
    fs = sampling_frequency
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    fft_result = np.fft.rfft(signal)

    # Compute the power spectral density
    psd = np.abs(fft_result) ** 2 / (fs * n)
    # Plot the power spectral density
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, psd)  # Plot in logarithmic scale
    plt.title('Power Spectral Density of Signal(FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()

def power_density_welch(signal):
    fs = sampling_frequency
    # Compute the power spectral density using Welch's method
    frequencies, psd = welch(signal, fs=fs, nperseg=1024)

    # Plot the power spectral density
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, psd)  # Plot in logarithmic scale
    plt.title('Power Spectral Density of Signal(Welch)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()
header_file = 'F:/signal/files/ice002_p_1of3.hea'
dat_file = 'F:/signal/files/ice002_p_1of3.dat'
adc_resolution = 131.068
sampling_frequency, n_samples = read_header_file(header_file)
signals = load_signals(dat_file, 16, n_samples)
# plot_signals(signals, sampling_frequency)

# plot_single_signal(signals[0], sampling_frequency, signal_number=1)
# plot_filtered_signal_with_original(signals[0], sampling_frequency, signal_number=1)
# plot_filtered_signal_for_duration(signals[0], sampling_frequency, lowcut=0.8, highcut=4, duration=10, signal_number=1)


power_density_fft(signals[0])
power_density_welch(signals[0])