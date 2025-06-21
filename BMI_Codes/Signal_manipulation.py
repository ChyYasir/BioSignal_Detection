import numpy as np
import matplotlib.pyplot as plt
import pywt
import wfdb
from scipy.signal import butter, filtfilt, welch, medfilt, decimate
from scipy.interpolate import interp1d
from sklearn import preprocessing
import copy
from math import sqrt
from TFdomain_feature_extraction import TFdomainFeaturesExtract
from concave_hull_fourier import AlphaConcaveHull
class SignalManipulation:
    def __init__(self,  signal_name):
        # self.header_file = header_file
        # self.dat_file = dat_file
        self.record = wfdb.rdrecord(signal_name)
        # self.adc_resolution = adc_resolution
        self.sampling_frequency, self.n_samples = self.read_header_file(signal_name+".hea")
        self.signals = self.load_signals()
        self.zero_crossing_rates = []
        self.time = []
        self.timestamps = []
        self.interpolated_normalized_power = []
        self.modulated_signal = []
        self.rms_values = []
        self.threshold = 0
        self.new_contraction_array = []
        self.contraction_segments = []

    def read_header_file(self, header_file):
        with open(header_file, 'r') as file:
            header_content = file.readlines()

        record_name, n_channels, sampling_frequency, n_samples = header_content[0].split()
        print("Record Name = ", record_name)
        print("Sampling Frequency = ", sampling_frequency)

        return int(sampling_frequency), int(n_samples)

    def load_signals(self):
        # data = np.fromfile(dat_file, dtype=np.int16)
        # signals = data.reshape((n_samples, n_channels)).T
        signals = self.record.p_signal
        # # print(signals)
        # for i in range(16):
        #     print(signals[:, i])
        return signals

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def plot_signals(self):
        time_axis = np.arange(self.signals.shape[1]) / self.sampling_frequency
        fig, axs = plt.subplots(nrows=self.signals.shape[0], ncols=1, figsize=(15, 20), sharex=True)

        for i in range(self.signals.shape[0]):
            axs[i].plot(time_axis, self.signals[i])
            axs[i].set_title(f'Signal {i+1}')
            axs[i].set_ylabel('Amplitude')

        plt.xlabel('Time (seconds)')
        plt.show()

    def decimate_signal(self, signal, original_fs, target_fs, decimation_factor):
        # Step 1: Reduce high-frequency signal components with a digital lowpass filter
        # Example: Butterworth lowpass filter
        trim_duration = 50
        trim_samples = int(trim_duration * original_fs)
        signal = signal[trim_samples:-trim_samples]


        lowcut = 0.8  # Define your lowcut frequency
        highcut = 3 # Define your highcut frequency
        nyquist_freq = 0.5 * self.sampling_frequency
        print(self.sampling_frequency, nyquist_freq)
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        # print(low, high)
        # Compute the filter coefficients
        order = 4
        b, a = butter(order, [low, high], btype='band')

        butter_filter = filtfilt(b, a, signal)

        if(original_fs == 20):
            return butter_filter, 20
        # filtered_signal = self.butter_bandpass_filter(signal, lowcut, highcut, original_fs)

        decimated_signal = decimate(butter_filter, decimation_factor, ftype='fir')

        # Return the decimated signal and the new sampling frequency
        new_fs = original_fs / decimation_factor

        return decimated_signal, new_fs

    def plot_single_signal(self, signal, sampling_frequency):
        # signal = self.signals[signal_number]
        # signal_mV = signal/ self.adc_resolution
        # target_fs = 20
        # decimation_factor = self.sampling_frequency // target_fs
        # processed_signal, new_fs = self.decimate_signal(signal_mV, self.sampling_frequency, target_fs, decimation_factor)
        time_axis = np.arange(len(signal)) / sampling_frequency
        plt.figure(figsize=(15, 4))
        plt.plot(time_axis, signal)
        # plt.title(f'Signal {signal_number+1}')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.show()


    # def signal_preprocess(self, signal):
    #
    #     decimation_factor = 50
    #     decimated_signal, new_fs = self.decimate_signal(signal, self.sampling_frequency, 4, decimation_factor)
    #
    #
    #     trim_duration = 20
    #     trim_samples = int(trim_duration * new_fs)
    #     decimated_signal_trimmed = decimated_signal[trim_samples:-trim_samples]
    #
    #     # Step 3: Wavelet packet passband filtering between 0.1 and 1 Hz
    #     # Perform wavelet packet decomposition
    #     wp = pywt.WaveletPacket(data=decimated_signal_trimmed, wavelet='db4', mode='symmetric', maxlevel=5)
    #
    #     # Filter out the desired frequency band
    #     passband_min_freq = 0.1  # Hz
    #     passband_max_freq = 1.0  # Hz
    #     filtered_wp = wp['a'].data  # Initialize with the approximation coefficients (low pass)
    #     for node in wp.get_level(5, 'freq'):
    #         # Extract frequency information from the node's path
    #         freq_str = node.path.split('.')[1]  # Frequency encoded in the node's name
    #         freq = float(freq_str)
    #         if passband_min_freq <= freq <= passband_max_freq:
    #             filtered_wp = np.append(filtered_wp, node.data)
    #
    #     # Update the filtered signal and sampling frequency
    #     filtered_signal = filtered_wp
    #     return filtered_signal
    def plot_filtered_signal_with_original(self, signal_number=0, lowcut=0.8, highcut=4):
        signal = self.signals[:, signal_number]
        signal_mV = signal
        original_signal_excluded = signal_mV[500:-1000]
        print(self.signals.shape)
        filtered_signal = self.butter_bandpass_filter(signal_mV[500:-1000], lowcut, highcut, self.sampling_frequency)
        time_axis = np.arange(500, signal.size-1000) / self.sampling_frequency

        fig, (ax1_2, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        plt.figure(figsize=(10, 6))
        ax1_2.plot(time_axis, original_signal_excluded, label="Original Signal", color="blue")
        # ax1_2.plot(self.time[exclude_start:exclude_end], filtered_signal[exclude_start:exclude_end],
        #            label="Filtered Signal", color="red")

        ax1_2.set_ylabel("Amplitude")
        ax1_2.legend()

        ax3.plot(time_axis, filtered_signal,
                 label="Filtered Signal", color="green")

        ax3.set_xlabel("Time(seconds)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()

        plt.title(f'Signal Plot ')

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_filtered_signal_for_duration(self, signal_number=0, lowcut=0.8, highcut=4, duration=10):
        signal = self.signals[signal_number]
        signal_mV = signal
        filtered_signal = self.butter_bandpass_filter(signal, lowcut, highcut, self.sampling_frequency)
        n_samples_duration = int(duration * self.sampling_frequency)
        original_signal_excluded = signal[:n_samples_duration]
        time_axis = np.arange(0, n_samples_duration) / self.sampling_frequency

        plt.figure(figsize=(15, 4))
        plt.plot(time_axis, original_signal_excluded, color='blue', label='Original Signal')
        plt.title(f'Signal {signal_number+1}: Filtered (Butterworth Bandpass) for First {duration} Seconds')
        plt.ylabel('Amplitude (mV)')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def power_density_fft(self, signal):

        n = len(signal)
        fs = self.sampling_frequency
        frequencies = np.fft.rfftfreq(n, d=1 / fs)
        fft_result = np.fft.rfft(signal)
        psd = np.abs(fft_result) ** 2 / (fs * n)

        max_index = np.argmax(psd)
        print("Maximum power/frequency(fft):", psd[max_index])
        print("Frequency with maximum power/frequency(fft):", frequencies[max_index])

        plt.figure(figsize=(10, 6))
        plt.semilogy(frequencies, psd)
        plt.scatter(frequencies[max_index], psd[max_index], color='red', label=f'Maximum at {frequencies[max_index]:.2f} Hz')
        plt.title('Power Spectral Density of Signal(FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def power_density_welch(self, signal):
        fs = self.sampling_frequency
        frequencies, psd = welch(signal, fs=fs, nperseg=1024)
        max_index = np.argmax(psd)
        print("Maximum power/frequency(welch):", psd[max_index])
        print("Frequency with maximum power/frequency(welch):", frequencies[max_index])

        # plt.figure(figsize=(10, 6))
        # plt.semilogy(frequencies, psd)
        # plt.scatter(frequencies[max_index], psd[max_index], color='red', label=f'Maximum at {frequencies[max_index]:.2f} Hz')
        # plt.title('Power Spectral Density of Signal(Welch)')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power/Frequency (dB/Hz)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        return psd[max_index], frequencies[max_index]

    def process(self, signal_number = 0):
        # fs = self.sampling_frequency

        signal = self.signals[:, signal_number]
        print(len(signal))
        signals = self.record.p_signal
        print(self.signals.shape[0])
        # signal = signal / self.adc_resolution

        # signal_mV = signal
        target_fs = 20
        decimation_factor = self.sampling_frequency // target_fs
        decimated_signal, fs = self.decimate_signal(signal, self.sampling_frequency, target_fs, decimation_factor)
        # self.plot_single_signal(decimated_signal, fs)
        # print(fs)
        # Take the first 30 minutes of the signal
        # duration_seconds = 3 * 60  # 30 minutes in seconds
        # signal_duration_samples = int(duration_seconds * fs)
        # filtered_signal = filtered_signal[:signal_duration_samples]

        # normal_signal = copy.copy(filtered_signal)

        self.time = np.arange(0, len(decimated_signal)) / fs
        # Define the filter parameters
        # lowcut = 0.01667
        #
        # highcut = 3
        #
        # nyquist_freq = 0.5 * fs
        # low = lowcut / nyquist_freq
        # high = highcut / nyquist_freq
        #
        # # Compute the filter coefficients
        # order = 4
        # b, a = butter(order, [low, high], btype='band')
        #
        # butter_filter = filtfilt(b, a, filtered_signal)
        # med_filtered_signal = medfilt(decimated_signal, kernel_size=3)
        # print(med_filtered_signal)
        # self.show(fs, normal_signal, filtered_signal, med_filtered_signal)
        self.concave_signal = copy.copy(decimated_signal)

        window_width = 120  # in seconds
        step_size = 1  # in seconds

        # Calculate the number of samples in each window
        window_size = int(window_width * fs)
        step_size_samples = int(step_size * fs)

        # Calculate zero crossing rate for each segment
        self.zero_crossing_rates = []
        self.timestamps = []
        power_zero_crossing = []
        cnt_1 = 0
        cnt_0 = 0
        for i in decimated_signal:
            if i > 0:
                cnt_1 = cnt_1 + 1
            else:
                cnt_0 = cnt_0 + 1
        # print(cnt_0)
        # print(cnt_1)
        for i in range(0, len(decimated_signal) - window_size + 1, step_size_samples):
            # segment = self.signal_data[i:i + window_size, self.signal_index]
            segment = decimated_signal[i: i + window_size]
            zero_crossings = np.where(np.diff(np.sign(segment)))[0]
            # print(zero_crossings)
            zero_crossing_rate = len(zero_crossings) / window_width
            self.zero_crossing_rates.append(zero_crossing_rate)
            power_zero_crossing.append(pow(zero_crossing_rate, 1.2))
            self.timestamps.append(self.time[i])
        # print(power_zero_crossing)
        # Normalize the power of zero crossing rates
        normalized_power = preprocessing.normalize([power_zero_crossing])[0]

        # Interpolate the normalized power to match the length of the signal
        interpolator = interp1d(np.array(self.timestamps), normalized_power, kind='linear', fill_value="extrapolate")
        self.interpolated_normalized_power = interpolator(self.time)

        # Modulate the original signal with the interpolated normalized power
        self.modulated_signal = decimated_signal * self.interpolated_normalized_power
        print(fs)
        N = int(40 * fs)

        self.rms_values = []
        # print(fs)
        print(len(self.modulated_signal))
        for i in range(0, len(self.modulated_signal)):
            j = i
            rms = 0.0

            for j in range(0, N):
                if i + j >= len(self.modulated_signal):
                    break
                rms = rms + (self.modulated_signal[i + j] * self.modulated_signal[i + j])

            rms = rms / N
            rms = sqrt(rms)
            self.rms_values.append(rms)

        # print(self.rms_values)
        # interval_duration = 40  # seconds
        # interval_samples = int(interval_duration * fs)
        #
        #
        # for i in range(0, len(med_filtered_signal), interval_samples):
        #     interval = med_filtered_signal[i:i + interval_samples]
        #     rms = np.sqrt(np.mean(interval ** 2))
        #     self.rms_values.append(rms)
        tmp_rms = self.rms_values
        sorted_rms = sorted(tmp_rms)

        signal_range = sorted_rms[len(sorted_rms) - 1] - sorted_rms[0]
        length = len(sorted_rms) // 10

        mean = 0
        for i in range(0, length):
            mean = mean + sorted_rms[i]

        mean = mean / length

        # mean_value = np.mean(self.modulated_signal)
        # std_deviation = np.std(self.modulated_signal)
        # h = 2

        # self.threshold = mean_value + h * std_deviation
        # t = mean + h * std
        print(N)
        self.threshold = 1.2 * (mean + 0.25 * (signal_range))
        print("Threshold = ", self.threshold)
        l = -1
        r = -1
        self.contraction_array = []

        self.contraction_segments = []
        # print(self.rms_values)
        for i in range(0, len(self.rms_values)):
            self.new_contraction_array.append(self.threshold)
            self.contraction_array.append(self.threshold)

            if l == -1:
                if self.rms_values[i] >= self.threshold:

                    l = i

                    continue

            if l != -1 and r == -1:
                if self.rms_values[i] <= self.threshold:

                    r = i
                    print("l = ", l, " r = ", r)
                    duration = r - l + 1
                    duration = duration / fs
                    print(duration)
                    segment = []
                    peak_value = 0
                    peak_idx = 0
                    if duration >= 30:
                        # print("l = " + str(l/fs) + " : r = " + str(r/fs))
                        for j in range(l, r + 1):
                            self.contraction_array[j] = self.rms_values[j]
                            segment.append(self.rms_values[j])
                            if self.rms_values[j] > peak_value:
                                peak_value = self.rms_values[j]
                                peak_idx = j
                    new_contraction_segment = []
                    if duration >= 30 and duration <= 100:
                        for j in range(l, r + 1):
                            new_contraction_segment.append(self.rms_values[j])
                            self.new_contraction_array[j] = self.rms_values[j]
                        self.contraction_segments.append(new_contraction_segment)
                    elif duration > 100:
                        fifty = int(50 * fs)
                        hundred = int(100 * fs)
                        lft = max(l, peak_idx - fifty)
                        rht = lft + hundred
                        mn = min(rht + 1, len(self.rms_values))
                        for j in range(lft, mn):
                            new_contraction_segment.append(self.rms_values[j])
                            # self.new_contraction_array[j] = self.rms_values[j]
                        self.contraction_segments.append(new_contraction_segment)

                    l = -1
                    r = -1

    def plot_rms_values(self):
        time_axis_rms = np.arange(0, len(self.rms_values)) / 20
        time_axis_signal = np.arange(0, len(self.modulated_signal)) / 20

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        axs[0].plot(time_axis_rms, self.rms_values, color='blue', label='RMS Values')
        axs[0].set_title('RMS Values')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(time_axis_signal, self.modulated_signal, color='red', label='Modulated Signal')
        axs[1].set_title('Modulated Signal')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def contraction_segments_power_density_welch(self):
        # print(self.contraction_segments)
        features = []
        for i, segment in enumerate(self.contraction_segments):
            # print(f"Contraction Segment {i + 1}:")
            max_power_frequency, frequency_with_max_power = self.power_density_welch(segment)
            tfdomain_features = TFdomainFeaturesExtract(segment, 20)
            energy, crest_factor, mean_frequency, median_frequency, peak_to_peak_amplitude, contraction_intensity, contraction_power, shannon_entropy, sample_entropy, Dispersion_entropy, log_detector = tfdomain_features.getFeatures()
            features.append([max_power_frequency, frequency_with_max_power, energy, crest_factor, mean_frequency, median_frequency, peak_to_peak_amplitude, contraction_intensity, contraction_power, shannon_entropy, sample_entropy, Dispersion_entropy, log_detector])
        return features

    def concave_signal_features(self):
        ConcaveHull = AlphaConcaveHull(self.concave_signal, 1.785)
        features = ConcaveHull.execute()
        return [features]
# Example usage:

# header_file = 'F:/signal/dataset/later_cesarean/over_weight/icehg675.hea'
# dat_file = 'F:/signal/dataset/later_cesarean/over_weight/icehg675.dat'
signal_name = 'F:/signal/dataset/later_cesarean/over_weight/ice043_p_1of2'
# adc_resolution = 131.068
#
# # Create an instance of SignalManipulation
signal_manipulator = SignalManipulation(signal_name)
# # signal_manipulator.plot_single_signal()
signal_manipulator.process(1)
# signal_manipulator.concave_signal_features()

# # signal_manipulator.plot_rms_values()
# signal_manipulator.contraction_segments_power_density_welch()
# Use the methods of the SignalManipulation instance as needed
# signal_manipulator.plot_signals()
# signal_manipulator.plot_single_signal(signal_number=0)
# signal_manipulator.plot_filtered_signal_with_original(signal_number=0)
# signal_manipulator.plot_filtered_signal_for_duration(signal_number=0)
# signal_manipulator.power_density_fft(signal_number=1)
# signal_manipulator.power_density_welch(signal_number=1)
