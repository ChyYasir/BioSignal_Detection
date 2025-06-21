import wfdb
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, medfilt, welch
from sklearn import preprocessing

from New_codes.concave_hull_fourier import AlphaConcaveHull
from New_codes.new_features_extraction import NewFeaturesExtract
from math import sqrt

class SignalProcess:
    def __init__(self, signal_name, signal_path):
        self.signal_name = signal_name
        self.path = signal_path + self.signal_name
        self.record = wfdb.rdrecord(self.path)
        self.signal_data = self.record.p_signal
        self.time = []
        self.timestamps = []
        self.zero_crossing_rates = []
        self.interpolated_normalized_power  = []
        self.modulated_signal = []
        self.rms_values = []
        self.contraction_array = []
        self.signal_index = 0
        self.threshold = 0
        self.contraction_segments = []
        self.new_contraction_array = []
        self.concave_signal = []

    def show(self, fs, normal_signal, filtered_signal, custom_filter):
        exclude_start = 150 * fs
        exclude_end = -150 * fs

        fig, (ax1_2, ax3) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

        plt.figure(figsize=(10, 6))
        ax1_2.plot(self.time, normal_signal, label="Normal Signal", color="blue")
        ax1_2.plot(self.time[exclude_start:exclude_end], filtered_signal[exclude_start:exclude_end],
                 label="Filtered Signal", color="red")

        ax1_2.set_ylabel("Amplitude")
        ax1_2.legend()
        ax3.plot(self.time[exclude_start:exclude_end], filtered_signal[exclude_start:exclude_end], label="Filtered Signal", color="red")
        ax3.plot(self.time[exclude_start:exclude_end], custom_filter[exclude_start:exclude_end], label="Custom Filtered Signal", color="green")

        ax3.set_xlabel("Time(seconds)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()


        plt.title(f'Signal Plot for {self.signal_name}')

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def process(self):
        # Load the self.record and extract the signal data
        # Extract sampling frequency
        fs = self.record.fs
        # print(fs)
        # Create self.time axis in seconds for the entire signal
        self.time = np.arange(0, self.signal_data.shape[0]) / fs
        print(self.signal_data.shape[0])
        # Choose the signal index you want to analyze
        normal_signal = self.signal_data[:, 0]
        filtered_signal = self.signal_data[:, 1]

        # Define the filter parameters
        lowcut = 0.01667

        highcut = 3

        nyquist_freq = 0.5 * fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq

        # Compute the filter coefficients
        order = 4
        b, a = butter(order, [low, high], btype='band')

        butter_filter = filtfilt(b, a, filtered_signal)
        med_filtered_signal = medfilt(butter_filter, kernel_size=3)
        # self.show(fs, normal_signal, filtered_signal, med_filtered_signal)
        self.concave_signal = copy.copy(med_filtered_signal)

        window_width = 120  # in seconds
        step_size = 1  # in seconds

        # Calculate the number of samples in each window
        window_size = int(window_width * fs)
        step_size_samples = int(step_size * fs)

        # Calculate zero crossing rate for each segment
        self.zero_crossing_rates = []
        self.timestamps = []
        power_zero_crossing = []
        for i in range(0, self.signal_data.shape[0] - window_size + 1, step_size_samples):
            # segment = self.signal_data[i:i + window_size, self.signal_index]
            segment = med_filtered_signal[i: i + window_size]
            zero_crossings = np.where(np.diff(np.sign(segment)))[0]
            zero_crossing_rate = len(zero_crossings) / window_width
            self.zero_crossing_rates.append(zero_crossing_rate)
            power_zero_crossing.append(pow(zero_crossing_rate, 1.2))
            self.timestamps.append(self.time[i])

        # Normalize the power of zero crossing rates
        normalized_power = preprocessing.normalize([power_zero_crossing])[0]

        # Interpolate the normalized power to match the length of the signal
        interpolator = interp1d(np.array(self.timestamps), normalized_power, kind='linear', fill_value="extrapolate")
        self.interpolated_normalized_power = interpolator(self.time)


        # Modulate the original signal with the interpolated normalized power
        self.modulated_signal = self.signal_data[:, self.signal_index] * self.interpolated_normalized_power

        N = int(40 * fs)

        self.rms_values = []
        # print(fs)
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

        self.threshold = 1.2 * (mean + 0.25 * (signal_range))

        l = -1
        r = -1
        self.contraction_array = []

        self.contraction_segments = []
        print(len(self.modulated_signal))
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
                    duration = r - l + 1
                    duration = duration / fs
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
                        for j in range(l, r+ 1):
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

        # self.power_density_fft(self.concave_signal)
        # self.power_density_welch(self.concave_signal)
        # print(self.topological_features())
    def topological_features(self):
        concave_hull = AlphaConcaveHull(self.concave_signal, 1.785)
        features = concave_hull.execute()
        return features

    def all_segment_topological_features(self):
        result = []
        for i in range(len(self.contraction_segments)):
            concave_hull = AlphaConcaveHull(self.contraction_segments[i], 1.785)
            features = concave_hull.execute()
            result.append(features)

        return result

    def all_segment_new_features(self):
        result = []
        for i in range(len(self.contraction_segments)):
            new_features = NewFeaturesExtract(self.contraction_segments[i], self.record.fs)
            features = new_features.getFeatures()
            result.append(features)

        return result

    def combined_features(self):
        result = []
        for i in range(len(self.contraction_segments)):
            # print(i)
            concave_hull = AlphaConcaveHull(self.contraction_segments[i], 1.785)
            features = concave_hull.execute()
            new_features = NewFeaturesExtract(self.contraction_segments[i], self.record.fs)
            second_features = new_features.getFeatures()
            features.extend(second_features)
            result.append(features)
        return result

    def combined_features_signal(self):
        result = []
        concave_hull = AlphaConcaveHull(self.concave_signal, 1.785)
        feature = concave_hull.execute()
        new_features = NewFeaturesExtract(self.concave_signal, self.record.fs)
        second_features = new_features.getFeatures()
        result = feature + second_features
        return result


    # def power_density_fft(self, signal):
    #     # Compute the FFT
    #     n = len(signal)
    #     fs = self.record.fs
    #     frequencies = np.fft.rfftfreq(n, d=1 / fs)
    #     fft_result = np.fft.rfft(signal)
    #
    #     # Compute the power spectral density
    #     psd = np.abs(fft_result) ** 2 / (fs * n)
    #     # Plot the power spectral density
    #     plt.figure(figsize=(10, 6))
    #     plt.semilogy(frequencies, psd)  # Plot in logarithmic scale
    #     plt.title('Power Spectral Density of Signal(FFT)')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power/Frequency (dB/Hz)')
    #     plt.grid(True)
    #     plt.show()

    def power_density_welch(self, signal):
        fs = self.record.fs
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

    def contraction_segments_power_density_welch(self):

        for i, segment in enumerate(self.contraction_segments):
            print(f"Contraction Segment {i + 1}:")
            self.power_density_welch(segment)

# print(1)
early_cesarean = SignalProcess("icehg675","F:/signal/dataset/later_cesarean/over_weight/")
#
# early_cesarean.process()
print(early_cesarean.signal_data)

# for i in range(5):
#     print("Signal number ", i)
#     for val in early_cesarean.signal_data[:, i]:
#         print(val)
# print(early_cesarean.signal_data[:, 0])
# for val in early_cesarean.signal_data[:, 0]:
#     print(val)
# early_cesarean.contraction_segments_power_density_welch()
# print(early_cesarean.combined_features_signal())


