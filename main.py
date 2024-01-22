import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, medfilt
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Point
import pywt
from concave_hull_fourier import AlphaConcaveHull
from new_features_extraction import NewFeaturesExtract
# Import the interpolation function
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
    def show(self):
        # plt.figure(1)

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig2, (ax4, ax5, ax6, ax7, ax8) = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

        # For the first window
        ax1.plot(self.time, self.signal_data[:, self.signal_index], label='Original Signal')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Signal: {self.record.sig_name[self.signal_index]}')
        ax1.grid()

        ax2.plot(self.timestamps, self.zero_crossing_rates, color='r', label='Zero Crossing Rate')
        ax2.set_ylabel('Zero Crossing Rate')
        ax2.set_title('Zero Crossing Rate')
        ax2.grid()

        ax3.plot(self.time, self.interpolated_normalized_power, color='g', label='Interpolated Normalized Power')
        ax3.set_ylabel('Interpolated Power')
        ax3.set_title('Interpolated Normalized Power')
        ax3.grid()

        # For the second window
        ax4.plot(self.time, self.modulated_signal, color='b', label='Modulated Signal')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Modulated Signal')
        ax4.grid()

        ax5.plot(self.time, self.signal_data[:, 6], label='Original Signal')
        ax5.set_ylabel('Amplitude')
        ax5.set_title(f'Signal: {self.record.sig_name[6]}')
        ax5.axhline(y=0.01, color='r')
        ax5.grid()

        ax6.plot(self.time, self.rms_values, label='RMS')
        ax6.set_ylabel('Amplitude')
        # ax6.set_xlabel('self.Time (seconds)')
        ax6.set_title("RMS")
        ax6.grid()
        ax6.axhline(y=self.threshold, color='r')

        ax7.plot(self.time, self.contraction_array, label='RMS')
        ax7.set_ylabel('Amplitude')
        # ax7.set_xlabel('self.Time (seconds)')
        ax7.set_title("Contraction")
        ax7.grid()

        ax8.plot(self.time, self.new_contraction_array, label='New Contraction Array')
        ax8.set_ylabel('Amplitude')
        ax8.set_xlabel('self.Time (seconds)')
        ax8.set_title("New Contraction")
        ax8.grid()
        # You can choose to add more subplots as needed

        # Display the two windows
        plt.show()

    def process(self):
        # Load the self.record and extract the signal data


        # Extract sampling frequency
        fs = self.record.fs
        # print(fs)
        # Create self.time axis in seconds for the entire signal
        self.time = np.arange(0, self.signal_data.shape[0]) / fs

        # Choose the signal index you want to analyze
        self.signal_index = 3  # Change this to the desired signal index

        # Define the sliding window width and step size
        window_width = 120  # in seconds
        step_size = 1  # in seconds

        # Calculate the number of samples in each window
        window_size = int(window_width * fs)
        step_size_samples = int(step_size * fs)

        # Calculate zero crossing rate for each segment
        self.zero_crossing_rates = []
        self.timestamps = []
        power_zero_crossing = []

        # Define the filter parameters
        lowcut = 0.01667

        highcut = 3

        nyquist_freq = 0.5 * fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq

        # Compute the filter coefficients
        order = 4
        b, a = butter(order, [low, high], btype='band')

        # print(len(self.signal_data[:, 11]))
        # print(self.signal_data[:, :])
        # Apply the filter to the signal
        for i in range(0, len(self.signal_data[:, self.signal_index])):
            self.signal_data[i, self.signal_index] = self.signal_data[i, self.signal_index]

        # print(self.signal_data[:, self.signal_index])
        filtered_signal = filtfilt(b, a, self.signal_data[:, self.signal_index])

        med_filtered_signal = medfilt(filtered_signal, kernel_size=3)

        # print(len(med_filtered_signal))
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
        #
        #
        # self.threshold = mean_value + h * std_deviation
        # t = mean + h * std

        self.threshold = 1.2 * (mean + 0.25 * (signal_range))

        l = -1
        r = -1
        self.contraction_array = []

        self.contraction_segments = []

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
                        for j in range(lft, rht + 1):
                            new_contraction_segment.append(self.rms_values[j])
                            self.new_contraction_array[j] = self.rms_values[j]
                        self.contraction_segments.append(new_contraction_segment)

                    l = -1
                    r = -1
        # print(len(self.contraction_array))
        # plt.figure(2)

        # self.show()

    def topological_features(self):
        concave_hull = AlphaConcaveHull(self.contraction_segments[0], 1.785)
        features = concave_hull.execute()

        # print("For " + self.signal_name + " = " + str(features))

        return features

    def peak_value(self):
        peak = max(self.contraction_array)
        return peak

    def show_contractions(self):

        for contraction in self.contraction_segments:
            print(int(len(contraction)/self.record.fs))

    def all_segment_topological_features(self):
        result = []
        for i in range(len(self.contraction_segments)):
            concave_hull = AlphaConcaveHull(self.contraction_segments[i], 1.785)
            features = concave_hull.execute()
            result.append(features)

        return result

    def get_new_features(self):
        new_features = NewFeaturesExtract(self.contraction_segments[0], self.record.fs)
        features = new_features.getFeatures()
        return features

    def all_segment_new_features(self):
        result = []
        for i in range(len(self.contraction_segments)):
            new_features = NewFeaturesExtract(self.contraction_segments[i], self.record.fs)
            features = new_features.getFeatures()
            result.append(features)

        return result




# signal1 = SignalProcess("tpehgt_t001", "D:/term-preterm-ehg-dataset-with-tocogram-1.0.0/")
#
#
# signal1.process()
#
# print(signal1.get_new_features())
# signal1.topological_features()
# signal1.show_contractions()
# signal1.show()
# print(signal1.topological_features())
# topologicalFeatures = signal1.topological_features()
# print(topologicalFeatures)
# basal_tone = mean of (10% of lowest values)








# Plot the original signal, power of zero crossing rates, and the modulated signal











# plt.tight_layout()
# plt.show()