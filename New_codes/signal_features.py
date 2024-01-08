import wfdb
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import butter, filtfilt, medfilt

from New_codes.concave_hull_fourier import AlphaConcaveHull


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

        # print(self.topological_features())
    def topological_features(self):
        concave_hull = AlphaConcaveHull(self.concave_signal, 1.785)
        features = concave_hull.execute()
        return features



# print(1)
# early_cesarean = SignalProcess("icehg666","F:/signal/dataset/early_cesarean/")
#
# early_cesarean.process();


