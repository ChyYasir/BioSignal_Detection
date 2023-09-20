import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, medfilt
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
import pywt


# Import the interpolation function
from math import sqrt
# Load the record and extract the signal data
record = wfdb.rdrecord('D:/term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_p008')
signal_data = record.p_signal

# Extract sampling frequency
fs = record.fs
print(fs)
# Create time axis in seconds for the entire signal
time = np.arange(0, signal_data.shape[0]) / fs

# Choose the signal index you want to analyze
signal_index = 1  # Change this to the desired signal index

# Define the sliding window width and step size
window_width = 120  # in seconds
step_size = 1  # in seconds

# Calculate the number of samples in each window
window_size = int(window_width * fs)
step_size_samples = int(step_size * fs)

# Calculate zero crossing rate for each segment
zero_crossing_rates = []
timestamps = []
power_zero_crossing = []

# Define the filter parameters
lowcut = 0.08
highcut = 4

nyquist_freq = 0.5 * fs
low = lowcut / nyquist_freq
high = highcut / nyquist_freq

# Compute the filter coefficients
order = 4
b, a = butter(order, [low, high], btype='band')

# Apply the filter to the signal
filtered_signal = filtfilt(b, a, signal_data[:, signal_index])

med_filtered_signal = medfilt(filtered_signal, kernel_size=3)
for i in range(0, signal_data.shape[0] - window_size + 1, step_size_samples):
    # segment = signal_data[i:i + window_size, signal_index]
    segment = med_filtered_signal[i: i + window_size]
    zero_crossings = np.where(np.diff(np.sign(segment)))[0]
    zero_crossing_rate = len(zero_crossings) / window_width
    zero_crossing_rates.append(zero_crossing_rate)
    power_zero_crossing.append(pow(zero_crossing_rate, 1.2))
    timestamps.append(time[i])

# Normalize the power of zero crossing rates
normalized_power = preprocessing.normalize([power_zero_crossing])[0]

# Interpolate the normalized power to match the length of the signal
interpolator = interp1d(np.array(timestamps), normalized_power, kind='linear', fill_value="extrapolate")
interpolated_normalized_power = interpolator(time)

# Modulate the original signal with the interpolated normalized power
modulated_signal = signal_data[:, signal_index] * interpolated_normalized_power
# print(len(zero_crossing_rates))
# print(len(interpolated_normalized_power))
# print(len(modulated_signal))

fourier = fft(modulated_signal)

fourier_x = []
fourier_y = []

fourier_x = [ele.real for ele in fourier]
fourier_y = [ele.imag for ele in fourier]

points = [[ele.real, ele.imag] for ele in fourier]
arr_points = np.array(points)
hull = ConvexHull(arr_points)



N = int(40 *fs)

rms_values = []
for i in range(0, len(modulated_signal)):
    j = i
    rms = 0.0
    for j in range(0, 40 * fs):
        if i + j >= len(modulated_signal):
            break
        rms = rms + (modulated_signal[i + j] * modulated_signal[i+j])

    rms = rms / N
    rms = sqrt(rms)
    rms_values.append(rms)
# Plot the original signal, power of zero crossing rates, and the modulated signal
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
#
# ax1.plot(time, med_filtered_signal, label='Original Signal')
# ax1.set_ylabel('Amplitude')
# ax1.set_title(f'Signal: {record.sig_name[signal_index]}')
# ax1.grid()
#
# ax2.plot(timestamps, zero_crossing_rates, color='r', label='Zero Crossing Rate')
# ax2.set_ylabel('Zero Crossing Rate')
# ax2.set_title('Zero Crossing Rate')
# ax2.grid()
#
# ax3.plot(time, interpolated_normalized_power, color='g', label='Interpolated Normalized Power')
# ax3.set_ylabel('Interpolated Power')
# ax3.set_title('Interpolated Normalized Power')
# ax3.grid()
#
# ax4.plot(time, modulated_signal, color='b', label='Modulated Signal')
# ax4.set_ylabel('Amplitude')
# ax4.set_title('Modulated Signal')
# ax4.grid()
#
# ax5.plot(time, signal_data[:, 6], label='Original Signal')
# ax5.set_ylabel('Amplitude')
# ax5.set_title(f'Signal: {record.sig_name[6]}')
# ax5.grid()
#
# ax6.plot(time, rms_values, label='RMS')
# ax6.set_ylabel('Amplitude')
# ax6.set_xlabel('Time (seconds)')
# ax6.set_title("RMS")
# ax6.grid()


plt.scatter(fourier_x, fourier_y)

for simplex in hull.simplices:
    plt.plot(arr_points[simplex, 0], arr_points[simplex, 1], 'k-')
# plt.xlim(-0.4, 0.4, 0.2)
plt.tight_layout()
plt.show()