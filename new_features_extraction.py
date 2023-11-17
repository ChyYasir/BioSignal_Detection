import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

class NewFeaturesExtract:
    def __init__(self, signal, sampling_frequency):
        self.signal = signal
        self.sampling_frequency = sampling_frequency


    def plot_psd(self, frequencies, power_spectrum, mean_frequency):
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, power_spectrum)
        plt.title("Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")

        # Highlight the Mean Frequency
        plt.axvline(mean_frequency, color='r', linestyle='--', label=f"Mean Frequency: {mean_frequency:.2f} Hz")
        plt.legend()

        plt.grid()
        plt.show()

    def sample_entropy(self, time_series, m, r):
        """
        Calculate the sample entropy of a time series.

        Parameters:
        - time_series: The input time series as a 1D NumPy array.
        - m: The length of compared runs of data.
        - r: Tolerance level, which is a fraction of the standard deviation of the time series.

        Returns:
        - SampEn: The sample entropy value.
        """

        N = len(time_series)

        # Define a function to calculate the maximum distance between two vectors
        def max_dist(x_i, x_j):
            return max([abs(x_i[k] - x_j[k]) for k in range(len(x_i))])

        def count_matches(template, data, m):
            count = 0
            for i in range(len(data) - m + 1):
                match = True
                for j in range(m):
                    if abs(template[j] - data[i + j]) > r:
                        match = False
                        break
                if match:
                    count += 1
            return count

        # Compute Bm(x, r) for m and m+1
        A_m = count_matches(time_series, time_series, m)
        A_m_plus_1 = count_matches(time_series, time_series, m + 1)

        # Calculate the probability of matching for m and m+1
        p_m = A_m / N
        p_m_plus_1 = A_m_plus_1 / N

        # Avoid division by zero
        if p_m == 0 or p_m_plus_1 == 0:
            return 0

        # Calculate sample entropy
        samp_en = -np.log(p_m_plus_1 / p_m)

        return samp_en

    def getFeatures(self):
        #For Energy
        energy = sum(x ** 2 for x in self.signal)

        # For Crest Factor
        signal = np.array(self.signal)
        peak_amplitude = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal ** 2))
        crest_factor = peak_amplitude / rms

        # For Mean Frequency
        signal = np.array(self.signal)

        # Perform a spectral analysis using Welch's method to obtain the power spectrum
        frequencies, power_spectrum = welch(signal, fs=self.sampling_frequency)


        mean_frequency = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)

        # For Median Frequency
        cumulative_power_spectrum = np.cumsum(power_spectrum)

        k = 0
        for i in range(len(cumulative_power_spectrum)):
            if cumulative_power_spectrum[i] >= (cumulative_power_spectrum[-1]/2.0):
                k = i
                break
        N = len(signal)
        median_frequency = (k * self.sampling_frequency)/N

        # For Peak-to-Peak amplitude
        peak_to_peak_amplitude = np.max(signal) - np.min(signal)

        # For Shannon Entropy
        squared_values = signal ** 2

        shannon_entropy = np.sum(squared_values * np.log2(squared_values))


        #For Sample Entropy
        m = 2
        r = 0.2

        sample_entropy = self.sample_entropy(signal, m, r)

        #For Time Revers
        return [energy, crest_factor, mean_frequency, median_frequency, peak_to_peak_amplitude, shannon_entropy, sample_entropy]
