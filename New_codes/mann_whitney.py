import pandas as pd
from scipy.stats import mannwhitneyu

column_headers = ["area", "perimeter", "circularity",  "variance", "bending_energy", "energy", "crest_factor", "mean_frequency", "median_frequency", "peak_to_peak_amplitude", "contraction_intensity", "contraction_power", "shannon_entropy", "sample_entropy","Dispersion_entropy" , "log_detector"]

# Load the data from CSV files with the specified headers
data1 = pd.read_csv('early_cesarean_features.csv', header=None, names=column_headers)
data2 = pd.read_csv('early_induced-cesarean_features.csv', header=None, names=column_headers)

# Display the DataFrames
# print("Data 1:")
# print(data1)
# print("\nData 2:")
# print(data2)

# Take the first 8 rows of data2
data1_subset = data1.head(17)
data2_subset = data2.head(17)


for column in column_headers:
    U1, p_value = mannwhitneyu(data1[column], data2_subset[column])
    print(f"\nMann-Whitney U test for {column}:")
    print(f"U1: {U1}")
    print(f"P-value: {p_value}")
