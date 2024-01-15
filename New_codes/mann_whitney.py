import pandas as pd
from scipy.stats import mannwhitneyu

column_headers = ["area", "perimeter", "circularity", "convexity", "variance", "bending_energy"]

# Load the data from CSV files with the specified headers
data1 = pd.read_csv('later_cesarean_features.csv', header=None, names=column_headers)
data2 = pd.read_csv('later_Spontaneous_features.csv', header=None, names=column_headers)

# Display the DataFrames
# print("Data 1:")
# print(data1)
# print("\nData 2:")
# print(data2)

# Take the first 8 rows of data2
data2_subset = data2.head(8)


for column in column_headers:
    U1, p_value = mannwhitneyu(data1[column], data2_subset[column])
    print(f"\nMann-Whitney U test for {column}:")
    print(f"U1: {U1}")
    print(f"P-value: {p_value}")
