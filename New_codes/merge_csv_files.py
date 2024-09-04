import pandas as pd

# Load the CSV files into DataFrames
induced_df = pd.read_csv("early_induced_features.csv")
spontaneous_df = pd.read_csv("early_spontaneous_features.csv")

# Merge the two DataFrames
merged_df = pd.concat([induced_df, spontaneous_df])

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("normal_features.csv", index=False)

print("Files have been merged successfully into normal_features.csv")
