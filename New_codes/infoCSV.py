import os
import csv

# Directory containing the .hea files
directory = "F:\signal\dataset\later_spontaneous"  # Replace with your directory path

# CSV output file
output_csv = "6_channel_info.csv"

# Function to extract gestation, rectime, and age from a .hea file
def extract_info(hea_file):
    gestation = rectime = age = None
    with open(hea_file, 'r') as file:
        for line in file:
            if "Gestation" in line:
                gestation = line.split()[2]
            elif "Rectime" in line:
                rectime = line.split()[2]
            elif "Age" in line:
                age = line.split()[2]
    return gestation, rectime, age

# List to store the extracted information
data = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".hea"):  # Process only .hea files
        filepath = os.path.join(directory, filename)
        gestation, rectime, age = extract_info(filepath)
        print(gestation, rectime, age)
        if gestation and rectime and age:  # Only append if all values are present
            data.append([filename, gestation, rectime, age])

# Check if the CSV file already exists
file_exists = os.path.isfile(output_csv)

# Write the extracted data to a CSV file (append if exists)
with open(output_csv, 'a', newline='') as csvfile:  # 'a' for append mode
    csvwriter = csv.writer(csvfile)

    # If the file doesn't exist, write the header
    if not file_exists:
        csvwriter.writerow(["File", "Gestation", "Rectime", "Age"])  # Header

    # Append the data
    csvwriter.writerows(data)

print(f"Data has been written to {output_csv}")
