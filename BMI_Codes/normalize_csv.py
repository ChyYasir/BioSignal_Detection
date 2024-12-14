import os
import csv
import re
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Helper function: Convert gestational age from w/d to numerical weeks
def convert_gestational_age(gestational_age):
    try:
        weeks, days = map(int, gestational_age.split('/'))
        return weeks + days / 7
    except Exception as e:
        print(f"Error converting gestational age '{gestational_age}': {e}")
        return None


# Helper function: Extract features from a single .hea file
def extract_features(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()

            # Extract features using regex
            age = int(re.search(r"Age\(years\):(\d+)", data).group(1))
            bmi_before = float(re.search(r"BMI before pregnancy:(\d+\.\d+)", data).group(1))
            bmi_at_recording = float(re.search(r"BMI at recording:(\d+\.\d+)", data).group(1))
            gravidity = int(re.search(r"Gravidity:(\d+)", data).group(1))
            parity = int(re.search(r"Parity:(\d+)", data).group(1))
            prev_c_section = 1 if re.search(r"Previous caesarean:Yes", data) else 0
            placental_position = 1 if re.search(r"Placental position:Posterior", data) else 0
            gestational_age_recording = convert_gestational_age(
                re.search(r"Gestational age at recording\(w/d\):(\d+/\d+)", data).group(1))
            gestational_age_delivery = convert_gestational_age(
                re.search(r"Gestational age at delivery:(\d+/\d+)", data).group(1))
            synthetic_oxytocin = 1 if re.search(r"Synthetic oxytocin use in labour:Yes", data) else 0

            # Return the extracted features
            return [
                os.path.basename(file_path),
                age,
                bmi_before,
                bmi_at_recording,
                gravidity,
                parity,
                prev_c_section,
                placental_position,
                gestational_age_recording,
                gestational_age_delivery,
                synthetic_oxytocin
            ]
    except (AttributeError, ValueError) as e:
        print(f"Error parsing file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in file '{file_path}': {e}")
        return None

# Helper function: Append features from a directory to a CSV file
# Helper function: Append features from a directory to a CSV file
def process_directory(signal_directory, csv_file_path):
    try:
        # Define the header for the CSV file
        header = [
            "File",
            "Age",
            "BMI Before Pregnancy",
            "BMI At Recording",
            "Gravidity",
            "Parity",
            "Previous Caesarean",
            "Placental Position",
            "Gestational Age at Recording (weeks)",
            "Gestational Age at Delivery (weeks)",
            "Synthetic Oxytocin Use"
        ]

        # Check if the CSV file exists
        file_exists = os.path.isfile(csv_file_path)

        # Open the CSV file in append mode
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header if the file doesn't exist
            if not file_exists:
                writer.writerow(header)

            # Process each .hea file in the directory
            for filename in os.listdir(signal_directory):
                if filename.endswith(".hea"):
                    file_path = os.path.join(signal_directory, filename)
                    features = extract_features(file_path)
                    if features:  # Only write valid feature sets
                        writer.writerow(features)
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"Unexpected error in directory '{signal_directory}': {e}")


# Normalize a feature value with a custom range
def normalize_feature(value, min_val, max_val, range_min, range_max):
    try:
        x_std = (value - min_val) / (max_val - min_val)  # Standardized to [0, 1]
        x_scaled = x_std * (range_max - range_min) + range_min  # Scale to [range_min, range_max]
        return round(x_scaled, 2)
    except ZeroDivisionError:
        print(f"Invalid range for normalization: min={min_val}, max={max_val}")
        return None


# Compute min and max values for each feature dynamically
def compute_min_max(csv_file_path):
    try:
        dataframe = pd.read_csv(csv_file_path)
        min_max_values = {}
        for column in dataframe.columns[1:]:  # Skip "File" column
            min_max_values[column] = (dataframe[column].min(), dataframe[column].max())
        return min_max_values
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


# Extract features and normalize with different ranges
def extract_and_normalize_features(file_path, min_max_values):
    with open(file_path, 'r') as file:
        data = file.read()

        # Extract raw feature values
        age = int(re.search(r"Age\(years\):(\d+)", data).group(1))
        bmi_before = float(re.search(r"BMI before pregnancy:(\d+\.\d+)", data).group(1))
        bmi_at_recording = float(re.search(r"BMI at recording:(\d+\.\d+)", data).group(1))
        gravidity = int(re.search(r"Gravidity:(\d+)", data).group(1))
        parity = int(re.search(r"Parity:(\d+)", data).group(1))
        prev_c_section = 1 if re.search(r"Previous caesarean:Yes", data) else 0
        placental_position = 1 if re.search(r"Placental position:Posterior", data) else 0
        gestational_age_recording = float(
            re.search(r"Gestational age at recording\(w/d\):(\d+/\d+)", data).group(1).split('/')[0]
        )
        gestational_age_delivery = float(
            re.search(r"Gestational age at delivery:(\d+/\d+)", data).group(1).split('/')[0]
        )
        synthetic_oxytocin = 1 if re.search(r"Synthetic oxytocin use in labour:Yes", data) else 0

        # Normalize features using dynamically computed min-max and unique ranges
        features = [
            os.path.basename(file_path),
            normalize_feature(age, *min_max_values["Age"], 1, 10),
            normalize_feature(bmi_before, *min_max_values["BMI Before Pregnancy"], 1, 10),
            normalize_feature(bmi_at_recording, *min_max_values["BMI At Recording"], 1, 10),
            normalize_feature(gravidity, *min_max_values["Gravidity"], 1, 8),
            normalize_feature(parity, *min_max_values["Parity"], 0, 8),
            normalize_feature(prev_c_section, 0, 1, 0, 1),  # Binary feature
            normalize_feature(placental_position, 0, 1, 0, 1),  # Binary feature
            normalize_feature(gestational_age_recording, *min_max_values["Gestational Age at Recording (weeks)"], 1, 10),
            normalize_feature(gestational_age_delivery, *min_max_values["Gestational Age at Delivery (weeks)"], 1, 10),
            normalize_feature(synthetic_oxytocin, 0, 1, 0, 1),  # Binary feature
        ]
        return features

# Process directory and append normalized data to CSV
def normalize_csv(signal_directory, csv_file_path, min_max_values):
    header = [
        "File",
        "Age",
        "BMI Before Pregnancy",
        "BMI At Recording",
        "Gravidity",
        "Parity",
        "Previous Caesarean",
        "Placental Position",
        "Gestational Age at Recording (weeks)",
        "Gestational Age at Delivery (weeks)",
        "Synthetic Oxytocin Use"
    ]

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        for filename in os.listdir(signal_directory):
            if filename.endswith(".hea"):
                file_path = os.path.join(signal_directory, filename)
                features = extract_and_normalize_features(file_path, min_max_values)
                writer.writerow(features)
# Main execution
if __name__ == "__main__":
    # Define the signal directory structure
    signal_type = "later_cesarean"
    signal_bmi = "healthy_obese"
    signal_directory = f"F:/signal/files/{signal_type}"

    # Create a CSV file name based on directory structure
    csv_file_name = f"{signal_type}_OF.csv"
    csv_file_path = csv_file_name  # Save in the current working directory

    # Process the directory and append features to the CSV file
    process_directory(signal_directory, csv_file_path)

    min_max_values = compute_min_max(csv_file_path)
    # Normalize the features and save to a new CSV file

    normalize_csv_file = f"{signal_type}_normalize.csv"
    normalize_csv(signal_directory, normalize_csv_file, min_max_values)
