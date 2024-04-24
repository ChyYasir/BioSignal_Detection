import os
import csv
from Signal_manipulation import SignalManipulation

# Directory containing the signal files

signal_type = "later_cesarean"
signal_bmi = "over_weight"
signal_directory = f"F:/signal/dataset/{signal_type}/{signal_bmi}/"

# Extract the parent directory name for the CSV file name
# parent_directory_name = os.path.basename(os.path.dirname(signal_directory))
#
# # Extract the current directory name for the CSV file name
# current_directory_name = os.path.basename(signal_directory)

# Create a CSV file name
csv_file_name = f"{signal_type}_{signal_bmi}.csv"

# Open the CSV file in write mode in the current directory
csv_file_path = csv_file_name

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as csv_file:
    # Define the CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Max Power/Frequency', 'Frequency with Max Power'])

    # Iterate over all files in the directory
    for filename in os.listdir(signal_directory):
        if filename.endswith(".hea"):
            header_file = os.path.join(signal_directory, filename)
            # Extract the signal name from the header file name

            signal_name = f"{signal_directory}{os.path.splitext(filename)[0].split('.')[0]}"
            print(signal_name)

            # Create an instance of SignalManipulation
            signal_manipulator = SignalManipulation( str(signal_name))

            # Check the sampling frequency
            if signal_manipulator.sampling_frequency == 20:
                # Process signals with index 1, 3, and 5
                for signal_index in [1, 3, 5]:
                    # Call the power_density_welch method
                    max_power_frequency, frequency_with_max_power = signal_manipulator.power_density_welch(
                        signal_manipulator.signals[:, signal_index])
                    # Write the results to the CSV file
                    csv_writer.writerow([max_power_frequency, frequency_with_max_power])
            else:
                # Process all signals from index 0 to 15
                for signal_index in range(16):
                    # Call the power_density_welch method
                    max_power_frequency, frequency_with_max_power = signal_manipulator.power_density_welch(
                        signal_manipulator.signals[:, signal_index])
                    # Write the results to the CSV file
                    csv_writer.writerow([max_power_frequency, frequency_with_max_power])

print(f"Results saved to {csv_file_path}")
