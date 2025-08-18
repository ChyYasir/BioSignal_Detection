import os
import csv
from Signal_manipulation import SignalManipulation

# Directory containing the signal files
signal_type = "later_spontaneous"
signal_bmi = "under_weight"
signal_directory = f"F:/signal/dataset/{signal_type}/{signal_bmi}/"

# Create a CSV file name for zero crossing rates
csv_file_name = f"{signal_type}_{signal_bmi}_zero_crossing_rates.csv"

# Open the CSV file in write mode in the current directory
csv_file_path = csv_file_name

# First pass: determine the maximum length of zero crossing rates array
# This is needed to create proper CSV headers
max_zcr_length = 0
print("Determining maximum ZCR array length...")

for filename in os.listdir(signal_directory):
    if filename.endswith(".hea"):
        signal_name = f"{signal_directory}{os.path.splitext(filename)[0].split('.')[0]}"

        try:
            signal_manipulator = SignalManipulation(str(signal_name))

            # Check the sampling frequency and determine which signals to process
            if signal_manipulator.sampling_frequency == 20:
                signal_indices = [1, 3, 5]
            else:
                signal_indices = range(16)

            # Check one signal to get the ZCR array length
            signal_manipulator.process(signal_number=signal_indices[0])
            zcr_array = signal_manipulator.get_zero_crossing_rates_array()

            if len(zcr_array) > max_zcr_length:
                max_zcr_length = len(zcr_array)

        except Exception as e:
            print(f"Error checking {filename}: {str(e)}")
            continue

print(f"Maximum ZCR array length found: {max_zcr_length}")

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as csv_file:
    # Define the CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    header = ['Filename', 'Signal_Index']
    # Add column headers for each zero crossing rate measurement
    for i in range(max_zcr_length):
        header.append(f'ZCR_{i + 1}')

    csv_writer.writerow(header)

    # Iterate over all files in the directory
    for filename in os.listdir(signal_directory):
        if filename.endswith(".hea"):
            # Extract the signal name from the header file name
            signal_name = f"{signal_directory}{os.path.splitext(filename)[0].split('.')[0]}"
            print(f"Processing: {signal_name}")

            try:
                # Create an instance of SignalManipulation
                signal_manipulator = SignalManipulation(str(signal_name))

                # Check the sampling frequency
                if signal_manipulator.sampling_frequency == 20:
                    # Process signals with index 1, 3, and 5
                    for signal_index in [1, 3, 5]:
                        print(f"  Processing signal index: {signal_index}")

                        try:
                            # Process the signal to calculate zero crossing rates
                            signal_manipulator.process(signal_number=signal_index)

                            # Get the zero crossing rates array
                            zcr_array = signal_manipulator.get_zero_crossing_rates_array()

                            if zcr_array:
                                # Create the row: filename, signal_index, then all ZCR values
                                row = [filename, signal_index] + zcr_array

                                # Pad with empty values if this array is shorter than max_zcr_length
                                while len(row) < len(header):
                                    row.append('')

                                print(f"    Writing {len(zcr_array)} zero crossing rates")
                                csv_writer.writerow(row)
                            else:
                                print(f"    Warning: No zero crossing rates found for signal {signal_index}")

                        except Exception as e:
                            print(f"    Error processing signal {signal_index}: {str(e)}")
                            continue

                else:
                    # Process all signals from index 0 to 15
                    for signal_index in range(16):
                        print(f"  Processing signal index: {signal_index}")

                        try:
                            # Process the signal to calculate zero crossing rates
                            signal_manipulator.process(signal_number=signal_index)

                            # Get the zero crossing rates array
                            zcr_array = signal_manipulator.get_zero_crossing_rates_array()

                            if zcr_array:
                                # Create the row: filename, signal_index, then all ZCR values
                                row = [filename, signal_index] + zcr_array

                                # Pad with empty values if this array is shorter than max_zcr_length
                                while len(row) < len(header):
                                    row.append('')

                                print(f"    Writing {len(zcr_array)} zero crossing rates")
                                csv_writer.writerow(row)
                            else:
                                print(f"    Warning: No zero crossing rates found for signal {signal_index}")

                        except Exception as e:
                            print(f"    Error processing signal {signal_index}: {str(e)}")
                            continue

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

print(f"\nZero crossing rates results saved to {csv_file_path}")
print(f"Each row contains the complete zero crossing rates array for one signal")
print(f"CSV structure: Filename, Signal_Index, ZCR_1, ZCR_2, ..., ZCR_{max_zcr_length}")