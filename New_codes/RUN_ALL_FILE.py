import csv
import os
import pandas as pd
from New_codes.signal_features import SignalProcess


def save_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def add_to_csv(file_path, new_data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(new_data)


class RunFiles:
    def __init__(self, group_name, directory_path):
        self.group_name = group_name
        self.directory_path = directory_path
    def makeFeatureArray(self):

        features_list = []
        # List all files in the directory
        directory_path = self.directory_path
        files = os.listdir(directory_path)
        # print()
        # Iterate through each file in the directory
        cnt = 0
        for file_name in files:
            if file_name.endswith(".dat"):
                # Construct the full file path
                file_path = os.path.join(directory_path, file_name)

                # Split the file name by the dot (.)
                file_parts = file_name.split(".")

                # Get the part before ".hea"
                file_name_without_extension = file_parts[0]
                print(file_name_without_extension)
                try:
                    signal = SignalProcess(file_name_without_extension, directory_path)
                    signal.process()
                    # features = signal.topological_features()
                    # features_list.append(features)

                    # allTopologicalFeatures = signal.all_segment_topological_features()
                    # allNewFeatures = signal.all_segment_new_features()

                    # For contraction segments
                    # allCombinedFeatures = signal.combined_features()
                    # # print(features_list)
                    # for features in allCombinedFeatures:
                    #     print(features)
                    #     features_list.append(features)

                    combined_features = signal.combined_features_signal()
                    print(combined_features)
                    features_list.append(combined_features)

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    cnt = cnt + 1
                    print("Error Count : ", cnt)
                    continue


        print("Total number of Errors: ", cnt)
        save_to_csv(str(self.group_name) + "_features.csv", features_list)

        headers = [
            "area", "perimeter", "circularity", "variance", "bending_energy",
            "energy", "crest_factor", "mean_frequency", "median_frequency",
            "peak_to_peak_amplitude", "contraction_intensity", "contraction_power",
            "shannon_entropy", "sample_entropy", "dispersion_entropy", "log_detector"
        ]

        # Read the CSV file
        df = pd.read_csv(str(self.group_name) + "_features.csv", header=None)

        # Assign column headers
        df.columns = headers

        # Save the DataFrame back to CSV with headers
        df.to_csv(str(self.group_name) + "_features.csv", index=False)


# run_early_cesarean = RunFiles("early_cesarean", "F:/signal/dataset/early_cesarean/")
# run_early_cesarean.makeFeatureArray()

run_early_induced = RunFiles("early_induced", "F:/signal/dataset/early_induced/")
run_early_induced.makeFeatureArray()


# run_early_induced_cesarean = RunFiles("early_induced-cesarean", "F:/signal/dataset/early_induced-cesarean/")
# run_early_induced_cesarean.makeFeatureArray()

# run_early_spontaneous = RunFiles("early_spontaneous", "F:/signal/dataset/early_spontaneous/")
# run_early_spontaneous.makeFeatureArray()


# run_later_spontaneous = RunFiles("later_spontaneous", "F:/signal/dataset/later_spontaneous/")
# run_later_spontaneous.makeFeatureArray()