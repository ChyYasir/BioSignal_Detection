import csv
import os
import numpy as np
from main import SignalProcess

import matplotlib.pyplot as plt


def save_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def add_to_csv(file_path, new_data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(new_data)
class RunTermFiles:
    def __init__(self):
        self.n = 0
    def makeFeatureArray(self):
        # term_features = np.loadtxt("term_features.txt", delimiter="\t")
        # term_features_list = term_features.tolist()
        term_features_list = []
        directory_path = "D:/term-preterm-ehg-dataset-with-tocogram-1.0.0/"
        cnt = 0
        self.n = 13
        for i in range(1, self.n + 1, 1):
            formatted_number = str(i).zfill(3)
            signal_name = "tpehgt_t" + formatted_number
            try:
                signal = SignalProcess(signal_name, directory_path)
                signal.process()

                allNewFeatures = signal.all_segment_new_features()

                for features in allNewFeatures:
                    print(features)
                    term_features_list.append(features)
                # term_features_list.append(topologicalFeatures)
            except Exception as e:
                print(f"Error processing {signal_name}: {str(e)}")
                cnt = cnt + 1
                print("Error Count : ", cnt)
                continue

        # self.save_to_csv("term_features.csv", term_features_list)
        save_to_csv("term_features.csv", term_features_list)






class RunPretermFiles:
    def __init__(self):
        self.n = 0
    def makeFeatureArray(self):
        # preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")
        # preterm_features_list = preterm_features.tolist()
        preterm_features_list = []
        directory_path = "D:/term-preterm-ehg-dataset-with-tocogram-1.0.0/"
        cnt = 0
        self.n = 13
        for i in range(1, self.n+1, 1):
            formatted_number = str(i).zfill(3)
            signal_name = "tpehgt_p" + formatted_number
            try:
                signal = SignalProcess(signal_name, directory_path)
                signal.process()
                allNewFeatures = signal.all_segment_new_features()

                for features in allNewFeatures:
                    print(features)
                    preterm_features_list.append(features)
            except Exception as e:
                print(f"Error processing {signal_name}: {str(e)}")
                cnt = cnt + 1
                print("Error Count : ", cnt)
                continue


        save_to_csv("preterm_features.csv", preterm_features_list)


class RunOldFiles:
    def makeFeatureArray(self):

        term_features_list = []
        preterm_features_list = []
        directory_path = "E:/term-preterm-ehg-database-1.0.1/tpehgdb/"
        # List all files in the directory
        files = os.listdir(directory_path)

        # Iterate through each file in the directory
        cnt = 0
        for file_name in files:
            if file_name.endswith(".hea"):
                # Construct the full file path
                file_path = os.path.join(directory_path, file_name)

                # Initialize a variable to store the Gestation value
                gestation = None
                # Open the .hea file for reading
                with open(file_path, 'r') as file:
                    # Read the file line by line
                    for line in file:
                        # Check if the line contains "Gestation"
                        if "Gestation" in line:
                            # Split the line by spaces and get the last element (the Gestation value)
                            elements = line.strip().split()
                            gestation = elements[-1]
                            break  # Exit the loop once the value is found

                # Split the file name by the dot (.)
                file_parts = file_name.split(".")

                # Get the part before ".hea"
                file_name_without_extension = file_parts[0]
                print(file_name_without_extension + " : " + gestation)

                try:
                    signal = SignalProcess(file_name_without_extension, directory_path)
                    signal.process()

                    # topologicalFeatures = signal.topological_features()
                    # peakValue = signal.peak_value()
                    # topologicalFeatures.append(peakValue)
                    allNewFeatures = signal.all_segment_new_features()
                    for features in allNewFeatures:
                        print(features)
                        if float(gestation) >= 37:
                            term_features_list.append(features)
                        else:
                            preterm_features_list.append(features)
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    cnt = cnt + 1
                    print("Error Count : ", cnt)
                    continue


        print("Total number of Errors: ", cnt)
        add_to_csv("term_features.csv", term_features_list)
        add_to_csv("preterm_features.csv", preterm_features_list)





# run_all_term_files = RunTermFiles()
# run_all_term_files.makeFeatureArray()
#
# run_all_preterm_files = RunPretermFiles()
# run_all_preterm_files.makeFeatureArray()

run_old_files = RunOldFiles()
run_old_files.makeFeatureArray()

# term_features = np.loadtxt("term_features.txt", delimiter="\t")
# preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")
# areas_of_term = term_features[:, 2]
#
# areas_of_preterm = preterm_features[:, 2]
# x_axis_term = [i for i in range(len(term_features))]
# x_axis_preterm = [i for i in range(len(preterm_features))]
# plt.scatter(x_axis_term, areas_of_term, c='blue', marker='o', label='term')
# plt.scatter(x_axis_preterm, areas_of_preterm, c='red', marker='o', label='preterm')
# plt.title('Scatter Plot Example')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()

# Show the plot
# plt.show()