import os
import numpy as np
from main import SignalProcess

import matplotlib.pyplot as plt
class RunTermFiles:
    def __init__(self):
        self.n = 0
    def makeFeatureArray(self, n):
        features_2d_Array = []
        self.n = n
        for i in range(1, self.n+1, 1):
            if i == 6:
                continue
            formatted_number = str(i).zfill(3)
            signal_name = "tpehgt_t" + formatted_number
            signal = SignalProcess(signal_name)
            topologicalFeatures = signal.topological_features()
            features_2d_Array.append(topologicalFeatures)
        features_2d_Array = np.array(features_2d_Array)
        print(features_2d_Array)
        np.savetxt('term_features.txt', features_2d_Array, fmt='%f', delimiter='\t')





class RunPretermFiles:
    def __init__(self):
        self.n = 0
    def makeFeatureArray(self, n):
        features_2d_Array = []
        self.n = n
        for i in range(1, self.n+1, 1):
            if i == 6:
                continue
            formatted_number = str(i).zfill(3)
            signal_name = "tpehgt_p" + formatted_number
            signal = SignalProcess(signal_name)
            topologicalFeatures = signal.topological_features()
            features_2d_Array.append(topologicalFeatures)
        features_2d_Array = np.array(features_2d_Array)
        print(features_2d_Array)
        np.savetxt('preterm_features.txt', features_2d_Array, fmt='%f', delimiter='\t')


class RunOldFiles:
    def makeFeatureArray(self):
        term_features = np.loadtxt("term_features.txt", delimiter="\t")
        preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")
        term_features_list = term_features.tolist()
        preterm_features_list = preterm_features.tolist()
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
                    signal = SignalProcess(file_name_without_extension)
                    signal.process()
                    topologicalFeatures = signal.topological_features()
                    if float(gestation) >= 37:
                        term_features_list.append(topologicalFeatures)
                    else:
                        preterm_features_list.append(topologicalFeatures)
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    cnt = cnt + 1
                    continue


        print("Total number of Errors: ", cnt)
        # np.savetxt('term_features.txt', term_features_list, fmt='%f', delimiter='\t')
        # np.savetxt('preterm_features.txt', preterm_features_list, fmt='%f', delimiter='\t')





# run_all_term_files = RunTermFiles(13)
# run_all_term_files.makeFeatureArray()

# run_all_preterm_files = RunPretermFiles(13)
# run_all_preterm_files.makeFeatureArray()

run_old_files = RunOldFiles()
run_old_files.makeFeatureArray()

# areas_of_term = term_features[:, 4]

# areas_of_preterm = preterm_features[:, 4]
# x_axis = [i for i in range(12)]
#
# plt.scatter(x_axis, areas_of_term, c='blue', marker='o', label='term')
# plt.scatter(x_axis, areas_of_preterm, c='red', marker='o', label='preterm')
# plt.title('Scatter Plot Example')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
#
# # Show the plot
# plt.show()