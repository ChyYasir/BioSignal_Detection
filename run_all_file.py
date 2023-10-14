import numpy as np
from main import SignalProcess
import matplotlib.pyplot as plt
class RunTermFiles:
    def __init__(self, n):
        self.n = n
    def makeFeatureArray(self):
        features_2d_Array = []
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
    def __init__(self, n):
        self.n = n
    def makeFeatureArray(self):
        features_2d_Array = []
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


# run_all_term_files = RunTermFiles(13)
# run_all_term_files.makeFeatureArray()

# run_all_preterm_files = RunPretermFiles(13)
# run_all_preterm_files.makeFeatureArray()

term_features = np.loadtxt("term_features.txt", delimiter="\t")
areas_of_term = term_features[:, 4]
preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")
areas_of_preterm = preterm_features[:, 4]
x_axis = [i for i in range(12)]

plt.scatter(x_axis, areas_of_term, c='blue', marker='o', label='term')
plt.scatter(x_axis, areas_of_preterm, c='red', marker='o', label='preterm')
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()