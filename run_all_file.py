import numpy as np
from main import SignalProcess
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

run_all_preterm_files = RunPretermFiles(13)
run_all_preterm_files.makeFeatureArray()