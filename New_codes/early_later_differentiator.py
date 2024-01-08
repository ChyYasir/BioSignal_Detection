import os
import shutil

class RunOldFiles:
    def makeFolders(self):
        directory_path = "F:/signal/term-preterm-ehg-database-1.0.1/tpehgdb"
        cnt = 0

        for file_name in os.listdir(directory_path):
            if file_name.endswith(".hea"):
                file_path = os.path.join(directory_path, file_name)
                gestation, rectime = self.extract_info_from_hea(file_path)

                file_name_without_extension = file_name.split(".")[0]

                try:
                    # signal = SignalProcess(file_name_without_extension, directory_path)
                    # signal.process()
                    # allNewFeatures = signal.all_segment_new_features()
                    # take only term files
                    if float(gestation) >= 37:
                        # Check Rectime conditions
                        if float(rectime) <= 25:
                            self.create_copy(file_name_without_extension, "early_spontaneous")
                        elif float(rectime) >= 30:
                            self.create_copy(file_name_without_extension, "later_spontaneous")
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    cnt += 1
                    print("Error Count: ", cnt)
                    continue

        print("Total number of Errors: ", cnt)

    @staticmethod
    def extract_info_from_hea(file_path):
        gestation = None
        rectime = None

        with open(file_path, 'r') as file:
            for line in file:
                if "Gestation" in line:
                    gestation = line.strip().split()[-1]
                elif "Rectime" in line:
                    rectime = line.strip().split()[-1]

        print(f"Gestation: {gestation}, Rectime: {rectime}")
        return gestation, rectime

    @staticmethod
    def create_copy(file_name, folder_name):

        source_path = "F:/signal/term-preterm-ehg-database-1.0.1/tpehgdb"
        destination_path = f"F:/signal/dataset/{folder_name}/"

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        dat_file = f"{file_name}.dat"
        hea_file = f"{file_name}.hea"

        shutil.copy(os.path.join(source_path, dat_file), os.path.join(destination_path, dat_file))
        shutil.copy(os.path.join(source_path, hea_file), os.path.join(destination_path, hea_file))

        print(f"Copied {dat_file} and {hea_file} to {folder_name}")


make_early_later_spontaneous = RunOldFiles()
make_early_later_spontaneous.makeFolders()