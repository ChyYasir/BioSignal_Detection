import os
import shutil

class RunOldFiles:
    def makeFolders(self, signal_type):
        directory_path = f"F:/signal/dataset/{signal_type}"
        cnt = 0

        for file_name in os.listdir(directory_path):
            if file_name.endswith(".hea"):
                file_path = os.path.join(directory_path, file_name)
                weight, height = self.extract_info_from_hea(file_path)
                weight_f = float(weight)
                height_f = float(height)
                height_f = height_f / 100
                BMI = weight_f / (height_f * height_f)

                print("Weight = ", weight_f, " Height = ", height_f, " BMI = ", BMI)
                file_name_without_extension = file_name.split(".")[0]

                try:


                    if BMI > 35:
                        self.create_copy(file_name_without_extension, signal_type, "unhealthy_obese")
                    elif BMI >= 30 and BMI < 35:
                        self.create_copy(file_name_without_extension, signal_type, "healthy_obese")
                    elif BMI >= 25 and BMI < 30:
                        self.create_copy(file_name_without_extension, signal_type, "over_weight")
                    elif BMI >= 20 and BMI < 25:
                        self.create_copy(file_name_without_extension, signal_type, "normal_weight")
                    else:
                        self.create_copy(file_name_without_extension, signal_type, "under_weight")

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    cnt += 1
                    print("Error Count: ", cnt)
                    continue

        print("Total number of Errors: ", cnt)

    @staticmethod
    def extract_info_from_hea(file_path):
        weight = None
        height = None


        with open(file_path, 'r') as file:
            for line in file:
                if "Weight" in line:
                    weight = line.strip().split()[-1]
                elif "Height" in line:
                    height = line.strip().split()[-1]

        # print(f"Gestation: {gestation}, Rectime: {rectime}")
        return weight, height

    @staticmethod
    def create_copy(file_name, signal_type, folder_name):

        source_path = f"F:/signal/dataset/{signal_type}"
        destination_path = f"F:/signal/dataset/{signal_type}/{folder_name}"

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        dat_file = f"{file_name}.dat"
        hea_file = f"{file_name}.hea"

        shutil.copy(os.path.join(source_path, dat_file), os.path.join(destination_path, dat_file))
        shutil.copy(os.path.join(source_path, hea_file), os.path.join(destination_path, hea_file))

        print(f"Copied {dat_file} and {hea_file} to {folder_name}")


class RunNewFiles:
    def __init__(self, directory):
        self.directory = directory

    def parse_hea_file(self, file_path):
        mode_of_delivery = None
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#Mode of delivery:'):
                    mode_of_delivery = line.split(':')[1].strip()
                    break
        return mode_of_delivery

    def copy_files(self, base_filename, mode_of_delivery):
        target_directory = os.path.join(self.directory, mode_of_delivery)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        for ext in ['.hea', '.dat', '.jpg']:
            src_file = os.path.join(self.directory, f'{base_filename}{ext}')
            if os.path.exists(src_file):
                shutil.copy(src_file, target_directory)

    def run(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.hea'):
                base_filename = filename[:-4]
                file_path = os.path.join(self.directory, filename)
                mode_of_delivery = self.parse_hea_file(file_path)
                if mode_of_delivery:
                    self.copy_files(base_filename, mode_of_delivery)
                else:
                    print(f"Mode of delivery not found for file: {filename}")



# make_early_later_spontaneous = RunOldFiles()
# make_early_later_spontaneous.makeFolders("early_cesarean")
# make_early_later_spontaneous.makeFolders("early_induced")
# make_early_later_spontaneous.makeFolders("early_induced-cesarean")
# make_early_later_spontaneous.makeFolders("later_cesarean")
# make_early_later_spontaneous.makeFolders("later_induced")
# make_early_later_spontaneous.makeFolders("later_induced-cesarean")

directory = "F:/signal/files"
runner = RunNewFiles(directory)
runner.run()
