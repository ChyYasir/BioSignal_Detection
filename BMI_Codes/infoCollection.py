import os
import re
import numpy as np


def convert_gestational_age(gestation):

    # print(gestation)
    weeks, days = map(int, gestation.split('/'))
    # print(weeks + days/7.0)
    return weeks + days / 7.0


def parse_hea_file(file_path, file_type):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = {}
    if file_type == '6_channel':
        for line in lines:
            if line.startswith('#'):
                if 'Age' in line:
                    # print(line.split())

                    metadata['Age'] = int(line.split()[2])
                elif 'Height' in line:
                    metadata['Height'] = int(line.split()[2])
                elif 'Weight' in line:
                    metadata['Weight'] = int(line.split()[2])
                elif 'Gestation' in line:
                    match = re.search(r'Gestation (\d+\.\d+)', line)
                    if match:
                        metadata['GestationalAge'] = float(match.group(1))
    elif file_type == '16_channel':
        for line in lines:
            if line.startswith('#'):
                if 'Age(years)' in line:
                    metadata['Age'] = int(line.split(':')[1].strip())
                elif 'Height' in line:
                    match = re.search(r'Height:(\d+)', line)
                    if match:
                        metadata['Height'] = int(match.group(1))
                elif 'Weight' in line:
                    match = re.search(r'Weight:(\d+)', line)
                    if match:
                        metadata['Weight'] = int(match.group(1))
                elif 'BMI at recording' in line:
                    match = re.search(r'BMI at recording:(\d+\.\d+)', line)
                    if match:
                        metadata['BMI'] = float(match.group(1))
                elif 'Gestational age at recording' in line:
                    match = re.search(r'Gestational age at recording\(w/d\):(\d+/\d+)', line)
                    if match:
                        metadata['GestationalAge'] = convert_gestational_age(match.group(1))

    return metadata


def calculate_bmi(height, weight):

    height_m = height / 100
    return weight / (height_m ** 2)


def process_directory(directory):
    all_metadata = []
    for filename in os.listdir(directory):
        if filename.endswith('.hea'):
            file_path = os.path.join(directory, filename)
            if 'icehg' in filename:
                file_type = '6_channel'
            else:
                file_type = '16_channel'

            metadata = parse_hea_file(file_path, file_type)

            if 'BMI' not in metadata and 'Height' in metadata and 'Weight' in metadata:
                metadata['BMI'] = calculate_bmi(metadata['Height'], metadata['Weight'])

            all_metadata.append(metadata)

    # Collect values for each field
    ages = [md['Age'] for md in all_metadata if 'Age' in md]
    bmis = [md['BMI'] for md in all_metadata if 'BMI' in md]
    gestational_ages = [md['GestationalAge'] for md in all_metadata if 'GestationalAge' in md]

    # Calculate statistics
    stats = {
        "Patient's Age": {'max': np.max(ages), 'min': np.min(ages), 'avg': np.mean(ages), 'std': np.std(ages)},
        'BMI': {'max': np.max(bmis), 'min': np.min(bmis), 'avg': np.mean(bmis), 'std': np.std(bmis)},
        'GestationalAge': {'max': np.max(gestational_ages), 'min': np.min(gestational_ages),
                           'avg': np.mean(gestational_ages), 'std': np.std(gestational_ages)}
    }

    return stats



directory = 'F:/signal/dataset/later_cesarean/healthy_obese'


stats = process_directory(directory)
for field, stat in stats.items():
    print(f"{field} - Max: {stat['max']}, Min: {stat['min']}, Avg: {stat['avg']}, Std: {stat['std']}")
