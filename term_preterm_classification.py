import os

# Define the directory path
directory_path = "E:/term-preterm-ehg-database-1.0.1/tpehgdb/"

# List all files in the directory
files = os.listdir(directory_path)

# Initialize a dictionary to store Gestation values by file name
gestation_values = {}

# Iterate through each file in the directory
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

        # Store the Gestation value in the dictionary
        gestation_values[file_name] = gestation
        # Split the file name by the dot (.)
        file_parts = file_name.split(".")

        # Get the part before ".hea"
        file_name_without_extension = file_parts[0]


        print(file_name_without_extension + " : " + gestation)

# Print the results
for file_name, gestation in gestation_values.items():
    if gestation is not None:
        print(f"File: {file_name} - Gestation: {gestation}")
    else:
        print(f"File: {file_name} - Gestation value not found in the file.")
