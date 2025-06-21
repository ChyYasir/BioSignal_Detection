import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the signal from file
with open('E:/term-preterm-ehg-database-1.0.1/tpehgdb/tpehg1756.dat', 'r',  encoding='latin-1') as file:
    data = file.read()
signal = []
print(data)
for datum in data:
    datum = datum.strip()
    try:
        signal.append(float(datum))
    except ValueError:
        print(f"Could not convert '{datum}' to float.")

# print(len(signal))
# print(signal)

# with open('E:/term-preterm-ehg-database-1.0.1/tpehgdb/tpehg1756.dat', 'rb') as file:
#     # Read the content of the file
#     file_content = file.read()
#
# # Print or display the content as a hexadecimal string
# hex_content = file_content.hex()
# print(hex_content)
#
# # Optionally, you can decode the binary content into a string if it's text
# # For example, if it's UTF-8 encoded text
# try:
#     text_content = file_content.decode('utf-8')
#     print(text_content)
# except UnicodeDecodeError:
#     print("The content is not text or not in UTF-8 encoding.")