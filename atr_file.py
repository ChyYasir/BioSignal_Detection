import wfdb
import matplotlib.pyplot as plt

# Set the path to the annotation file
file_path = 'F:/signal/files/ice002_p_1of3'

# Read the annotation file
try:
    annotation = wfdb.rdann(file_path, extension='atr')
except FileNotFoundError:
    print(f"Error: Annotation file not found at {file_path}.atr")
    exit(1)

# Define the annotation type mappings based on the provided .atr file structure
annotation_types = {
    'C': 'Contraction',
    '(c)': 'Possible contraction',
    'pm': 'Participant movement',
    'pos': 'Participant change of position',
    'fm': 'Fetal movement',
    'em': 'Equipment manipulation'
}

# Print basic annotation file information
print('Record Name:', annotation.record_name)
print('Sampling Frequency (Hz):', annotation.fs)
print('Number of Annotations:', annotation.ann_len)

# Extract and display annotation details
print('\nAnnotation Details:')
print(f"{'Sample Index':<15} {'Time (s)':<12} {'Event Type':<20} {'Description'}")
print('-' * 65)

for i in range(annotation.ann_len):
    sample_index = annotation.sample[i]
    time_seconds = sample_index / annotation.fs
    # Get the annotation symbol and map it to the description
    ann_symbol = annotation.aux_note[i].strip() if annotation.aux_note[i] else annotation.symbol[i]
    description = annotation_types.get(ann_symbol, 'Unknown event')
    print(f"{sample_index:<15} {time_seconds:<12.3f} {ann_symbol:<20} {description}")

# Optional: Plot the annotation events over time (if a signal is available)
# Note: Since we only have the .atr file, we'll plot the event occurrences
plt.figure(figsize=(12, 4))
plt.eventplot(annotation.sample / annotation.fs, linelengths=0.5, colors='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Events')
plt.title(f'Event Annotations for Record {annotation.record_name}')
plt.grid(True)
plt.show()

# If the corresponding signal (.dat) file exists, you can plot it with annotations
try:
    record = wfdb.rdrecord(file_path)
    wfdb.plot_wfdb(record=record, annotation=annotation, time_units='seconds', title=f'Signal and Annotations for {annotation.record_name}')
except FileNotFoundError:
    print("Signal (.dat) file not found. Only annotation plot is shown.")