import wfdb

# Set the path to the annotation file
file_path = 'D:/term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_t008'

# Read the annotation file
annotation = wfdb.rdann(file_path, extension='atr')

# Print the attributes of the annotation object
print('Record name:', annotation.record_name)
print('Sampling frequency:', annotation.fs)
# print('Annotation labels:', annotation.anntype)

annotation_sample = [ele/annotation.fs for ele in annotation.sample]
print('Annotation samples:', annotation_sample)
print('Auxiliary data:', annotation.aux_note)

# Display the annotation file
wfdb.plot_wfdb(annotation=annotation, time_units='seconds')