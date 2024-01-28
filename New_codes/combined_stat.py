import pandas as pd

def calculate_statistics(file_name, row_label):

    df = pd.read_csv(file_name, usecols=[0, 1, 2, 4, 5])


    df.columns = ['Area', 'Perimeter', 'Circularity', 'Variance', 'Bending Energy']

    stats_max = pd.Series([df[col].max() for col in df.columns], index=df.columns, name=row_label + " (Max)")
    stats_min = pd.Series([df[col].min() for col in df.columns], index=df.columns, name=row_label + " (Min)")
    stats_avg = pd.Series([df[col].mean() for col in df.columns], index=df.columns, name=row_label + " (Average)")

    return pd.concat([stats_max, stats_min, stats_avg], axis=1).transpose()


cesarean_stats = calculate_statistics('early_cesarean_features.csv', 'Early Cesarean')
spontaneous_stats = calculate_statistics('early_spontaneous_features.csv', 'Early Spontaneous')


combined_stats = pd.concat([cesarean_stats, spontaneous_stats])


combined_stats.to_csv('combined_statistics.csv')