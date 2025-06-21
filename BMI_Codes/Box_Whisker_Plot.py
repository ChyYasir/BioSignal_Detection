import pandas as pd
import matplotlib.pyplot as plt

# Names of your CSV files
files = [
    'later_cesarean_normal_weight.csv',
    'later_cesarean_over_weight.csv',
    'later_induced_normal_weight.csv',
    'later_spontaneous_normal_weight.csv'
]

feature = "Max Power/Frequency"

# Read the specified column from each CSV file and compute statistics
data = []
stats = []  # To store stats for display
for file in files:
    # print(file)
    df = pd.read_csv(file)
    data.append(df[feature])
    # print(df[feature])
    # Compute descriptive statistics
    # print(df[feature].describe())
    stats.append(df[feature].describe())

# print(stats)
# Box plot
plt.figure(figsize=(10, 6))
box = plt.boxplot(data, patch_artist=True, vert=True, showmeans=True, meanline=True)

colors = ['blue', 'green', 'red', 'purple']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Set x-tick labels
group_labels = ['Early Cesarean', 'Early Induced-Cesarean', 'Early Induced', 'Early Spontaneous']
plt.xticks([1, 2, 3, 4], group_labels)

plt.ylabel('Feature Value')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Display range values (min, 25%, 50%, 75%, max) for each group
for i, stat in enumerate(stats, start=1):
    # print(stat)
    # print(stat[0])
    text = ""
    print(stat["min"], stat["max"])
    text = f"Min: {stat['min']:.2f}\n25%: {stat['25%']:.2f}\nMedian: {stat['50%']:.2f}\n75%: {stat['75%']:.2f}\nMax: {stat['max']:.2f}"
    plt.text(i, stat['max'], text, ha='center', va='bottom')

plt.show()
