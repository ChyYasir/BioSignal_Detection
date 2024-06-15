import pandas as pd
from scipy.stats import mannwhitneyu
from imblearn.over_sampling import RandomOverSampler

column_headers = ["Area", "Perimeter", "Circularity", "Variance", "Bending Energy"]

# Load the data from CSV files with the specified headers
# x = pd.read_csv('later_spontaneous (NW,OW,UHO,HO).csv')
# y = pd.read_csv('later_cesarean (UHO,OW,NW,HO).csv')
df = pd.read_csv ('smote.csv')
x = df.drop(['activity'], axis = 1)
y = df ['activity']
#
# #data balancing
ros = RandomOverSampler(sampling_strategy='not majority')
x_res, y_res = ros.fit_resample(x, y)
ax = y_res.value_counts()
# print (x_res)
print(ax)


# Display the DataFrames
# print("Data 1:")
# print(data1)
# print("\nData 2:")
# print(data2)

# Take the first 8 rows of data2
# data1_subset = data1.head(110)
# data2_subset = data2.head(110)

activities = y_res.unique()


activity1 = activities[0]
activity2 = activities[1]

subset1 = x_res[y_res == activity1]
subset2 = x_res[y_res == activity2]

print(len(subset1))
print(len(subset2))

for column in column_headers:
    U1, p_value = mannwhitneyu(subset1[column], subset2[column])
    print(f"\nMann-Whitney U test for {column}:")
    print(f"U1: {U1}")
    print(f"P-value: {p_value}")