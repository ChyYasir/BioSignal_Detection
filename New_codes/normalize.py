import pandas as pd
from sklearn import preprocessing

# Load your CSV file
df = pd.read_csv('LCHO1.7.csv')


normalized_df = pd.DataFrame(preprocessing.normalize(df, axis=0), columns=df.columns)


normalized_df.to_csv('normalized_LCHO1.7.csv', index=False)

print("Normalized data saved to 'normalized_LCHO1.7.csv'.")