import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


early_cesarean_data = pd.read_csv("early_cesarean_features.csv", usecols=[0, 1, 2]).head(33).values
early_cesarean_labels = np.zeros((early_cesarean_data.shape[0], 1))

early_spontaneous_data = pd.read_csv("early_spontaneous_features.csv", usecols=[0, 1, 2]).head(33).values
early_spontaneous_labels = np.ones((early_spontaneous_data.shape[0], 1))

early_induced_data = pd.read_csv("early_induced-cesarean_features.csv", usecols=[0, 1, 2]).head(33).values
early_induced_labels = np.full((early_induced_data.shape[0], 1), 2)


data = np.vstack((early_cesarean_data, early_spontaneous_data, early_induced_data))
labels = np.vstack((early_cesarean_labels, early_spontaneous_labels, early_induced_labels)).flatten()

# Apply PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(data)


# Plot the data in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each class with different colors

# ax.scatter(data_3d[labels == 0, 0], data_3d[labels == 0, 1], data_3d[labels == 0, 2], c='blue', label='Early Cesarean')
# ax.scatter(data_3d[labels == 1, 0], data_3d[labels == 1, 1], data_3d[labels == 1, 2], c='red', label='Early Spontaneous')
# ax.scatter(data_3d[labels == 2, 0], data_3d[labels == 2, 1], data_3d[labels == 2, 2], c='green', label='Early Induced')


ax.scatter(data[labels == 0, 0], data[labels == 0, 1], data[labels == 0, 2], c='blue', label='Early Cesarean')
ax.scatter(data[labels == 1, 0], data[labels == 1, 1], data[labels == 1, 2], c='red', label='Early Spontaneous')
ax.scatter(data[labels == 2, 0], data[labels == 2, 1], data[labels == 2, 2], c='green', label='Early Induced')


ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('3D Plot of Early Cesarean, Spontaneous, and Induced Classes')


ax.legend()
plt.show()
