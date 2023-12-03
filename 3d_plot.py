import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

term_data = pd.read_csv("term_features.csv", usecols=[0, 4, 5]).values



term_labels = np.ones((term_data.shape[0], 1))  # Label for term features is 1

preterm_data = pd.read_csv("preterm_features.csv", usecols=[0, 4, 5]).values
preterm_labels = np.zeros((preterm_data.shape[0], 1))  # Label for preterm features is 0


data = np.vstack((term_data, preterm_data))
labels = np.vstack((term_labels, preterm_labels)).flatten()

data = np.vstack((term_data, preterm_data))
labels = np.vstack((term_labels, preterm_labels)).flatten()

# Apply PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(data)

# Plot the data in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(data_3d[labels == 1, 0], data_3d[labels == 1, 1], data_3d[labels == 1, 2], c='blue', label='Term')


ax.scatter(data_3d[labels == 0, 0], data_3d[labels == 0, 1], data_3d[labels == 0, 2], c='red', label='Preterm')

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('3D Plot of Term and Preterm Data')

plt.legend()
# plt.show()

# umap = UMAP(n_components=3, random_state=42)
# data_3d_umap = umap.fit_transform(data)
#
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot term data points
# ax.scatter(data_3d_umap[labels == 1, 0], data_3d_umap[labels == 1, 1], data_3d_umap[labels == 1, 2], c='blue', label='Term')
#
# # Plot preterm data points (you can customize the color)
# ax.scatter(data_3d_umap[labels == 0, 0], data_3d_umap[labels == 0, 1], data_3d_umap[labels == 0, 2], c='red', label='Preterm')
#
# # Customize plot appearance
# ax.set_xlabel('UMAP Component 1')
# ax.set_ylabel('UMAP Component 2')
# ax.set_zlabel('UMAP Component 3')
# ax.set_title('UMAP Visualization of Data')

# Add legend
ax.legend()

# Show the plot
plt.show()