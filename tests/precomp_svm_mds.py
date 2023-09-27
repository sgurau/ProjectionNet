import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths to data
folder_paths = [
    '/home/sgurau/Desktop/output/functional_connectivity/AD/sub-*',
    '/home/sgurau/Desktop/output/functional_connectivity/CN/sub-*'
]

# Create labels for data
labels = []
data_matrices = []  # List to store individual correlation matrices
for folder_path in folder_paths:
    if 'AD' in folder_path:
        label = 'AD'
    elif 'CN' in folder_path:
        label = 'CN'
    
    data_files = glob.glob(os.path.join(folder_path, 'sub-*_connectivity_matrix.txt'))
    for data_file in data_files:
        # Load correlation matrix from data_file
        correlation_matrix = np.loadtxt(data_file)  # Or use np.genfromtxt for CSV

        data_matrices.append(correlation_matrix)
        labels.append(label)
        
# Load geodesic distances file
geodesic_distances_file = '/home/sgurau/Desktop/geodesic_distances.csv'

# Load the geodesic distance matrix from the CSV file
geodesic_distances = pd.read_csv(geodesic_distances_file, header=None).values

# Compute the precomputed kernel matrix
gamma = 20 
precomputed_kernel_matrix = np.exp(-(geodesic_distances-95) / gamma) 

# Convert the data_matrices list to a NumPy array
data_matrices = np.array(data_matrices)

# Create train-test split based on labels (maintaining class balance)
data_train_indices, data_test_indices = train_test_split(np.arange(len(data_matrices)), test_size=0.2, stratify=labels, random_state=47) 

# Define the SVM classifier with a precomputed kernel
classifier = SVC(kernel='precomputed')

# Fit the model using the kernel matrix 
classifier.fit(precomputed_kernel_matrix[data_train_indices][:, data_train_indices], np.array(labels)[data_train_indices])

# Predict labels for the testing set
predicted_labels = classifier.predict(precomputed_kernel_matrix[data_test_indices][:, data_train_indices])

# Calculate accuracy and print classification report
accuracy = accuracy_score(np.array(labels)[data_test_indices], predicted_labels)
classification_rep = classification_report(np.array(labels)[data_test_indices], predicted_labels)

print("Accuracy with SVM Classifier after Hyperparameter Tuning:", accuracy)
print("SVM Classification Report after Hyperparameter Tuning:\n", classification_rep)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(np.array(labels)[data_test_indices], predicted_labels, labels=['AD', 'CN'])

print("Confusion Matrix:")
print(conf_matrix)

###############################################################################

# Set a fixed random seed for reproducibility
np.random.seed(47) 

## MDS with geodesic distances matrix
from sklearn.manifold import MDS

embedding = MDS(n_components=2)
data_mds_geodesic = embedding.fit_transform(geodesic_distances)
data_mds_geodesic.shape

import matplotlib.pyplot as plt
# Define color mapping
color_mapping = {'AD': 'red', 'CN': 'blue'}
colors = [color_mapping[label] for label in labels]

# Plot MDS results
plt.figure(figsize=(8, 6))
plt.scatter(data_mds_geodesic[:,0], data_mds_geodesic[:,1], c=colors, alpha=1, s=20)  # Plot on one axis since LDA has only one dimension
plt.title('MDS - Geodesic Visualization')
plt.xlabel
plt.ylabel('')  # No need to label the y-axis since it's one-dimensional

# Create custom legend handles and labels
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], markersize=10) for label in color_mapping]
legend_labels = list(color_mapping.keys())

plt.legend(legend_handles, legend_labels, title='Labels')
plt.show()

