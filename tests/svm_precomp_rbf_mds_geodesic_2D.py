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
gamma = 25
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

print("="*50)  

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

###############################################################################
# Testing SVM with rbf Kernel 

from sklearn.model_selection import cross_val_score, KFold # RepeatedStratifiedKFold

# Split data into training and testing sets
data_train, data_test, labels_train, labels_test = train_test_split(data_mds_geodesic, labels, test_size=0.2, random_state=42)

# SVM with rbf kernel
svm_rbf = SVC(kernel='rbf', gamma=0.001, C=1, random_state=47)
svm_rbf.fit(data_train, labels_train)

# Predict labels for the testing set
predicted_labels = svm_rbf.predict(data_test)

# Calculate accuracy and print classification report for the initial model
accuracy = accuracy_score(labels_test, predicted_labels)
classification_rep = classification_report(labels_test, predicted_labels)

print("Accuracy with SVM - rbf Kernel only, without Cross-validation:", accuracy)
print("Classification Report with SVM - rbf only, without Cross-validation:\n", classification_rep)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(labels_test, predicted_labels, labels=['AD', 'CN'])

print("Confusion Matrix with SVM - rbf only:")
print(conf_matrix)

print("="*50)  

# Cross-validation 
# Create k-fold cross-validation iterators
kfold = KFold(n_splits=5, shuffle=True, random_state=47) # n_splits=5 gives best result; KFold Accuracy: 0.91 (+/- 0.06)
 
# Perform cross-validation and get accuracy scores for KFold
cross_val_scores_kfold = cross_val_score(svm_rbf, data_train, labels_train, cv=kfold, scoring='accuracy')
mean_accuracy_kfold = cross_val_scores_kfold.mean()

# Print average accuracy and standard deviation for KFold
print("Cross-validation results for SVM - rbf Kernel (KFold):")
print("KFold Accuracy: {:.2f} (+/- {:.2f})".format(mean_accuracy_kfold, cross_val_scores_kfold.std() * 2))

print("="*50)  

###############################################################################

# Hyperparameter Tuning
# Random range for hyperparameters
# param_grid_svm_rbf = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# Fine tuning hyperparameters
param_grid_svm_rbf = {'C': [0.01, 0.5, 1, 1.5, 2], 'gamma': [ 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]}
cv = kfold
grid_search_rbf = GridSearchCV(svm_rbf, param_grid_svm_rbf, cv=cv, scoring='accuracy')
# Perform the grid search
grid_search_rbf.fit(data_mds_geodesic, labels)

# Get the best hyperparameters and accuracy
best_params_rbf = grid_search_rbf.best_params_
best_accuracy_rbf = grid_search_rbf.best_score_

print(f"Best Hyperparameters for SVM - rbf Kernel only, with KFold cross-validation: {best_params_rbf}")
print(f"Best Accuracy for SVM - rbf Kernel only, with KFold cross-validation: {best_accuracy_rbf:.2f}")


###############################################################################
# Prediction using best model with best set of hyperparameters found during grid search 

predicted_labels = grid_search_rbf.best_estimator_.predict(data_test) 

# Calculate accuracy and print classification report for the initial model
accuracy = accuracy_score(labels_test, predicted_labels)
classification_rep = classification_report(labels_test, predicted_labels)

print("Accuracy with SVM - rbf Kernel, hyperparameter fine-tuned:", accuracy)
print("Classification Report with SVM - rbf kernel, hyperparameter fine-tuned:\n", classification_rep)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(labels_test, predicted_labels, labels=['AD', 'CN'])

print("Confusion Matrix with SVM - rbf, hyperparameter fine-tuned:")
print(conf_matrix)

print("="*50)  