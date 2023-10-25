import glob
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import LabelEncoder

np.random.seed(43)

# Define paths to data
folder_paths = [
    '/home/sgurau/Desktop/output/functional_connectivity/AD/sub-*',
    '/home/sgurau/Desktop/output/functional_connectivity/CN/sub-*'
]

# Create labels for data
labels_string = []
data_matrices = []  # List to store individual correlation matrices
for folder_path in folder_paths:
    if 'AD' in folder_path:
        label = 'AD'
    elif 'CN' in folder_path:
        label = 'CN'
    
    data_files = glob.glob(os.path.join(folder_path, 'sub-*_connectivity_matrix.txt'))
    for data_file in data_files:
        # Load correlation matrix from data_file
        correlation_matrix = np.loadtxt(data_file)

        data_matrices.append(correlation_matrix)
        labels_string.append(label)
        
encoder = LabelEncoder()
labels = encoder.fit_transform(labels_string)        

# Load geodesic distances file
geodesic_distances_file = '/home/sgurau/Desktop/geodesic_distances.csv'

# Load the geodesic distance matrix from the CSV file
geodesic_distances = pd.read_csv(geodesic_distances_file, header=None).values

# Define the values of gamma and C to loop over
gamma_values = [3, 30, 300, 3000, 30000]
C_values = [0.1, 1, 10]

# Define the number of folds for cross-validation
kfold = 5

# Create a list to store fold results
fold_results = []

# Calculate the number of samples per fold
samples_per_fold = len(geodesic_distances) // kfold

# Shuffle data
all_indices = list(range(len(geodesic_distances)))
np.random.shuffle(all_indices)

accuracy_values = []  # Store accuracy values for this fold
    
# Loop over different folds
for fold in range(1, kfold+1):
    start_idx = (fold - 1) * samples_per_fold
    end_idx = fold * samples_per_fold

    train_indices = all_indices[:start_idx] + all_indices[end_idx:]
    test_indices = all_indices[start_idx:end_idx]
     
    for gamma in gamma_values:
        # Compute the precomputed kernel matrix for each gamma
        precomputed_kernel_matrix = np.exp(-(geodesic_distances)**2 / gamma)
        
        # Regularize the kernel matrix
        # precomputed_kernel_matrix += 1e-4 * np.eye(len(geodesic_distances))
        
        for C in C_values:
            # Define the SVM classifier with a precomputed kernel
            classifier = SVC(kernel='precomputed', C=C) # SVC(kernel='precomputed', C=C, class_weight='balanced') # class_weight={0:2, 1:1}

            # Fit the model using the kernel matrix 
            classifier.fit(precomputed_kernel_matrix[train_indices][:, train_indices], np.array(labels)[train_indices])

            # Predict labels for the testing set
            predicted_labels = classifier.predict(precomputed_kernel_matrix[test_indices][:, train_indices])

            # Calculate accuracy for this combination of gamma and C
            accuracy = accuracy_score(np.array(labels)[test_indices], predicted_labels)
            accuracy_values.append(accuracy)
       
            print(f"Fold={fold}, Gamma={gamma}, C={C}, Accuracy: {accuracy:.2f}")
    
accuracy_values = np.array(accuracy_values).reshape(5, 15)

# Calculate and store the average accuracy and standard deviation for this fold
average_accuracy = np.mean(accuracy_values, axis=0)
std_dev = np.std(accuracy_values, axis=0)

fold_results.append({
    "Fold": fold,
    "Average Accuracy": average_accuracy,
    "Std Dev": std_dev
})
  
# Print the fold results after the loop
#for result in fold_results:
  #  print(f"Fold={result['Fold']}, Average Accuracy: {result['Average Accuracy']:.2f}, Std Dev: {result['Std Dev']:.2f}")



