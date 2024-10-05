""" Optimized for roc_auc with Optuna"""

import os
import optuna
import json
from joblib import dump
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import load_from_hdf5 # imported from utils.py
from optuna.pruners import HyperbandPruner  

np.random.seed(47) #47 original

def load_data(file_path, dataset_type='train'):
    # Choose dataset based on type
    dataset_name = f'projection_{dataset_type}'
    
    # Load the combined features and labels from the HDF5 file
    projection_data = load_from_hdf5(file_path, dataset_name)
    
    # Split the projection_data into features and labels
    features = projection_data[:, :-1]  # All rows, all but the last column
    labels = projection_data[:, -1]  # All rows, only the last column
            
    return features, labels

def svm_with_known_parameters(features, labels, best_params):
    model = SVC(
        C=best_params['C'], 
        gamma=best_params['gamma'], 
        kernel=best_params['kernel'], 
        probability=True, 
        class_weight='balanced'
    )

    model.fit(features, labels)
          
    return model 

def process_files(train_directory_path, test_directory_path, results_directory_path, file_prefix='projection_', indices_to_process=None):
    for filename in os.listdir(train_directory_path):
        if filename.startswith(file_prefix + 'train_') and filename.endswith('.h5'):
            file_index = int(filename.split('_')[-1].split('.')[0])
            
            if indices_to_process is None or file_index in indices_to_process:
                train_file_path = os.path.join(train_directory_path, filename)
                test_file_path = os.path.join(test_directory_path, f'projection_test_{file_index}.h5')

                # Load training data
                features, labels = load_data(train_file_path, 'train')

                # Train best model
                best_params = {
                    "kernel": "rbf",
                    "C": 475.6895816544916,
                    "gamma": 1.3239645217802024e-05
                    }
                
                (
                    best_model 
                    # train_accuracy,
                    # val_accuracy,
                    # mean_roc_auc
                ) = svm_with_known_parameters(features, labels, best_params)

                # print(f"Training File Index: {file_index} - Best Parameters: {best_params} - Validation Accuracy: {val_accuracy:.2f}")
                print(f"Training File Index: {file_index} - Best Parameters: {best_params}")

                # Load test data and evaluate if exists
                if os.path.exists(test_file_path):
                    features_test, labels_test = load_data(test_file_path, 'test')
                    test_predictions = best_model.predict(features_test)
                    test_accuracy = accuracy_score(labels_test, test_predictions)
                    print(f"Test File Index: {file_index} - Test Accuracy: {test_accuracy:.2f}")
                
                # Prepare summary data for saving
                summary_data = {
                    'file_index': file_index,
                    'best_parameters': best_params,
                    # 'training_accuracy': train_accuracy,
                    # 'validation_accuracy': val_accuracy,
                    # 'mean_roc_auc': mean_roc_auc,
                    'test_accuracy': test_accuracy
                    }
                
                # Define the base path for saving files         
                # base_path = f'/home/sgurau/Desktop/BrainProjectionNet/Projections/SVM_model_{file_index}' 
                save_path = os.path.join(results_directory_path, f'SVM_model_{file_index}')
                
                # Check if the directory exists
                if not os.path.exists(save_path):
                    os.makedirs(save_path) # If it doesn't exist, create it

                    
                model_save_path = os.path.join(save_path, f'best_model_{file_index}.joblib')     
                
                # features_train_path = os.path.join(save_path, f'features_train_{file_index}.joblib')
                # labels_train_path = os.path.join(save_path, f'labels_train_{file_index}.joblib')
                
                # features_val_path = os.path.join(save_path, f'features_val_{file_index}.joblib')
                # labels_val_path = os.path.join(save_path, f'labels_val_{file_index}.joblib')
                
                features_test_path = os.path.join(save_path, f'features_test_{file_index}.joblib')
                labels_test_path = os.path.join(save_path, f'labels_test_{file_index}.joblib')
                
                summary_save_path = os.path.join(save_path, f'summary_{file_index}.json')
                
                dump(best_model, model_save_path)
                
                # dump(X_train, features_train_path)
                # dump(y_train, labels_train_path)
                
                # dump(X_val, features_val_path)
                # dump(y_val, labels_val_path)
                
                dump(features_test, features_test_path)
                dump(labels_test, labels_test_path)
                
                # Save the summary data as JSON
                with open(summary_save_path, 'w') as f:
                   json.dump(summary_data, f, indent=4)
                   
                print(f"Summary for file index {file_index} saved to {summary_save_path}")


if __name__ == "__main__":
    # Dynamically get home directory of current user
    home_dir = os.path.expanduser('~')
    base_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections')
    
    # Define directory where files are located
    train_directory_path = os.path.join(base_path, 'Train')
    test_directory_path = os.path.join(base_path, 'Test')
    results_directory_path =  os.path.join(base_path, 'Results')
    
    # Define indices of the projection files to process
    """ Speecify a list or a range of indices to process."""
    """ Specify indices as a list: indices_to_process = [4] or [4, 5, 6]"""
    """ Or specify a range: indices_to_process = range(4, 7)"""
    # indices_to_process = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  
    indices_to_process = range(0, 11) # range(15, 22)  # Specify as needed
           
    # Call the function to process files
    """ Setting 'None' for indices_to_process, or by excluding this in function call which is None by default, will run all projections with specified file pattern."""
    process_files(train_directory_path, test_directory_path, results_directory_path, indices_to_process=indices_to_process)
