""" Optimized for roc_auc with Optuna."""

import os
import json
import optuna
from joblib import dump
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score 
from utils import load_from_hdf5 # load_data_and_labels       # imported from utils.py
from optuna.pruners import HyperbandPruner 
# from itertools import product

np.random.seed(43)

def load_data(file_path, dataset_type='train'):
    # Choose dataset based on type
    dataset_name = f'projection_{dataset_type}'
    
    # Load the combined features and labels from the HDF5 file
    projection_data = load_from_hdf5(file_path, dataset_name)
    
    # Split the projection_data into features and labels
    features = projection_data[:, :-1]  # All rows, all but the last column
    labels = projection_data[:, -1]  # All rows, only the last column
            
    return features, labels

def mlp_cross_validation_and_hyperparameter_tuning(features, labels):
    def objective(trial):
        
        # Dynamically define the number of layers and neurons per layer
        n_layers = trial.suggest_int('n_layers', 1, 5)  # Number of layers
        hidden_layer_sizes = []
        for i in range(n_layers):
            n_neurons = trial.suggest_int(f'n_neurons_layer_{i}', 1, 50)
            hidden_layer_sizes.append(n_neurons)
        hidden_layer_sizes = tuple(hidden_layer_sizes)

        # Define the hyperparameters to be tuned
        activation = trial.suggest_categorical('activation', ['tanh', 'relu']) # Supported activations are ['identity', 'logistic', 'relu', 'softmax', 'tanh']
        solver = trial.suggest_categorical('solver', ['sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e3, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])

        # Create and fit the MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                            activation=activation, 
                            solver=solver, 
                            alpha=alpha, 
                            learning_rate=learning_rate, 
                            max_iter=3000) 
        
        # Initialize a dictionary to hold roc_auc and validation accuracy values
        roc_aucs = []
        accuracies = []
        
        # Initialize a StratifiedKFold object
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(features, labels):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds_proba = model.predict_proba(X_val)[:, 1]  # Get probability of positive class

            # Evaluate the model on this fold
            roc_auc = roc_auc_score(y_val, preds_proba)
            roc_aucs.append(roc_auc)
            
            accuracy = accuracy_score(y_val, preds)
            accuracies.append(accuracy)
            
            
        val_accuracy = np.mean(accuracies)
        trial.set_user_attr("val_accuracy", val_accuracy)
        
        mean_roc_auc = np.mean(roc_aucs)
        trial.set_user_attr("mean_roc_auc", mean_roc_auc)
        
        # Return average ROC AUC score across all folds
        return mean_roc_auc
        
    # Create a study object and optimize the objective function
    pruner = HyperbandPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=350)  
    
    # Extract best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Flip the sign to get the accuracy
    
    # Retrieve the average validation accuracy and roc_auc
    val_accuracy = study.best_trial.user_attrs["val_accuracy"]   
    mean_roc_auc = study.best_trial.user_attrs["mean_roc_auc"]
    
    # Remove 'n_layers' and 'n_neurons_layer_{i}' entries from `best_params`, 
    """ The dictionary of  parameters 'best_params' returned by Optuna for MLPClassifier does not recognize 'n_layers' 
    and 'n_neurons_layer_{i}' as parameters. 'n_layers' and 'n_neurons_layer_{i}' were defined witin optuna framework 
    to dynamically explore the number of layers and number of neurons in each layer refor MLP model during 
    hyperparameter optimization process.
    """
    
    model_architecture = {
    'n_layers': best_params['n_layers'],
    'layers': {f'layer_{i}': best_params[f'n_neurons_layer_{i}'] for i in range(best_params['n_layers'])}
    }
    
    if 'n_layers' in best_params:
        del best_params['n_layers']  
        
    # Remove 'n_neurons_layer_{i}' entries
    for key in list(best_params.keys()):
        if key.startswith('n_neurons_layer_'):
            del best_params[key]
            
    # Train and evaluate classifier with best parameters found
    best_model = MLPClassifier(**best_params, max_iter=3000).fit(features, labels)
    
    # Training accuracy
    predictions_train = best_model.predict(features)
    train_accuracy = accuracy_score(labels, predictions_train)
    
    return best_model, best_params, train_accuracy, val_accuracy, mean_roc_auc, model_architecture

def process_files(train_directory_path, test_directory_path, results_directory_path, file_prefix='projection_', indices_to_process=None):
    for filename in os.listdir(train_directory_path):
        if filename.startswith(file_prefix + 'train_') and filename.endswith('.h5'):
            file_index = int(filename.split('_')[-1].split('.')[0])
            
            if indices_to_process is None or file_index in indices_to_process:
                train_file_path = os.path.join(train_directory_path, filename)
                test_file_path = os.path.join(test_directory_path, f'projection_test_{file_index}.h5')

                # Load training data
                features, labels = load_data(train_file_path, 'train')
                
                (
                    best_model, 
                    best_params, 
                    train_accuracy,
                    val_accuracy,
                    mean_roc_auc,
                    model_architecture
                    ) = mlp_cross_validation_and_hyperparameter_tuning(features, labels)
                
                # print(f"Training File Index: {file_index} - Best Parameters: {best_params} - Validation Accuracy: {val_accuracy:.2f}")
                print(f"Training File Index: {file_index} - Best Parameters: {best_params}")

                # Load test data and evaluate if exists
                if os.path.exists(test_file_path):
                    features_test, labels_test = load_data(test_file_path, 'test')
                    test_predictions = best_model.predict(features_test)
                    test_accuracy = accuracy_score(labels_test, test_predictions)
                    print(f"Test File Index: {file_index} - Test Accuracy: {test_accuracy:.2f}")
                    
                summary_data = {
                    'file_index': file_index,
                    'best_parameters': best_params,
                    'model_architecture': model_architecture,
                    'training_accuracy': train_accuracy,
                    'validation_accuracy': val_accuracy,
                    'mean_roc_auc': mean_roc_auc,
                    'test_accuracy': test_accuracy
                    }
                
                # Define the base path for saving files  
                save_path = os.path.join(results_directory_path, f'sklearnMLP_model_{file_index}')
                
                # Check if the directory exists
                if not os.path.exists(save_path):
                    os.makedirs(save_path) # If it doesn't exist, create it
                    
                model_save_path = os.path.join(save_path, f'best_model_{file_index}.joblib')  
                
                features_test_path = os.path.join(save_path, f'features_test_{file_index}.joblib')
                labels_test_path = os.path.join(save_path, f'labels_test_{file_index}.joblib')

                summary_save_path = os.path.join(save_path, f'summary_{file_index}.json')
                
                dump(best_model, model_save_path)

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
    # indices_to_process = [0, 1, 2, 4]
    indices_to_process = range(0, 11)  # Specify as needed
    
    # Call the function to process files
    """ Setting 'None' for indices_to_process, or by excluding this in function call which is None by default, will run all projections with specified file pattern."""
    process_files(train_directory_path, test_directory_path, results_directory_path, indices_to_process=indices_to_process)