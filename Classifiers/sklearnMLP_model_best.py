import os
import json
from joblib import dump
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score 
from sklearn.model_selection import StratifiedKFold
from utils import load_from_hdf5

np.random.seed(43) #43 original

def load_data(file_path, dataset_type='train'):
    dataset_name = f'projection_{dataset_type}'
    projection_data = load_from_hdf5(file_path, dataset_name)
    features = projection_data[:, :-1]
    labels = projection_data[:, -1]
    return features, labels

def train_and_evaluate_with_best_params(features, labels, best_params, model_architecture):
    # Construct hidden_layer_sizes from model_architecture
    hidden_layer_sizes = tuple(model_architecture['layers'].values())

    # Create the MLP model with best parameters
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          activation=best_params['activation'],
                          solver=best_params['solver'],
                          alpha=best_params['alpha'],
                          learning_rate=best_params['learning_rate'],
                          max_iter=3000)

    # Train and evaluate classifier with best parameters found
    best_model = model.fit(features, labels)

    return model 

def process_files(train_directory_path, test_directory_path, results_directory_path, file_prefix='projection_', indices_to_process=None):
    
    # Manually set the best parameters and model architecture
    best_params = {
        "activation": "tanh",
        "solver": "sgd",
        "alpha": 0.46050382439402626,
        "learning_rate": "constant"
    }

    model_architecture = {
        "n_layers": 1,
        "layers": {
            "layer_0": 34
            }
    }

    for filename in os.listdir(train_directory_path):
        if filename.startswith(file_prefix + 'train_') and filename.endswith('.h5'):
            file_index = int(filename.split('_')[-1].split('.')[0])

            if indices_to_process is None or file_index in indices_to_process:
                train_file_path = os.path.join(train_directory_path, filename)
                test_file_path = os.path.join(test_directory_path, f'projection_test_{file_index}.h5')

                features, labels = load_data(train_file_path, 'train')

                model = train_and_evaluate_with_best_params(features, labels, best_params, model_architecture)

                print(f"Training File Index: {file_index} - Best Parameters: {best_params}")

                if os.path.exists(test_file_path):
                    features_test, labels_test = load_data(test_file_path, 'test')
                    test_predictions = model.predict(features_test)
                    test_accuracy = accuracy_score(labels_test, test_predictions)
                    print(f"Test File Index: {file_index} - Test Accuracy: {test_accuracy:.2f}")

                summary_data = {
                    'file_index': file_index,
                    'best_parameters': best_params,
                    'model_architecture': model_architecture,
                    'test_accuracy': test_accuracy
                }

                save_path = os.path.join(results_directory_path, f'sklearnMLP_model_{file_index}')

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                model_save_path = os.path.join(save_path, f'best_model_{file_index}.joblib')
                dump(model, model_save_path)
                
                features_test_path = os.path.join(save_path, f'features_test_{file_index}.joblib')
                labels_test_path = os.path.join(save_path, f'labels_test_{file_index}.joblib')
                
                dump(features_test, features_test_path)
                dump(labels_test, labels_test_path)

                summary_save_path = os.path.join(save_path, f'summary_{file_index}.json')
                with open(summary_save_path, 'w') as f:
                    json.dump(summary_data, f, indent=4)

                print(f"Summary for file index {file_index} saved to {summary_save_path}")

if __name__ == "__main__":
    home_dir = os.path.expanduser('~')
    base_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections')

    train_directory_path = os.path.join(base_path, 'Train')
    test_directory_path = os.path.join(base_path, 'Test')
    results_directory_path = os.path.join(base_path, 'Results')

    indices_to_process = range(0, 11)

    process_files(train_directory_path, test_directory_path, results_directory_path, indices_to_process=indices_to_process)
