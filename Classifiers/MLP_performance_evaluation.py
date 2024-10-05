"""UPDATED  TO ADAPT TO MLP models......."""
""" Dynamically loads model, test data and predictions for a proejction to compute and visualize preformance."""
""" Change the file_index value at the end of the script to load and evaluate different models and datasets."""
""" Dependency MLP_flexible_layer_nodes_stratified_CV_stable.py"""

import argparse
import os
import torch
import json
from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from MLP_model import FlexMLP

def load_metadata(base_path, file_index):
    metadata_path = os.path.join(base_path, f"best_model_metadata_{file_index}.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_model(base_path, file_index):
    # Load the metadata
    metadata = load_metadata(base_path, file_index)
    
    # Extract hyperparameters (ensure this matches how you've structured your metadata)
    input_size = metadata['input_size'] 
    params = metadata['best_hyperparameters']
    num_layers = params['num_layers']
    # units_per_layer = params['units_per_layer']
    
    # Dynamically compute hidden_layers sizes from metadata
    hidden_layers = [params[f'units_layer_{i}'] for i in range(num_layers)]
    
    # Initialize the model with the correct structure
    # hidden_layers = [units_per_layer] * num_layers     # Compute hidden_layers after getting the values from metadata
    model = FlexMLP(input_size, hidden_layers)
 
    # Load the saved state dictionary
    model_path = os.path.join(base_path, f"best_model_{file_index}.pt")
    model.load_state_dict(torch.load(model_path))
    
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, features_test):
    # Ensure features_test is a tensor
    if not isinstance(features_test, torch.Tensor):
        features_test = torch.tensor(features_test, dtype=torch.float32)
    model.eval()    
    with torch.no_grad():
        outputs = model(features_test).squeeze()
    # For a binary classification problem, apply sigmoid to convert outputs to probabilities
    probabilities = torch.sigmoid(outputs).numpy()
    return probabilities  # Return just the probabilities

def calculate_metrics(labels_test, predictions, model, probabilities, file_index, base_path, threshold_desc): # labels_test = labels_test; predictions = predictions
    auc_score = roc_auc_score(labels_test, probabilities)
    precision = precision_score(labels_test, predictions, zero_division = 1)
    recall = recall_score(labels_test, predictions)
    f1 = f1_score(labels_test, predictions)
    test_accuracy = accuracy_score(labels_test, predictions)
    conf_matrix = confusion_matrix(labels_test, predictions)
    
    # Calculate FPR, TPR for calculating and plotting ROC/ Does the same thing as auc_score
    fpr, tpr, thresholds = roc_curve(labels_test, probabilities)
    roc_auc = auc(fpr, tpr)

    # Print the metrics
    print_evaluation_results(file_index, auc_score, precision, recall, f1, test_accuracy, conf_matrix, threshold_desc)
    
    # Plot ROC Curve
    plot_roc_curve(fpr, tpr, roc_auc, base_path, file_index)
    
    # Plot and save the confusion matrix, passing threshold_desc directly
    plot_confusion_matrix(labels_test, predictions, model, base_path, file_index, threshold_desc)
    
    # Save metrics to JSON
    save_metrics(file_index, base_path, threshold_desc, auc_score, precision, recall, f1, test_accuracy, fpr, tpr, thresholds, roc_auc, conf_matrix)

def scalar_performance_metrics_to_csv(metrics, csv_file):
    scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    df = pd.DataFrame([scalar_metrics])
    
    file_exists = os.path.isfile(csv_file)
    
    # with open(csv_file, 'a' if file_exists else 'w') as f: when opening in append mode with 'a'
    # 'w' is default mode that creates a new file and owerwrites existing file, can be omitted.
    with open(csv_file, 'w') as f:  
        df.to_csv(f, index=False, header=True)

def save_metrics(file_index, base_path, threshold_desc, auc_score, precision, recall, f1, test_accuracy, fpr, tpr, thresholds, roc_auc, conf_matrix):
    
    # Sanitize and standardize the threshold_desc for file naming
    safe_threshold_desc = threshold_desc.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_").lower()

    performance_metrics_path = os.path.join(base_path, f'performance_metrics_{file_index}_{safe_threshold_desc}.json')
    csv_file_path = os.path.join(base_path, f'performance_metrics_{file_index}_{safe_threshold_desc}.csv')

    
    results = {
        "AUC Score": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Test Accuracy": test_accuracy,
        "FPR": fpr.tolist(),  # Converting numpy array to list for JSON serialization
        "TPR": tpr.tolist(),
        "Thresholds": thresholds.tolist(),
        # "ROC AUC": roc_auc,
        "Confusion Matrix": conf_matrix.tolist()
    }

    with open(performance_metrics_path, 'w') as file:
        json.dump(results, file, indent=4)
    
    # Define the CSV file path
    scalar_performance_metrics_to_csv(results, csv_file_path)
    
    print(f"Performance metrics for file index {file_index} ({threshold_desc}) saved to {performance_metrics_path}")    
    print(f"Scalar performance metrics for file index {file_index} saved to {csv_file_path}")

def print_evaluation_results(file_index, auc_score, precision, recall, f1, test_accuracy, conf_matrix, threshold_desc):
    print(f"========= Results for File Index: {file_index}, threshold - {threshold_desc} =========")
    print(f"AUC Score: {auc_score:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Confusion Matrix\n: {conf_matrix}")
    print("=" * 50)

def plot_confusion_matrix(labels_test, predictions, model, base_path, file_index, threshold_desc):
    safe_threshold_desc = threshold_desc.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_").lower()

    # Convert tensors to numpy arrays if necessary
    if isinstance(labels_test, torch.Tensor):
        labels_test = labels_test.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Create the confusion matrix and plot it
    cm = confusion_matrix(labels_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - File Index: {file_index}, {threshold_desc}')
    
    plt.savefig(os.path.join(base_path,f"confusion_matrix_{file_index}_{safe_threshold_desc}.png"), dpi=300)
    # plt.savefig(f'{base_path}/confusion_matrix_{file_index}_{safe_threshold_desc}.png', dpi=300)
    plt.show()
    plt.close()

"""Plot ROC Curve to visualize AUC"""
def plot_roc_curve(fpr, tpr, roc_auc, base_path, file_index):
    # plt.style.use('Solarize_Light2')
    plt.style.use('ggplot')
    # plt.style.use('bmh')
    
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Receiver Operating Characteristic - {file_index}', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
      
    # # Save plot
    plt.savefig(os.path.join(base_path, f"ROC_Curve_{file_index}.png"), dpi=300) 
    # filename = f'ROC_Curve_{file_index}_{safe_threshold_desc}.png'
    # plt.savefig(f'{base_path}/{filename}', dpi=300)
    plt.show()
    plt.close() 
    
def load_and_evaluate_models(file_index, custom_threshold=None, base_path=None):
    
    # Use the hardcoded path as a fallback if the environment variable isn't set
    fallback_base_path = os.path.join(
        os.path.expanduser('~'), 'Desktop',
        'ProjectionNet',
        'Projections',
        'Results',
        f'MLP_model_{file_index}' 
        )
    
    if base_path is None:
        # Try to get the base path from the environment variable, otherwise use the fallback
        base_path = os.getenv('MODEL_OUTPUT_PATH', fallback_base_path)
        
        # # Dynamically get the home directory of the current user
        # home_dir = os.path.expanduser('~')
        # base_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections', 'Results', f'MLP_model_{file_index}')
        features_test_path = os.path.join(base_path, f'features_test_{file_index}.pt')
        labels_test_path = os.path.join(base_path, f'labels_test_{file_index}.pt')
        # performance_metrics_path = os.path.join(base_path, f'performance_metrics_{file_index}.json')
    
        model = load_model(base_path, file_index)
        features_test = torch.load(features_test_path)
        labels_test = torch.load(labels_test_path) # true labels
        
        # Use the predict function to get probabilities
        probabilities = predict(model, features_test)  # This now only returns probabilities
        
        # Option A to handle 0.5 in custom_threshold list 
        if custom_threshold is not None and custom_threshold != 0.5:
            predictions = (probabilities >= custom_threshold).astype(int)
            threshold_desc = f'threshold {custom_threshold}'
        else:
            predictions = (probabilities > 0.5).astype(int) # Predictions with default (0.5) threshold
            threshold_desc = 'threshold 0.5'
            
        calculate_metrics(labels_test, predictions, model, probabilities, file_index, base_path, threshold_desc)    
    
def evaluate_all_models(file_indices, custom_thresholds):
    for file_index in file_indices:
        # Always perform the default evaluation first
        load_and_evaluate_models(file_index)
        
        # Then, evaluate for each specified custom threshold
        for threshold in custom_thresholds:
            if threshold is not None:  # Ensure we don't duplicate the default evaluation
                load_and_evaluate_models(file_index, custom_threshold=threshold)

# # Example usage
# file_indices_to_evaluate = [0,1,2,3,4] # [0, 1, 2, 3, 4]  # Adjust according to files to process
# custom_thresholds = [0.3, 0.4, 0.6, 0.7]  # Specify custom thresholds here; Leave empty for Default (0.5); Do not include None in this list

# evaluate_all_models(file_indices_to_evaluate, custom_thresholds)

if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate ML models.")
    parser.add_argument('--file-indices-to-evaluate', nargs='+', type=int, help='List of file indices to evaluate', required=True)
    parser.add_argument('--custom-thresholds', nargs='*', type=float, help='List of custom thresholds for evaluation.')
    parser.add_argument('--base-path', type=str, help='Base path for loading models and saving results.')

    args = parser.parse_args()
           
    # In case of any issues, remove/comment out 'default' definitions and the 'fall back to default' block, then execute main() as follows:  
    evaluate_all_models(args.file_indices_to_evaluate, args.custom_thresholds) 


