""" Data is split outside the optimization code blocks and used to find optimized set of parameters for the model 
    by optuna. This is done beacuse the MLP needs to be retrained with the best parameters and the same data used for 
    optimization before it can be saved unlike models in the sklearn library. 
    n_trials (100 models oprimized) * n_splits (10 splits cross-validated per model)
    { * epochs (early_stopping)}"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from joblib import dump
from utils import load_from_hdf5
import json
import matplotlib.pyplot as plt
from optuna.pruners import HyperbandPruner  

# Set random seed for reproducibility
torch.manual_seed(47)
# np.random.seed(74)  

# Define MLP classifier
class FlexMLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(FlexMLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], 1))  # For binary classification
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.layers(x))

def load_data(file_path, dataset_type='train'):
    # Choose dataset based on type
    dataset_name = f'projection_{dataset_type}'
    
    # Load the combined features and labels from the HDF5 file
    projection_data = load_from_hdf5(file_path, dataset_name)
    
    # Split the projection_data into features and labels
    features = projection_data[:, :-1]  # All rows, all but the last column
    labels = projection_data[:, -1]  # All rows, only the last column
            
    return features, labels

def train_and_evaluate(model, optimizer, loss_function, train_loader, val_loader, epochs):
    epoch_losses = {"train": [], "val": []}  # To store average losses per epoch
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            
            loss = loss_function(output.squeeze(1), target)
            
            """For Debugging..."""
            # if output.dim() > 1:  # More than one dimension indicates a potential issue
            #     output = output.squeeze(1)
            # if output.shape != target.shape:
            #     print(f"Shape mismatch detected: Output {output.shape}, Target {target.shape}")             
            # print(f"Output shape: {output.shape}, Target shape: {target.shape}")
            # loss = loss_function(output, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        epoch_losses["train"].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                
                # loss = loss_function(output.squeeze(1), target) 
                
                if output.dim() > 1:  # More than one dimension indicates a potential issue
                    output = output.squeeze(1)
                loss = loss_function(output, target)
                
                val_loss += loss.item() * data.size(0)
                predicted = (output > 0.5).float()
                correct += (predicted.squeeze() == target).sum().item()
        val_loss /= len(val_loader.dataset)
        epoch_losses["val"].append(val_loss)
        
        accuracy = correct / len(val_loader.dataset)
        # Optional: Print average losses and accuracy for each epoch
        #print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    return accuracy, epoch_losses 

# Optuna objective function
def objective(trial, features, labels):
    # try:       
        losses = []
        accuracies = []  # To track accuracy separately
        
        # Hyperparameters to optimize
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True) #('lr', 1e-4, 1e-2, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
            
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      
        for train_index, val_index in skf.split(features, labels):
            X_train, X_val = features[train_index], features[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            
            train_dataset = TensorDataset(X_train.clone().detach().float(), y_train.clone().detach().float())
            val_dataset = TensorDataset(X_val.clone().detach().float(), y_val.clone().detach().float())
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            
            # Dynamically suggest the number of units for each layer
            hidden_layers = [trial.suggest_int(f'units_layer_{i}', 1, 100) for i in range(num_layers)]
            
            """For Debugging..."""
            # for batch in train_loader:
            #     data, target = batch
            #     print(f"Batch data shape: {data.shape}, Batch target shape: {target.shape}")
            #     break  # Check the shape of the first batch

            model = FlexMLP(X_train.shape[1], hidden_layers)
            
            optimizer_class = getattr(optim, optimizer_name)
            optimizer = optimizer_class(model.parameters(), lr=lr)
            loss_function = nn.BCELoss()
            
            fold_accuracy, epoch_losses = train_and_evaluate(model, optimizer, loss_function, train_loader, val_loader, epochs=400)
               
            # Collect the final validation loss for each fold
            losses.append(epoch_losses["val"][-1])  # Append the last validation loss of the fold
            accuracies.append(fold_accuracy)  
            
        # Compute mean loss for Optuna to minimize
        mean_loss = np.mean(losses)
        
        # Compute mean accuracy 
        mean_accuracy = np.mean(accuracies)
        
        # Log mean accuracy using set_user_att
        trial.set_user_attr("mean_accuracy", mean_accuracy)
   
        # Mean val loss is returned for Optuna's optimization
        return mean_loss
    
    # except Exception as e:
    #     print(f"An error occurred during trial: {e}")
    #     return None
                
# def plot_losses(epoch_losses, save_path, file_index):
#     train_losses = epoch_losses["train"]
#     val_losses = epoch_losses["val"]
#     epochs = range(1, len(train_losses) + 1)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_losses, label='Training Loss', linestyle='--') 
#     plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--') 
#     plt.title(f'Training and Validation Loss - {file_index}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.show()
#     plt.close()

def process_files(train_directory_path, test_directory_path, results_directory_path, file_prefix='projection_', indices_to_process=None):
    for filename in os.listdir(train_directory_path):
        if filename.startswith(file_prefix + 'train_') and filename.endswith('.h5'):
            file_index = int(filename.split('_')[-1].split('.')[0])
            if indices_to_process is None or file_index in indices_to_process:
                train_file_path = os.path.join(train_directory_path, filename)
                test_file_path = os.path.join(test_directory_path, f'projection_test_{file_index}.h5')

                features, labels = load_data(train_file_path, 'train')
                features_test, labels_test = load_data(test_file_path, 'test')

                # Perform hyperparameter tuning and retraining
                pruner = HyperbandPruner()
                study = optuna.create_study(direction='minimize', pruner=pruner)
                study.optimize(lambda trial: objective(trial, features, labels), n_trials=200)
                best_params = study.best_trial.params
                
                # Retrieve the best trial's mean accuracy
                best_trial = study.best_trial
                mean_accuracy = best_trial.user_attrs['mean_accuracy']
                mean_loss = best_trial.value  # This is the mean loss that Optuna optimized

                save_path = os.path.join(results_directory_path, f'MLP_model_{file_index}')
                # test_accuracy, test_loss = retrain_best_model(features, labels, features_test, labels_test, best_params, file_index, save_path)
                
                # Retrieve best parameters for model initialization
                input_size = features.shape[1]  # Dimension of features
                num_layers = best_params['num_layers']
                hidden_layers = [best_params[f'units_layer_{i}'] for i in range(num_layers)]
                lr = best_params['lr']
                batch_size = best_params['batch_size']
                optimizer_name = best_params['optimizer']

                # Initialize and train the model
                model = FlexMLP(input_size, hidden_layers)
                optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
                loss_function = torch.nn.BCELoss()

                # Prepare DataLoaders
                train_dataset = TensorDataset(features.clone().detach().float(), labels.clone().detach().float())
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataset = TensorDataset(features_test.clone().detach().float(), labels_test.clone().detach().float())
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Retrain the model with best parameters
                #  _, epoch_losses = train_and_evaluate(model, optimizer, loss_function, train_loader, train_loader, epochs=400)  

                model.train()
                for epoch in range(400):  # Example of training for 400 epochs
                    for data, target in train_loader:
                        optimizer.zero_grad()
                        output = model(data)
                        loss = loss_function(output.squeeze(1), target)
                        loss.backward()
                        optimizer.step()
                  
                # Evaluate on test data
                test_loss = 0.0
                correct = 0
                model.eval()
                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        loss = loss_function(output.squeeze(1), target)
                        test_loss += loss.item() * data.size(0)
                        predicted = (output > 0.5).float()
                        correct += (predicted.squeeze() == target).sum().item()

                test_loss /= len(test_loader.dataset)
                test_accuracy = correct / len(test_loader.dataset)
                
                # Check if the directory exists, if not, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                # Save the model
                model_save_path = os.path.join(save_path, f"best_model_{file_index}.pt")
                torch.save(model.state_dict(), model_save_path)
                
                # Construct the paths to save test features and labels
                features_test_path = os.path.join(save_path, f"features_test_{file_index}.pt")
                labels_test_path = os.path.join(save_path, f"labels_test_{file_index}.pt")
                  
                # Saving the tensors
                torch.save(features_test, features_test_path)
                torch.save(labels_test, labels_test_path)

                # Save the hyperparameters and accuracy
                metadata_save_path = os.path.join(save_path, f"best_model_metadata_{file_index}.json")
                # metadata_save_path = os.path.join(save_path, f"best_model_metadata_{file_index}.joblib")
                metadata = {
                    "best_hyperparameters": best_params,
                    "input_size": input_size,
                    "mean_cv_accuracy": mean_accuracy,
                    "mean_cv_loss": mean_loss,
                    "test_accuracy": test_accuracy
                }
                                
                # Save the summary data as JSON
                with open(metadata_save_path, 'w') as f:
                   json.dump(metadata, f, indent=4)
                
                print(f"Model for file index {file_index} saved to {model_save_path}")
                print(f"Metadata for file index {file_index} saved to {metadata_save_path}")
                
                # Plotting and saving the losses
                # plot_save_path = os.path.join(save_path, f"loss_plot_{file_index}.png")
                # plot_losses(epoch_losses, plot_save_path, file_index)  
                # print(f"Loss plot saved to {plot_save_path}")
                
                print(f"File Index: {file_index} - Best Parameters: {best_params} - Accuracy: {mean_accuracy:.2f}")
             
if __name__ == "__main__": # This block will ensure the code within will run only when the script is exectued directly and not when imported elsewhere (evaluation script) 
    # directory_path = '/home/sgurau/Desktop/BrainProjectionNet/Projections/'
    
    # Get home directory of current user
    home_dir = os.path.expanduser('~')
    base_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections')
    
    # Define directory where files are located
    train_directory_path = os.path.join(base_path, 'Train')
    test_directory_path = os.path.join(base_path, 'Test')
    results_directory_path =  os.path.join(base_path, 'Results')
    
    indices_to_process = [1, 2, 3, 4, 5]  # Example: [0, 1, 2, 3, 4] / range(0,31) 
    process_files(train_directory_path, test_directory_path, results_directory_path, indices_to_process=indices_to_process)



