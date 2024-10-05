import os
import numpy
import optuna
import pandas as pd
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from projection_net import ProjectionNet, PairwiseDataset # import ProjectionNet and PairwiseDataset from projection_net script
from optuna.pruners import HyperbandPruner  
from utils import save_to_hdf5  
import matplotlib.pyplot as plt

torch.manual_seed(41)

def train_model(model, optimizer, pairwise_loader, epochs=300): # If default epochs value(=400) is unspecified here, it should always be specified during function call   
    mse_loss = nn.MSELoss()
    model.train()    
    epoch_losses = []

    best_avg_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10
    
    # patience = trial.suggest_int('patience', 5, 20)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False) 
    
    # # Scheduler parameters suggested by Optuna
    # scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.5)
    # scheduler_patience = trial.suggest_int('scheduler_patience', 5, 15)

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        for Di, Dj, Dij in pairwise_loader:
            optimizer.zero_grad()
            Pi = model(Di)
            Pj = model(Dj)
            loss = mse_loss(torch.norm((Pi - Pj) * 100, dim=1), Dij)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f'Epoch {epoch+1}, Loss: {total_loss / len(pairwise_loader)}')
        train_loss = total_loss / len(pairwise_loader)
        epoch_losses.append(train_loss)  # Store the average loss for current epoch
        print(f'Epoch {epoch+1}, Loss: {train_loss}')
        
        # # Learning Rate Scheduling
        # scheduler.step(train_loss)
        
        # Early stopping logic
        if train_loss < best_avg_loss:
            best_avg_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
                
    # return total_loss / len(pairwise_loader)
    return train_loss, epoch_losses  # Return the last average loss

def objective(trial, featurues_train_tensor):
    # Hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]) # SGD
    projection_size = trial.suggest_int("projection_size", 1, 50)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    activation_name = trial.suggest_categorical("activation_fn", ["ReLU", "Tanh", "Sigmoid"]) # LeakyReLU
        
    # dynamically gets the activation function class based on its name and instantiates it
    activation_fn = getattr(nn, activation_name)()
    
    # Initialize dataset and DataLoader for the split data
    pairwise_dataset = PairwiseDataset(featurues_train_tensor)
    pairwise_loader = DataLoader(pairwise_dataset, batch_size=batch_size, shuffle=True)

    projection_net = ProjectionNet(input_size=featurues_train_tensor.shape[1],
                                   projection_size=projection_size,
                                   num_layers=num_layers,
                                   activation_fn=activation_fn)
    optimizer = getattr(optim, optimizer_name)(projection_net.parameters(), lr=lr)
    
    loss, _ = train_model(projection_net, optimizer, pairwise_loader)  # Skip epochs to use default value in when its specified or specify epochs= some interger value to override default    
    return loss

def plot_training_loss(epoch_losses, save_path, file_index):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Training Loss Over Epochs - {file_index}')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, f'training_loss_{file_index}.png'))
    plt.show()
    plt.close()  # Close plot to free up memory

def train_projections(dir_path, save_path, file_prefix='train_split_', file_suffix = '.csv', indices_to_process=None):
    train_path = os.path.join (dir_path, 'Splits', 'Train_splits', 'Train_csv')
    for filename in os.listdir(train_path):
        if filename.startswith(file_prefix) and filename.endswith(file_suffix):
            file_index = int(filename.split('_')[-1].split('.')[0])
            
            if indices_to_process and file_index in indices_to_process:
                # Specify the path for training splits
                train_file= os.path.join(train_path, filename)
                
                df_train = pd.read_csv(train_file, index_col=False)
                
                # Split the combined_data into projected_features and labels
                features_train = df_train.iloc[:, 1:-1]  # All rows, all but the first and last columns
                labels_train = df_train.iloc[:, -1]  # All rows, only the last column
                print('Training features shape:',features_train.shape)
                
                features_train_numpy = features_train.to_numpy(dtype=float)
                featurues_train_tensor = torch.tensor(features_train_numpy, dtype=torch.float32)
                labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
                                
                print(f"Processing Projection split {file_index}...")
                
                # Initialize Optuna study to find the best hyperparameters              
                n_trials = 20
                pruner = HyperbandPruner(min_resource=50,  # Minimum number of epochs
                                          max_resource=300, # Maximum number of epochs
                                          reduction_factor=3)
                
                study = optuna.create_study(direction='minimize', pruner=pruner)
                study.optimize(lambda trial: objective(trial, featurues_train_tensor), n_trials=n_trials)
                best_hyperparams = study.best_params
                print("Best hyperparameters:", study.best_params)
                
                # Use getattr to dynamically get the activation function class from nn based on the best name
                best_activation_fn = getattr(nn, best_hyperparams['activation_fn'])()
                
                # Train model with best hyperparameters found by Optuna 
                projection_net = ProjectionNet(input_size=featurues_train_tensor.shape[1], 
                                               projection_size=best_hyperparams['projection_size'], 
                                               num_layers=best_hyperparams['num_layers'],
                                               activation_fn=best_activation_fn)
                optimizer = getattr(torch.optim, best_hyperparams['optimizer'])(projection_net.parameters(), lr=best_hyperparams['lr'])
                pairwise_dataset = PairwiseDataset(featurues_train_tensor)
                pairwise_loader = DataLoader(pairwise_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
                train_loss, epoch_losses = train_model(projection_net, optimizer, pairwise_loader)  # Skip epochs to use default value when its specified or specify epochs= some integer value to override default    
                
                plot_training_loss(epoch_losses, save_path, file_index)
                       
                projection_net.eval()
                with torch.no_grad():
                    projected_features = projection_net(featurues_train_tensor)
                    # print(f"Shape of projected features for split {i}: {projected_features.shape}")
                                  
                # Append labels to the projected features
                labels_train_expanded = labels_train_tensor.unsqueeze(1)
                projection_train = torch.cat((projected_features, labels_train_expanded), dim=1)
                                
                # Define save paths
                projection_save_path = os.path.join(save_path, f'projection_train_{file_index}.h5')    
                model_state_path = os.path.join(save_path, f'best_projection_model_state_dict_{file_index}.pth')
                config_path = os.path.join(save_path, f'model_config_{file_index}.json')
                
                # Save projection and best model state
                save_to_hdf5(projection_save_path, projection_train.numpy(), 'projection_train')                
                torch.save(projection_net.state_dict(), model_state_path)
                
                # Define configuration data in a dictionary before opening the config file
                config_data = {
                    'split_number': file_index,  # Iteration (split) number
                    'input_size': featurues_train_tensor.shape[1],
                    'projection_size': best_hyperparams['projection_size'],
                    'num_layers': best_hyperparams['num_layers'],
                    'lr': best_hyperparams['lr'],
                    'optimizer': best_hyperparams['optimizer'],
                    'batch_size': best_hyperparams['batch_size'],
                    'activation_fn': best_hyperparams['activation_fn'],
                    'last_avg_loss': train_loss, 
                }
                
                # Open config file and save the predefined dictionary
                with open(config_path, 'w') as config_file:
                    json.dump(config_data, config_file, indent=4)
                
                print(f"Configuration for split number {file_index} saved to {config_path}")     
                print("Loop over splits completed.")
    
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train projections")
    parser.add_argument("--indices", type=int, nargs="+", help="Indices to process")
    args = parser.parse_args()
    
    # Define path for loading splits
    home_dir = os.path.expanduser('~')
    dir_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet')
    
    # Define save path for projections
    save_path = os.path.join(dir_path, 'Projections', 'Train')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_projections(dir_path, save_path, indices_to_process=args.indices)

if __name__ == "__main__":
    main()