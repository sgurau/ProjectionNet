import os
import pandas as pd
import torch
import json
import optuna
from projection_net import ProjectionNet 
import torch.nn as nn 
from utils import save_to_hdf5

def load_settings_from_json(settings_path):
    with open(settings_path, 'r') as f:
        settings = json.load(f)
        
    # Directly use getattr to dynamically get the activation function class from nn based on the saved name and instantiate it
    activation_fn = getattr(nn, settings["activation_fn"])() # Instantiate the class usinng ()
    
    # Extract parameters needed for initializing ProjectionNet, including the instantiated activation function
    model_settings = {
        "input_size": settings["input_size"],
        "projection_size": settings["projection_size"],
        "num_layers": settings["num_layers"],
        "activation_fn": activation_fn
        }
   
    return model_settings

def test_projections(dir_path, save_path, model_save_path, file_prefix='test_split_', file_suffix='.csv', indices_to_process=None):
    test_path = os.path.join(dir_path, 'Splits', 'Test_splits', 'Test_csv')
    for filename in os.listdir(test_path):
        if filename.startswith(file_prefix) and filename.endswith(file_suffix):
            file_index = int(filename.split('_')[-1].split('.')[0])
            
            if indices_to_process is None or file_index in indices_to_process:
                test_file = os.path.join(test_path, filename)
                
                df_test = pd.read_csv(test_file, index_col=False)
                features_test = df_test.iloc[:, 1:-1]
                labels_test = df_test.iloc[:, -1]  # You might or might not need labels depending on your use-case
                print('Training features shape:',features_test.shape)
               
                features_test_numpy = features_test.to_numpy(dtype=float)
                features_test_tensor = torch.tensor(features_test_numpy, dtype=torch.float32)
                labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)
                                
                # Load the settings for ProjectionNet
                settings_path = os.path.join(model_save_path, f'model_config_{file_index}.json')
                projection_net_settings = load_settings_from_json(settings_path)
                
                # Load the best model for this index
                model_state_path = os.path.join(model_save_path, f'best_projection_model_state_dict_{file_index}.pth')
                projection_net = ProjectionNet(**projection_net_settings)  # Initialize the model with only the necessary parameters
                projection_net.load_state_dict(torch.load(model_state_path))
                                
                # Run the model on test data
                projection_net.eval()
                with torch.no_grad():
                    projected_features_test = projection_net(features_test_tensor)
                
                # Append labels to the projected features
                labels_test_expanded = labels_test_tensor.unsqueeze(1)
                projection_test = torch.cat((projected_features_test, labels_test_expanded), dim=1)  
                    
                # Save projected test features
                projection_save_path = os.path.join(save_path, f'projection_test_{file_index}.h5')
                save_to_hdf5(projection_save_path, projection_test.numpy(), 'projection_test')
                
                print(f"Projection split {file_index} processed.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Test projections")
    parser.add_argument("--indices", type=int, nargs="+", help="Indices to process")
    args = parser.parse_args()
    
    # Define paths to load splits and saved projection model    
    home_dir = os.path.expanduser('~')
    dir_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet')
    model_save_path = os.path.join(dir_path, 'Projections', 'Train')
    
    # Define save path for projections
    save_path = os.path.join(dir_path, 'Projections', 'Test')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    test_projections(dir_path, save_path, model_save_path, indices_to_process=args.indices)

if __name__ == "__main__":
    main()