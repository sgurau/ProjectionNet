import subprocess
import importlib.util
import os
import argparse
import numpy as np

# Base directory paths
home_dir = os.path.expanduser('~')
models_directory = os.path.join(home_dir, 'Desktop', 'Production_codes')
evaluation_script_path = os.path.join(models_directory, 'Performance_evaluation.py')
mlp_evaluation_script_path = os.path.join(models_directory, 'MLP_performance_evaluation.py')

# Models, their corresponding indices, custom thresholds for evaluation, and output folder pattern
models_to_run = {
    "SVM_model": {"run_model": False,"run_eval": False, "concat": False, "indices": range(0,31), "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "SVM_model_{file_index}"},
    "MLP_model": {"run_model": False, "run_eval": False,"concat": False, "indices": [26], "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "MLP_model_{file_index}"},
    "sklearnMLP_model": {"run_model": False, "run_eval": False,"concat": False, "indices": range(0,31), "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "sklearnMLP_model_{file_index}"},
    "SVM_model_best": {"run_model": True,"run_eval": True, "concat": True, "indices": range(0, 31), "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "SVM_model_{file_index}"},
    "MLP_model_best": {"run_model": True, "run_eval": True,"concat": True, "indices": range(0, 31), "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "MLP_model_{file_index}"},
    "sklearnMLP_model_best": {"run_model": True, "run_eval": True,"concat": True, "indices": range(0, 31), "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "output_pattern": "sklearnMLP_model_{file_index}"},
    }

def run_model(model_name, indices, output_pattern):
    """
    Dynamically imports and runs model script with specified indices,
    """
    try:
        # Run the model
        model_path = os.path.join(models_directory, f"{model_name}.py")
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Determine base paths based on the model_name
        base_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections')
        train_directory_path = os.path.join(base_path, 'Train')
        test_directory_path = os.path.join(base_path, 'Test')
        
        results_directory_path =  os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections', 'Results')
        
        module.process_files(train_directory_path, test_directory_path, results_directory_path, indices_to_process=indices)
        print(f"Completed running {model_name} with indices {indices}.")
    
    except FileNotFoundError:
        print(f"Model script {model_name}.py not found in {models_directory}.")
    except AttributeError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def evaluate_model(model_name, indices, thresholds, output_pattern):
    for index in indices:
        # Generate the path for the current index
        base_path_for_model_output = os.path.join(
            os.getenv('HOME'), 'Desktop', 'ProjectionNet', 'Projections', 'Results',
            output_pattern.format(file_index=index)
        )
        
        # Set the environment variable for the base path for current index
        os.environ['MODEL_OUTPUT_PATH'] = base_path_for_model_output
        
        # Select the appropriate evaluation script based on the model_name
        script_path = mlp_evaluation_script_path if model_name in ["MLP_model", "MLP_model_best"] else evaluation_script_path

        # Construct the command with the current index and the list of thresholds
        command = [
            'python', script_path,
            '--file-indices-to-evaluate', str(index),
            '--custom-thresholds', *map(str, thresholds),
            '--base-path', base_path_for_model_output
            ]
        
        print("=" * 150)
        print(f"Running evaluation for {model_name} with index {index} and thresholds {thresholds} including default threshold 0.5")
        print("=" * 150)
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the evaluation script for index {index}: {e}")

def concat_metrics(model_name, indices, output_pattern):
    # Base directory where the models' directories are located
    base_directory_path = os.path.join(os.getenv('HOME'), 'Desktop', 'ProjectionNet', 'Projections')
    
    # Convert output_pattern to a directory pattern suitable for globbing
    dir_pattern = output_pattern.replace("{file_index}", "*")
    
    # Construct command for the subprocess
    command = [
        'python', os.path.join(models_directory, 'Concatenate_metrics.py'),
        '--dir-pattern', dir_pattern,
        '--base-directory', base_directory_path  
    ]
    
    print("=" * 100)
    print(f"Running metrics concatenation for {model_name} using pattern {dir_pattern} in base directory {base_directory_path}")
    print("=" * 100)
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the concatenate_metrics script: {e}")

# def main(run_models_flag=False, evaluate_models_flag=False):
def main(run_models_flag=False, evaluate_models_flag=False, concat_metrics_flag=False):
    # Main functionality
    for model_name, settings in models_to_run.items():
        if run_models_flag and settings.get("run_model", False):
            run_model(model_name, settings["indices"], settings["output_pattern"])
        if evaluate_models_flag and settings.get("run_eval", False):
            evaluate_model(model_name, settings["indices"], settings["thresholds"], settings["output_pattern"])
        if concat_metrics_flag and settings.get("concat", False):
            concat_metrics(model_name, settings["indices"], settings["output_pattern"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and/or Evaluate Models")
    parser.add_argument('--run-models', action='store_true', help='Run selected models')
    parser.add_argument('--evaluate-models', action='store_true', help='Evaluate selected models')
    parser.add_argument('--concat-metrics', action='store_true', help='Concatenate metrics for selected models')
    
    args = parser.parse_args()

    # Directly calling main with parsed CLI arguments
    main(run_models_flag=args.run_models, evaluate_models_flag=args.evaluate_models, concat_metrics_flag=args.concat_metrics)
        
    # if no CLI arguments are provided, or based on some other condition:
    if not (args.run_models or args.evaluate_models or args.concat_metrics):
        # Default behavior when no arguments are provided
        print("No arguments provided. Running default behavior.")
        # Update here to specify the default behavior for concat_metrics_flag as well
        main(run_models_flag=True, evaluate_models_flag=True, concat_metrics_flag=True)  # default behavior

    
