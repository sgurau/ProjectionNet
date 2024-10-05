""" Install opeenpyxl using: pip install openpyxl."""
""" Change dir_pattern accordingly to model. """

import argparse
import os
import glob
import re
import pandas as pd

def find_directories(directory_path, dir_pattern):
    # Use the ** wildcard to search directories recursively and match the pattern
    # dir_paths = glob.glob(f"{directory_path}/**/{dir_pattern}", recursive=True)
    dir_paths = glob.glob(f"{directory_path}/{dir_pattern}")
    return dir_paths

def extract_model_info(dir_pattern):
    # Split pattern at '_model_' to get the base model name and version details
    parts = dir_pattern.split('_model_')
    base_model_name = parts[0]
    
    # Remove the wildcard '*' at the end; 
    version_details = parts[1].rstrip('*').rstrip('_')
    
    # Reformat the version details 
    if version_details:
        version_details = '_' + version_details
    
    # Construct the final model name by combining the base model name with version details
    final_model_name = base_model_name + version_details
    
    return final_model_name

def process_files_in_directories(directories, file_pattern, output_csv, output_excel_path):
    concatenated_df = pd.DataFrame()

    for directory in directories:
        csv_files = glob.glob(f"{directory}/{file_pattern}")
        
        for file in csv_files:
            # Extract the run ID and the entire suffix/version info
            match = re.search(r'performance_metrics_(\d+)_([a-zA-Z0-9_]+).csv', file)
            if match:
                file_id = int(match.group(1))
                version_info = match.group(2)
                
                # Optional: Convert numeric suffixes like "0_3" to "0.3" for clarity
                if re.match(r'^\d+_\d+$', version_info):
                    version_info = version_info.replace('_', '.')
                
                df = pd.read_csv(file)
                # Insert the Run ID and Converted Version Info at the start
                df.insert(0, 'Version', version_info) #threshold
                df.insert(0, 'Run ID', file_id)
                
                concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    # Save final concatenated DataFrame to both CSV and Excel after processing all files
    concatenated_df.to_csv(output_csv, index=False)
    concatenated_df.to_excel(output_excel_path, index=False)

def main(dir_pattern):
    home_dir = os.path.expanduser('~')
    directory_path = os.path.join(home_dir, 'Desktop', 'ProjectionNet', 'Projections', 'Results')

    model_name = extract_model_info(dir_pattern)
    directories = find_directories(directory_path, dir_pattern)
    file_pattern = 'performance_metrics_*.csv'
    
    output_csv_path = os.path.join(directory_path, f"{model_name}_concatenated_performance_metrics_all_versions.csv")
    output_excel_path = os.path.join(directory_path, f"{model_name}_concatenated_performance_metrics_all_versions.xlsx")
    

    process_files_in_directories(directories, file_pattern, output_csv_path, output_excel_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate metrics from specified directories.')
    parser.add_argument('--dir-pattern', type=str, required=True,
                        help='The pattern for directories to match.')
    parser.add_argument('--base-directory', type=str, required=True,
                        help='The base directory within which to search for matching directories.')
    args = parser.parse_args()
    
    main(args.dir_pattern)


