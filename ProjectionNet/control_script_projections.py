import subprocess
import os

# Base directory paths
home_dir = os.path.expanduser('~')
script_directory = os.path.join(home_dir, 'Desktop', 'Production_codes')

# Scripts and their corresponding indices
scripts_to_run = {
    "projection_train_best": {"run_script": True, "indices": [12]}, # 27
    "projection_train": {"run_script": False, "indices": [ 2, 6, 12, 15, 16, 22, 24]}, 
    "projection_test": {"run_script": True,  "indices": [12]} 
}

def run_script(script_name, indices=None):
    """
    Executes a Python script based on its name.
    """
    try:
        script_path = os.path.join(script_directory, f"{script_name}.py")
        
        """ Using the Python '-u' option to force the stdout and stderr streams to be unbuffered.
        This ensures that output is displayed in real-time when the script is executed via subprocess.
        This approach resolves issues with output ordering and buffering, making subprocess output consistent
        with direct execution output. """
        
        cmd = ["python", "-u", script_path]
        if indices is not None:
            cmd.extend(["--indices"] + list(map(str, indices)))  # Convert indices to strings and add as separate arguments
        subprocess.run(cmd, check=True)
        print(f"Completed running {script_name}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")
    except FileNotFoundError:
        print(f"Script {script_name}.py not found in {script_directory}.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def main():
    for script_name, settings in scripts_to_run.items():
        if settings.get("run_script", False):
            indices = settings.get("indices") 
            run_script(script_name, indices=indices)

if __name__ == "__main__":
    main()
