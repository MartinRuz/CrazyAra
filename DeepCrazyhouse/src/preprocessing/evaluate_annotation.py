import numpy as np
import os


def calculate_difference(zarr_folder):
    # Get all zarr files in folder
    zarr_files = [f for f in os.listdir(zarr_folder) if f.endswith('.zip')]

    # Loop over all files
    for file in zarr_files:
        # Load data from file
        data = np.load(os.path.join(zarr_folder, file))

        # Get eval_init and eval_search arrays
        eval_init = data['eval_init']
        eval_search = data['eval_search']

        # Calculate difference
        diff = eval_init - eval_search

        # Calculate standard deviation and min,max of difference
        std_dev = np.std(diff)
        min_diff = np.min(diff)
        max_diff = np.max(diff)

        print(f"File: {file}")
        print(f"Standard deviation: {std_dev}")
        print(f"Min difference: {min_diff}")
        print(f"Max difference: {max_diff}")

if __name__ == "__main__":
    zarr_folder = '/home/ml-mruzicka/modifiedplanes//train/2018-09-27-10-43-39/'