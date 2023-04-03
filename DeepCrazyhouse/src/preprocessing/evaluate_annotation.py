import glob

import numpy as np
import os

import zarr


def calculate_difference(zarr_folder):
    # Get all zarr files in folder
    print(zarr_folder)
    zarr_files = glob.glob(zarr_folder + "*.zip")
    n = len(zarr_files)
    sum_std_dev = 0
    num_diffs = 0
    # Loop over all files
    for file in zarr_files:
        # Load data from file
        store = zarr.ZipStore(file, mode="r")
        data = zarr.group(store=store, overwrite=False)
        # Get eval_init and eval_search arrays
        eval_init = np.array(data['eval_init'])
        eval_search = np.array(data['eval_search'])

        # Calculate difference
        diff = np.abs(eval_init - eval_search)

        # Calculate standard deviation and min,max of difference
        std_dev = np.std(diff)
        min_diff = np.min(diff)
        max_diff = np.max(diff)

        sum_std_dev += std_dev
        if max_diff >= 1.0:
            num_diffs += 1
            print(max_diff)
    avg_std_dev = sum_std_dev/n
    print(avg_std_dev)
    print(n)
    print(num_diffs)

if __name__ == "__main__":
    zarr_folder = '/home/ml-mruzicka/modifiedplanes/train/2018-09-27-10-43-39/'
    calculate_difference(zarr_folder)