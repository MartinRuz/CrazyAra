import glob

import numpy as np
import zarr

from DeepCrazyhouse.src.domain.util import get_numpy_arrays
from DeepCrazyhouse.src.preprocessing.annotate_for_uncertainty_2016_1 import planes_to_board


def load_and_compare():
    zarr_filepath = []
    zarr_filepath = '/home/ml-mruzicka/modifiedplanes/train/2018-09-27-10-43-39/lichess_db_crazyhouse_rated_2018-07_9.zip'
    print(zarr_filepath)
    store = zarr.ZipStore(zarr_filepath, mode="r")
    zarr_file = zarr.group(store=store, overwrite=False)
    start_indices, x, y_value, y_policy, plys_to_end, y_best_move_q = get_numpy_arrays(zarr_file)
    eval_init = np.array(zarr_file["eval_init"])
    eval_search = np.array(zarr_file["eval_search"])
    for i in range(0,61):
        plane = planes_to_board(planes=x[i])
        fen = plane.fen()
        print(fen)
        print(eval_search[i])



if __name__ == "__main__":
    load_and_compare()