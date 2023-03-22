import glob
import logging
import os
import subprocess

from rtpt import RTPT

from DeepCrazyhouse.src.preprocessing.annotate_for_uncertainty_2016 import load_pgn_dataset, analyze_fen, \
    planes_to_board, put, get
from engine.src.rl.binaryio import BinaryIO
from engine.src.rl.fileio import FileIO


def change_binary_name(binary_dir: str, current_binary_name: str, process_name: str, nn_update_idx: int):
    """
    Change the name of the binary to the process' name (which includes initials,
    binary name and remaining time) & additionally add the current epoch.

    :return: the new binary name
    """
    idx = process_name.find(f'#')
    new_binary_name = f'{process_name[:idx]}_UP={nn_update_idx}{process_name[idx:]}'

    if not os.path.exists(binary_dir + new_binary_name):
        os.rename(binary_dir + current_binary_name, binary_dir + new_binary_name)
        logging.info(f'Changed binary name from {current_binary_name} to: {new_binary_name}')

    return new_binary_name



if __name__ == "__main__":
    rtpt = RTPT(name_initials='MR', experiment_name='AnnotateCrazyhouse_17', max_iterations=196)
    rtpt.start()
    engine_init = subprocess.Popen(
        '/root/CrazyAra/CrazyAra',#C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/CrazyAra.exe', #root/CrazyAra/CrazyAra',
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    put('setoption name First_Device_ID value 10', engine_init)
    put('\n', engine_init)
    put('setoption name Last_Device_ID value 10', engine_init)
    put('\n', engine_init)
    get(engine_init)
    file_io = FileIO(orig_binary_name='CrazyAra', binary_dir='/root/CrazyAra/',
                     uci_variant='crazyhouse', framework='pytorch')
    binary_io = None
    current_binary_name = 'CrazyAra'
    zarr_filepaths = glob.glob("/home/ml-mruzicka/planes/train/**/*2016*.zip")
    idx = 0
    for filepath in zarr_filepaths:
        start_indices, planes, x_value, y_value, y_policy, _ = load_pgn_dataset(filepath, 0, True, False, 0)
        planes = planes[:50]
        for plane in planes:
            eval_search, eval_init = analyze_fen(planes_to_board(planes=plane))
        rtpt.step()
        current_binary_name = change_binary_name(file_io.binary_dir, current_binary_name,
                                                 rtpt._get_title(), idx)
        binary_io = BinaryIO(binary_path=file_io.binary_dir + current_binary_name)
        idx += 1
