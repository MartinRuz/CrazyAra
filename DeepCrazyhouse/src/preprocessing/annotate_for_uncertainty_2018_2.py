"""
@file: annotate_for_uncertainty.py
Created on 15.03.23
@project: DeepCrazyhouse
@author: Martin Ruzicka

TODO description
"""
import os
from rtpt.rtpt import RTPT
from engine.src.rl.binaryio import BinaryIO
from engine.src.rl.fileio import FileIO
import subprocess
from multiprocessing import Pool
from time import time
from numcodecs import Blosc

from DeepCrazyhouse.src.domain.util import get_dic_sorted_by_key
from DeepCrazyhouse.src.preprocessing.pgn_converter_util import get_planes_from_pgn
import numpy as np
import logging
import chess.pgn
from chess import engine
from chess.variant import CrazyhouseBoard
from DeepCrazyhouse.src.domain.variants.constants import (
    MODE,
    NB_CHANNELS_TOTAL,
    MODES,
    VERSION,
    MODE_CRAZYHOUSE,
    MODE_LICHESS,
    MODE_CHESS,
    MODE_XIANGQI,
    POCKETS_SIZE_PIECE_TYPE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHANNEL_MAPPING_CONST,
    CHANNEL_MAPPING_POS,
    CHANNEL_MAPPING_VARIANTS,
    MAX_NB_MOVES,
    MAX_NB_NO_PROGRESS,
    MAX_NB_PRISONERS,
    NB_CHANNELS_CONST,
    NB_CHANNELS_POS,
    NB_CHANNELS_VARIANTS,
    NB_LAST_MOVES,
    NB_CHANNELS_HISTORY,
    PIECES,
    chess,
    VARIANT_MAPPING_BOARDS)
from DeepCrazyhouse.src.domain.util import get_board_position_index, get_row_col, np, get_numpy_arrays
import glob
import zarr
from DeepCrazyhouse.src.domain.variants.input_representation import MATRIX_NORMALIZER

from DeepCrazyhouse.configs.main_config import main_config


def load_pgn_dataset(
        filepath, part_id=0, verbose=True, normalize=False, q_value_ratio=0,
):
    """
    Loads one part of the pgn dataset in form of planes / multidimensional numpy array.
    It reads all files which are located either in the main_config['test_dir'] or main_config['test_dir']

    :param dataset_type: either ['train', 'test', 'mate_in_one']
    :param part_id: Decides which part of the data set will be loaded
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1
    ! Note this only supported for hist-length=1 at the moment
    :param q_value_ratio: Ratio for mixing the value return with the corresponding q-value
    For a ratio of 0 no q-value information will be used. Value must be in [0, 1]
    :return: numpy-arrays:
            start_indices - defines the index where each game starts
            x - the board representation for all games
            y_value - the game outcome (-1,0,1) for each board position
            y_policy - the movement policy for the next_move played
            plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
            pgn_datasets - the dataset file handle (you can use .tree() to show the file structure)
    """

    # load the zarr-files

    pgn_dataset = zarr.group(store=zarr.ZipStore(filepath, mode="r"))
    start_indices, x, y_value, y_policy, plys_to_end, y_best_move_q = get_numpy_arrays(pgn_dataset)  # Get the data

    if verbose:
        logging.info("STATISTICS:")
        try:
            for member in pgn_dataset["statistics"]:
                print(member, list(pgn_dataset["statistics"][member]))
        except KeyError:
            logging.warning("no statistics found")

        logging.info("PARAMETERS:")
        try:
            for member in pgn_dataset["parameters"]:
                print(member, list(pgn_dataset["parameters"][member]))
        except KeyError:
            logging.warning("no parameters found")

    if q_value_ratio != 0:
        y_value = (1 - q_value_ratio) * y_value + q_value_ratio * y_best_move_q

    if normalize:
        x = x.astype(np.float32)
        # the y-vectors need to be casted as well in order to be accepted by the network
        y_value = y_value.astype(np.float32)
        y_policy = y_policy.astype(np.float32)
        # apply rescaling using a predefined scaling constant (this makes use of vectorized operations)
        x *= MATRIX_NORMALIZER
    return start_indices, x, y_value, y_policy, plys_to_end, pgn_dataset


def planes_to_board(planes, normalized_input=False, mode=MODE_CRAZYHOUSE):
    """
    Converts a board in plane representation to the python chess board representation
    see get_planes_of_board() for input encoding description

    :param planes: Input plane representation
    :param normalized_input: True if the input has been normalized to range[0., 1.]
    :param mode: 0 - MODE_CRAZYHOUSE: Crazyhouse only specification.
                 (Visit variants.crazyhouse.input_representation for detailed documentation)
                 1 - MODE_LICHESS: Specification for all supported variants on lichess.org
                 (Visit variants.lichess.input_representation for detailed documentation)
                 2 - MODE_CHESS: Specification for chess only with chess960 support
                 (Visit variants.chess.input_representation for detailed documentation)
    :return: python chess board object
    """

    # extract the maps for the board position
    planes_pos = planes[:NB_CHANNELS_POS]
    # extract the last maps which for the constant values
    end_board_idx = NB_CHANNELS_POS + NB_CHANNELS_CONST
    planes_const = planes[NB_CHANNELS_POS:end_board_idx]

    board = CrazyhouseBoard()

    # clear the full board (the pieces will be set later)
    board.clear()

    # iterate over all piece types
    for idx, piece in enumerate(PIECES):
        # iterate over all fields and set the current piece type
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                # check if there's a piece at the current position
                if planes_pos[idx, row, col] == 1:
                    # check if the piece was promoted
                    promoted = False
                    if mode == MODE_CRAZYHOUSE or mode == MODE_LICHESS:
                        # promoted pieces are not defined in the chess plane representation
                        channel = CHANNEL_MAPPING_POS["promo"]
                        if planes_pos[channel, row, col] == 1 or planes_pos[channel + 1, row, col] == 1:
                            promoted = True

                    board.set_piece_at(
                        square=get_board_position_index(row, col),
                        piece=chess.Piece.from_symbol(piece),
                        promoted=promoted,
                    )

    # (I) Fill in the Repetition Data
    # check how often the position has already occurred in the game
    # TODO: Find a way to set this on the board state
    # -> apparently this isn't possible because it's also not available in the board uci representation

    # ch = channel_mapping['repetitions']

    # Fill in the Prisoners / Pocket Pieces
    if mode == MODE_CRAZYHOUSE or board.uci_variant == "crazyhouse":
        # iterate over all pieces except the king
        for p_type in chess.PIECE_TYPES[:-1]:
            # p_type -1 because p_type starts with 1
            channel = CHANNEL_MAPPING_POS["prisoners"] + p_type - 1

            # the full board is filled with the same value
            # it's sufficient to take only the first value
            nb_prisoners = planes_pos[channel, 0, 0]

            # add prisoners for the current player
            # the whole board is set with the same entry, we can just take the first one
            if normalized_input is True:
                nb_prisoners *= MAX_NB_PRISONERS
                nb_prisoners = int(round(nb_prisoners))

            for _ in range(nb_prisoners):
                board.pockets[chess.WHITE].add(p_type)

            # add prisoners for the opponent
            nb_prisoners = planes_pos[channel + 5, 0, 0]
            if normalized_input is True:
                nb_prisoners *= MAX_NB_PRISONERS
                nb_prisoners = int(round(nb_prisoners))

            for _ in range(nb_prisoners):
                board.pockets[chess.BLACK].add(p_type)

    # (I.5) En Passant Square
    # mark the square where an en-passant capture is possible
    channel = CHANNEL_MAPPING_POS["ep_square"]
    ep_square = np.argmax(planes_pos[channel])
    if ep_square != 0:
        # if no entry 'one' exists, index 0 will be returned
        board.ep_square = ep_square

    # (II) Constant Value Inputs
    # (II.1) Total Move Count
    channel = CHANNEL_MAPPING_CONST["total_mv_cnt"]
    total_mv_cnt = planes_const[channel, 0, 0]

    if normalized_input is True:
        total_mv_cnt *= MAX_NB_MOVES
        total_mv_cnt = int(round(total_mv_cnt))

    board.fullmove_number = total_mv_cnt

    # (II.2) Castling Rights
    channel = CHANNEL_MAPPING_CONST["castling"]

    # reset the castling_rights for initialization
    # set to 0, previously called chess.BB_VOID for chess version of 0.23.X and chess.BB_EMPTY for versions > 0.27.X
    board.castling_rights = 0

    # WHITE
    # check for King Side Castling
    # White can castle with the h1 rook

    # add castling option by modifying the castling fen
    castling_fen = ""
    # check for King Side Castling
    if planes_const[channel, 0, 0] == 1:
        castling_fen += "K"
        board.castling_rights |= chess.BB_H1
    # check for Queen Side Castling
    if planes_const[channel + 1, 0, 0] == 1:
        castling_fen += "Q"
        board.castling_rights |= chess.BB_A1

    # BLACK
    # check for King Side Castling
    if planes_const[channel + 2, 0, 0] == 1:
        castling_fen += "k"
        board.castling_rights |= chess.BB_H8
    # check for Queen Side Castling
    if planes_const[channel + 3, 0, 0] == 1:
        castling_fen += "q"
        board.castling_rights |= chess.BB_A8

    # configure the castling rights
    if castling_fen:
        board.set_castling_fen(castling_fen)

    # (II.3) No Progress Count
    channel = CHANNEL_MAPPING_CONST["no_progress_cnt"]
    no_progress_cnt = planes_const[channel, 0, 0]
    if normalized_input is True:
        no_progress_cnt *= MAX_NB_NO_PROGRESS
        no_progress_cnt = int(round(no_progress_cnt))

    board.halfmove_clock = no_progress_cnt

    # (II.4) Color
    channel = CHANNEL_MAPPING_CONST["color"]

    if planes_const[channel, 0, 0] == 1:
        board.board_turn = chess.WHITE
    else:
        board = board.mirror()
        board.board_turn = chess.BLACK

    return board


def put(command, engine):
    engine.stdin.write(command)


def get(engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    put('isready\n', engine)
    # print('\nengine:')
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break


def get_eval_init(fen):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    get(engine_init)
    put('position fen ' + fen, engine_init)
    get(engine_init)
    put('go nodes 1', engine_init)
    put('isready\n', engine_init)
    while True:
        text = engine_init.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            # print('\t' + text)
            txt = text.split(' ')
            idx = txt.index('value')
            result = fen, txt[idx + 1]
    return result


def get_eval(fen, num_nodes, engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    # print("first get")
    get(engine)
    put('position fen ' + fen + '\n', engine)
    get(engine)
    put('go nodes ' + num_nodes + '\n', engine)
    put('isready\n', engine)
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            txt = text.split(' ')
            idx = txt.index('value')
            result = txt[idx + 1]
    return result


def analyze_fen(game, engine_init, engine_search):
    eval_init = np.empty(len(game))  # used to store the eval of the net, i.e. CrazyAra without search
    eval_search = np.empty(len(game))  # used to store the eval after search with NUM_PLAYOUTS playouts
    NUM_PLAYOUTS = '400'  # the number of playouts in CrazyAra that we use to get the "ground-truth"
    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no evaluation to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefor we can iterate over 'all' moves
    i = 0
    for move in game:
        fen = move.fen()
        result_search = get_eval(fen, NUM_PLAYOUTS, engine_search)
        result_init = get_eval(fen, '1', engine_init)
        eval_search[i] = float(result_search)
        eval_init[i] = float(result_init)
        i += 1
    return eval_search, eval_init

def zarr_test(filepath, results_search, results_init):
    zarr_filepath = filepath
    store = zarr.ZipStore(zarr_filepath, mode="a")
    zarr_file = zarr.group(store=store, overwrite=False)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    eval_init_np = results_init
    eval_search_np = results_search


    zarr_file.create_dataset(
        name="eval_init",
        data=eval_init_np,
        shape=eval_init_np.shape,
        dtype=eval_init_np.dtype,
        chunks=(eval_init_np.shape[0]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    zarr_file.create_dataset(
        name="eval_search",
        data=eval_search_np,
        shape=eval_search_np.shape,
        dtype=eval_search_np.dtype,
        chunks=(eval_search_np.shape[0]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    store.close()

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
        logging.info(f'Changed binary name to: {new_binary_name}')

    return new_binary_name


if __name__ == "__main__":
    dummy = []
    dummy.extend(glob.glob(
        '/home/ml-mruzicka/failed/*2016-06*.zip'))
    max_iter = len(dummy)
    rtpt = RTPT(name_initials='MR', experiment_name='AnnotateCrazyhouse_16_6', max_iterations=max_iter)
    rtpt.start()
    current_binary_name = 'CrazyAra'
    dataset_types = ["train"]#, "val", "test", "mate_in_one"]
    for dataset_type in dataset_types:
        zarr_filepaths = []
        if dataset_type == "train":
            zarr_filepaths.extend(glob.glob(
                '/home/ml-mruzicka/failed/*2016-06*.zip'))
        elif dataset_type == "val":
            zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
        elif dataset_type == "test":
            zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
        elif dataset_type == "mate_in_one":
            zarr_filepaths = glob.glob(main_config["planes_mate_in_one_dir"] + "**/*.zip")
        idx = 0
        for filepath in zarr_filepaths:
            file_io = FileIO(orig_binary_name=current_binary_name, binary_dir='/root/CrazyAra/',
                             uci_variant='crazyhouse', framework='pytorch')
            binary_io = None
            print(f'filepath: {filepath}')
            engine_init = subprocess.Popen(
                '/root/CrazyAra/' + current_binary_name,
                # C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/CrazyAra.exe', #root/CrazyAra/CrazyAra',
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=1,
            )
            engine_search = subprocess.Popen(
                '/root/CrazyAra/' + current_binary_name,
                # C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/CrazyAra.exe', #/root/CrazyAra/CrazyAra',
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=1,
            )

            ROOT = logging.getLogger()
            ROOT.setLevel(logging.INFO)
            put('setoption name Use_Raw_Network value true', engine_init)
            put('\n', engine_init)
            put('setoption name MCTS_Solver value false', engine_init)
            put('\n', engine_init)
            put('setoption name First_Device_ID value 5', engine_init)
            put('\n', engine_init)
            put('setoption name Last_Device_ID value 5', engine_init)
            put('\n', engine_init)
            put('setoption name Model_Directory value /root/CrazyAra/model/CrazyAra/crazyhouse/', engine_init)
            put('\n', engine_init)
            get(engine_init)
            put('setoption name First_Device_ID value 5', engine_search)
            put('\n', engine_search)
            put('setoption name Last_Device_ID value 5', engine_search)
            put('\n', engine_search)
            get(engine_search)
            i = 0
            game = []
            j = 1
            results_search = np.array([])
            results_init = np.array([])
            start_indices, planes, x_value, y_value, y_policy, _ = load_pgn_dataset(filepath, 0, True, False, 0)
            for plane in planes:
                game.append(planes_to_board(planes=plane))
                i += 1
                if start_indices[j] == i or i == len(planes):
                    print(j)
                    eval_search, eval_init = analyze_fen(game, engine_init, engine_search)
                    results_search = np.append(results_search, eval_search)
                    results_init = np.append(results_init, eval_init)
                    if j != len(start_indices) - 1:
                        j += 1
                    game = []
                    # if j == 5:
                    #    break
            zarr_test(filepath, results_search, results_init)
            rtpt.step()
            current_binary_name = change_binary_name(file_io.binary_dir, current_binary_name,
                                                     rtpt._get_title(), idx)
            binary_io = BinaryIO(binary_path=file_io.binary_dir+current_binary_name)
            idx += 1
            engine_init.kill()
            engine_search.kill()
