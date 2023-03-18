"""
@file: annotate_for_uncertainty.py
Created on 15.03.23
@project: DeepCrazyhouse
@author: Martin Ruzicka

TODO description
"""
import os
import subprocess
from multiprocessing import Pool
from time import time
import xarray as xr
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
from DeepCrazyhouse.src.domain.util import get_board_position_index, get_row_col, np, get_numpy_arrays, get_x_y_and_indices
import glob
import zarr
from DeepCrazyhouse.src.domain.variants.input_representation import MATRIX_NORMALIZER

import DeepCrazyhouse.src.domain.variants.classical_chess.v2.input_representation as chess_v2
import DeepCrazyhouse.src.domain.variants.classical_chess.v3.input_representation as chess_v3

from DeepCrazyhouse.src.domain.variants.constants import NB_LAST_MOVES
from DeepCrazyhouse.src.domain.variants.output_representation import move_to_policy
from DeepCrazyhouse.src.domain.variants.input_representation import board_to_planes
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
    #train_dataset = "C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/DeepCrazyhouse/src/preprocessing/lichess_db_crazyhouse_rated_2018-07_3.zip"
    #pgn_dataset = zarr.group(store=zarr.ZipStore(train_dataset, mode="r"))
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
        y_value = (1-q_value_ratio) * y_value + q_value_ratio * y_best_move_q

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
    end_board_idx = NB_CHANNELS_POS+NB_CHANNELS_CONST
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
    #print('\nyou:\n\t'+command)
    engine.stdin.write(command)

def get(engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    #print("before is ready")
    put('isready\n', engine)
    #print('\nengine:')
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        #if text !='':
        #     print('\t'+text)
        #if text.__contains__('Unknown'):
        #    print('\t'+text)
def get_eval_init(fen):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    get(engine_init)
    put('position fen ' + fen, engine_init)
    get(engine_init)
    put('go nodes 1', engine_init)
    put('isready\n', engine_init)
    #print('\nengine:')
    while True:
        text = engine_init.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            #print('\t' + text)
            txt = text.split(' ')
            idx = txt.index('value')
            result = fen, txt[idx+1]
    return result

def get_eval(fen, num_nodes, engine):
    # using the 'isready' command (engine has to answer 'readyok')
    # to indicate current last line of stdout
    #print("first get")
    get(engine)
    #print("first put")
    put('position fen ' + fen + '\n', engine)
    #print("second get")
    get(engine)
    #print("second put")
    put('go nodes ' + num_nodes + '\n', engine)
    #print("third put")
    put('isready\n', engine)
    #print('\nengine:')
    while True:
        text = engine.stdout.readline().strip()
        if text == 'readyok':
            break
        if text.__contains__('value'):
            #print('\t' + text)
            txt = text.split(' ')
            idx = txt.index('value')
            result = txt[idx+1]
    return result

def analyze_fen(game):
    eval_init = [] #used to store the eval of the net, i.e. CrazyAra without search
    eval_search = [] #used to store the eval after search with NUM_PLAYOUTS playouts
    NUM_PLAYOUTS = '800' #the number of playouts in CrazyAra that we use to get the "ground-truth"
    fen_dic = {}  # A dictionary which maps the fen description to its number of occurrences
    # Iterate through all moves (except the last one) and play them on a board.
    # you don't want to push the last move on the board because you had no evaluation to learn from in this case
    # The moves get pushed at the end of the for-loop and is only used in the next loop.
    # Therefor we can iterate over 'all' moves

    for move in game:
        fen = move.fen()
        result_search = get_eval(fen, NUM_PLAYOUTS, engine_search)
        result_init = get_eval(fen, '1', engine_init)
        eval_search.append(result_search)
        eval_init.append(result_init)
    return eval_search, eval_init

def export_pgn_batch(self, cur_part, game_idx_start, game_idx_end, pgn_sel, nb_white_wins, nb_black_wins, nb_draws):
    """
    Exports one part of the pgn-files of the current games selected.
    After the export of one part the memory can be freed of the local variables.
    If the function has been ran successfully a new dataset-partfile was created in the dataset export directory
    For loading and exporting multiprocessing is used

    :param cur_part: Current part (integer value which start at 0).
    :param game_idx_start: Starting game index of the selected game for this part
    :param game_idx_end: End game index of the current part
    :param pgn_sel: Selected PGN data which will be used for the export
    :param nb_white_wins: Number of games which white won in the current part
    :param nb_black_wins: Number of games which black won in the current part
    :param nb_draws: Number of draws in the current part
    :return:
    """
    if not self.use_all_games and self._cur_min_elo_both is None:
        raise Exception("self._cur_min_elo_both")

    # Refactoring is probably a good idea
    # Too many arguments (8/5) - Too many local variables (32/15) - Too many statements (69/50)
    params_inp = []  # create a param input list which will concatenate the pgn with it's corresponding game index
    for i, pgn in enumerate(pgn_sel):
        game_idx = game_idx_start + i
        params_inp.append((pgn, game_idx, self._mate_in_one))

    logging.info("starting conversion to planes...")
    t_s = time()
    pool = Pool()
    x_dic = {}
    y_value_dic = {}
    y_policy_dic = {}
    plys_to_end_dic = {}
    metadata_dic = {}
    eval_init_dic = {}
    eval_search_dic = {}

    if not os.path.exists(self._export_dir):
        os.makedirs(self._export_dir)
        logging.info("the dataset_export directory was created at: %s", self._export_dir)
    # create a directory of the current timestmp
    if not os.path.exists(self._timestmp_dir):
        os.makedirs(self._timestmp_dir)
    # http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
    zarr_path = self._timestmp_dir + self._pgn_name.replace(".pgn", "_" + str(cur_part) + ".zip")
    # open a dataset file and create arrays
    store = zarr.ZipStore(zarr_path, mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)
    # the games occur in random order due to multiprocessing
    # in order to keep structure we store the result in a dictionary first
    for metadata, game_idx, x, y_value, y_policy, plys_to_end, eval_init, eval_search in pool.map(get_planes_from_pgn, params_inp):
        metadata_dic[game_idx] = metadata
        x_dic[game_idx] = x
        y_value_dic[game_idx] = y_value
        y_policy_dic[game_idx] = y_policy
        plys_to_end_dic[game_idx] = plys_to_end
        eval_init_dic[game_idx] = eval_init
        eval_search_dic[game_idx] = eval_search
    pool.close()
    pool.join()
    t_e = time() - t_s
    logging.debug("elapsed time: %fs", t_e)
    logging.debug("mean time for 1 game: %f ms", t_e / self._batch_size * 1000)
    # logging.debug('approx time for whole file (nb_games: %d): %fs', self._nb_games, t_mean * self._nb_games)
    # now we can convert the dictionary to a list
    metadata = get_dic_sorted_by_key(metadata_dic)
    x = get_dic_sorted_by_key(x_dic)
    y_value = get_dic_sorted_by_key(y_value_dic)
    y_policy = get_dic_sorted_by_key(y_policy_dic)
    plys_to_end = get_dic_sorted_by_key(plys_to_end_dic)
    start_indices = np.zeros(len(x))  # create a list which describes where each game starts

    for i, x_cur in enumerate(x[:-1]):
        start_indices[i + 1] = start_indices[i] + len(x_cur)

    # next we stack the list into a numpy-array
    metadata = np.concatenate(metadata, axis=0)
    x = np.concatenate(x, axis=0)
    y_value = np.concatenate(y_value, axis=0)
    y_policy = np.concatenate(y_policy, axis=0)
    plys_to_end = np.concatenate(plys_to_end, axis=0)
    logging.debug("metadata.shape %s", metadata.shape)
    logging.debug("x.shape %s", x.shape)
    logging.debug("y_value.shape %s", y_value.shape)
    logging.debug("y_policy.shape %s", y_policy.shape)
    # Save the dataset to a file
    logging.info("saving the dataset to a file...")
    # define the compressor object
    compressor = Blosc(cname=self._compression, clevel=self._clevel, shuffle=Blosc.SHUFFLE)
    # export the metadata
    zarr_file.create_dataset(
        name="metadata",
        data=metadata,
        shape=metadata.shape,
        dtype=metadata.dtype,
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    # export the images
    zarr_file.create_dataset(
        name="x",
        data=x,
        shape=x.shape,
        dtype=np.int16,
        chunks=(128, x.shape[1], x.shape[2], x.shape[3]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    # create the label arrays and copy the labels data in them
    zarr_file.create_dataset(
        name="y_value", shape=y_value.shape, dtype=np.int16, data=y_value, synchronizer=zarr.ThreadSynchronizer()
    )
    zarr_file.create_dataset(
        name="y_policy",
        shape=y_policy.shape,
        dtype=np.int16,
        data=y_policy,
        chunks=(128, y_policy.shape[1]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    zarr_file.create_dataset(
        name="plys_to_end",
        shape=plys_to_end.shape,
        dtype=np.int16,
        data=plys_to_end,
        synchronizer=zarr.ThreadSynchronizer()
    )
    zarr_file.create_dataset(
        name="start_indices",
        shape=start_indices.shape,
        dtype=np.int32,
        data=start_indices,
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    zarr_file.create_group("/parameters")  # export the parameter settings and statistics of the file
    zarr_file.create_dataset(
        name="/parameters/pgn_name",
        shape=(1,),
        dtype="S" + str(len(self._pgn_name) + 1),
        data=[self._pgn_name.encode("ascii", "ignore")],
        compression=compressor,
    )

    zarr_file.create_dataset(
        name="/parameters/limit_nb_games",
        data=[self._limit_nb_games],
        shape=(1,),
        dtype=np.int16,
        compression=compressor,
    )
    zarr_file.create_dataset(
        name="/parameters/batch_size", shape=(1,), dtype=np.int16, data=[self._batch_size], compression=compressor
    )
    zarr_file.create_dataset(
        name="/parameters/max_nb_files",
        shape=(1,),
        dtype=np.int16,
        data=[self._max_nb_files],
        compression=compressor,
    )
    if not self.use_all_games:
        zarr_file.create_dataset(
            name="/parameters/min_elo_both",
            shape=(1,),
            dtype=np.int16,
            data=[self._cur_min_elo_both],
            compression=compressor,
        )
    if self._compression:
        zarr_file.create_dataset(
            "/parameters/compression",
            shape=(1,),
            dtype="S" + str(len(self._compression) + 1),
            data=[self._compression.encode("ascii", "ignore")],
            compression=compressor,
        )
    # https://stackoverflow.com/questions/23220513/storing-a-list-of-strings-to-a-hdf5-dataset-from-python
    ascii_list = [n.encode("ascii", "ignore") for n in self._termination_conditions]
    max_length = max(len(s) for s in self._termination_conditions)
    zarr_file.create_dataset(
        "/parameters/termination_conditions",
        shape=(1, 1),
        dtype="S" + str(max_length),
        data=ascii_list,
        compression=compressor,
    )
    zarr_file.create_group("/statistics")
    zarr_file.create_dataset(
        "/statistics/number_selected_games", shape=(1,), dtype=np.int16, data=[len(pgn_sel)], compression=compressor
    )
    zarr_file.create_dataset(
        "/statistics/game_idx_start", shape=(1,), dtype=np.int16, data=[game_idx_start], compression=compressor
    )
    zarr_file.create_dataset(
        "/statistics/game_idx_end", shape=(1,), dtype=np.int16, data=[game_idx_end], compression=compressor
    )
    zarr_file.create_dataset(
        "/statistics/white_wins", shape=(1,), dtype=np.int16, data=[nb_white_wins], compression=compressor
    )
    zarr_file.create_dataset(
        "/statistics/black_wins", shape=(1,), dtype=np.int16, data=[nb_black_wins], compression=compressor
    )
    zarr_file.create_dataset(
        "/statistics/draws", shape=(1,), dtype=np.int16, data=[nb_draws], compression=compressor
    )
    store.close()
    logging.debug("dataset was exported to: %s", zarr_path)
    return True


def add_annotation_to_store(game_idx_start, pgn_sel, results_search, results_init, dataset_type = "train"):
    """
    :params evals: 2x1000 array that has the initial and searched evaluation of 1000 games
    """
    #params_inp = []
    #for i, pgn in enumerate(pgn_sel):
    #    game_idx = game_idx_start + i
    #    params_inp.append((pgn, game_idx))

    #lists for our new folders
    t_s = time()
    eval_init_list = []
    eval_search_list = []

    if dataset_type == "train":
        export_dir = main_config["planes_train_dir"]
        zarr_filepaths = glob.glob(main_config["planes_train_dir"] + "**/*.zip")
    elif dataset_type == "val":
        export_dir = main_config["planes_val_dir"]
        zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
    elif dataset_type == "test":
        export_dir = main_config["planes_test_dir"]
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val" or "test"' % dataset_type
        )
    for zarr_filepath in zarr_filepaths:
        store = zarr.ZipStore(zarr_filepath, mode="w")
        zarr_file = zarr.group(store=store, overwrite=True)

        for init in results_init:
            eval_init_list.append(init)
        for search in results_search:
            eval_search_list.append(search)

        t_e = time() - t_s
        logging.debug("elapsed time: %fs", t_e)
        logging.debug("mean time for 1 game: %f ms", t_e / 1000)

        eval_init_list = np.concatenate(eval_init_list, axis=0)
        eval_search_list = np.concatenate(eval_search_list, axis=0)

        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

        zarr_file.create_dataset(
            name="eval_init",
            data=eval_init_list,
            shape=eval_init_list.shape,
            dtype=eval_init_list.dtype,
            chunks=(128, eval_init_list.shape[1]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
        zarr_file.create_dataset(
            name="eval_search",
            data=eval_search_list,
            shape=eval_search_list.shape,
            dtype=eval_search_list.dtype,
            chunks=(128, eval_search_list.shape[1]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
        store.close()

def zarr_test(filepath, results_search, results_init):
    t_s = time()
    eval_init_list = []
    eval_search_list = []

    export_dir = main_config["planes_train_dir"]
    zarr_filepath = filepath

    for init in results_init:
        eval_init_list.append(init)
    for search in results_search:
        eval_search_list.append(search)

    store = zarr.ZipStore(zarr_filepath, mode="a")
    zarr_file = zarr.group(store=store, overwrite=False)
    start_indices, x, y_value, y_policy, plys_to_end, y_best_move_q = get_numpy_arrays(zarr_file)  # Get the data
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    eval_search_np = np.concatenate([np.array(x) for x in eval_search_list])
    eval_init_np = np.concatenate([np.array(x) for x in eval_init_list])

    zarr_file.create_dataset(
        name="eval_init",
        data=eval_init_np,
        shape=eval_init_np.shape,
        dtype=eval_init_np.dtype,
        chunks=(eval_init_np.shape[0]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    print("done with init")
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





if __name__ == "__main__":
    engine_init = subprocess.Popen(
        'C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/CrazyAra.exe',
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    engine_search = subprocess.Popen(
        'C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/CrazyAra.exe',
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    dataset_type = "train" #change to val or test whenever necessary
    #TODO: change to main_config load
    if dataset_type == "train":
        zarr_filepaths = glob.glob("C:/Users/Martin/Documents/Uni/WS22/BA/openinvc/CrazyAra/dataset_crazyhouse/planes/train/dataset_08_03/" + "*.zip") #lichess_db_crazyhouse_rated_2018-07_0.zip" #glob.glob(main_config["planes_train_dir"] + "**/*.zip")
    elif dataset_type == "val":
        zarr_filepaths = glob.glob(main_config["planes_val_dir"] + "**/*.zip")
    elif dataset_type == "test":
        zarr_filepaths = glob.glob(main_config["planes_test_dir"] + "**/*.zip")
    elif dataset_type == "mate_in_one":
        zarr_filepaths = glob.glob(main_config["planes_mate_in_one_dir"] + "**/*.zip")
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val", "test" or "mate_in_one"' % dataset_type
        )
    ROOT = logging.getLogger()
    ROOT.setLevel(logging.INFO)
    put('setoption name Use_Raw_Network value true', engine_init)
    put('\n', engine_init)
    get(engine_init)
    for filepath in zarr_filepaths:
        i = 0
        game = []
        j = 1
        results_search = []
        results_init = []
        start_indices, planes, x_value, y_value, y_policy, _ = load_pgn_dataset(filepath, 0, True, False, 0)
        print("filepath: ")
        print(filepath)
        for plane in planes:
            game.append(planes_to_board(planes=plane))
            i += 1
            if i == start_indices[j] or i == len(planes) - 1: #TODO: second clause should capture the last game
                print(j)
                if j != len(start_indices) - 1:
                    j+=1
                eval_search, eval_init = analyze_fen(game)
                results_search.append(eval_search)
                results_init.append(eval_init)
                game = []
                #if j == 5:
                #    break
        zarr_test(filepath, results_search, results_init)
