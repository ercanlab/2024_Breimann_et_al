import os
import sys
import logging
from glob import glob
import numpy as np
import tifffile as tif
import pandas as pd
from shutil import rmtree

from skimage.transform import resize 
from stardist.models import StarDist2D


##############################################################################
#                        LOGGER SETUP FUNCTION                               #
##############################################################################

def setup_logger(log_file_path=None, to_console=True):
    """
    Sets up a root logger. Optionally writes to file (log_file_path) and/or to console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers (to avoid duplication if run multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler to console
    if to_console:
        handler_console = logging.StreamHandler(sys.stdout)
        handler_console.setLevel(logging.DEBUG)
        formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler_console.setFormatter(formatter_console)
        logger.addHandler(handler_console)

    # Optional file handler
    if log_file_path is not None:
        handler_file = logging.FileHandler(log_file_path)
        handler_file.setLevel(logging.DEBUG)
        formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler_file.setFormatter(formatter_file)
        logger.addHandler(handler_file)

    logging.info("Logger initialized.")


##############################################################################
#                         HELPER FUNCTION: resize_data                       #
##############################################################################

def resize_data(data, img_size, anti_aliasing=True, order=1):
    """
    Resizes a stack of images (data) to (img_size, img_size) using skimage.transform.resize.
    :param data:         3D numpy array of shape (N, Y, X)
    :param img_size:     Desired height/width to resize to
    :param anti_aliasing:Whether to apply anti-aliasing filter
    :param order:        The order of the spline interpolation (0=nearest, 1=bilinear, etc.)
    :return:             A 3D numpy array of shape (N, img_size, img_size)
    """
    data_rescaled = np.zeros((data.shape[0], img_size, img_size), dtype=np.float32)

    for i, im in enumerate(data):
        im_resized = resize(im, (img_size, img_size), 
                            anti_aliasing=anti_aliasing, 
                            mode='constant', 
                            order=order)
        data_rescaled[i] = im_resized
        
    return data_rescaled


##############################################################################
#                         MAIN FUNCTION: run_stardist_predict                #
##############################################################################

def run_stardist_predict(
    dir_path_maxp_gfp,
    predicted_npz_path,
    log_file_path=None,
    cleanup=True,
    img_size_down=512,
    img_size_up=1024,
    stardist_model_name='stardist',
    stardist_model_basedir=''
):
    """
    Driver function that:
      1) Finds all max-projected GFP images in `dir_path_maxp_gfp`.
      2) Loads and normalizes them.
      3) Downscales them to `img_size_down`, runs StarDist2D prediction, 
         and then upscales the label masks back to `img_size_up`.
      4) Saves results in a .npz file (`predicted_npz_path`).
      5) Optionally removes `dir_path_maxp_gfp` after prediction (cleanup).
      6) Logs permission errors (if any) to the log file.

    :param dir_path_maxp_gfp:       Path to directory containing max-projection GFP images (TIF).
    :param predicted_npz_path:      Path to output NPZ file containing predicted masks & corresponding names.
    :param log_file_path:           Path to log file (optional). If None, logs only to console.
    :param cleanup:                 If True, remove the `dir_path_maxp_gfp` after prediction.
    :param img_size_down:           The intermediate size to downscale to before prediction.
    :param img_size_up:             The final size to resize back up to after prediction.
    :param stardist_model_name:     Name of the StarDist model (default 'stardist').
    :param stardist_model_basedir:  Base directory for the StarDist model (default '').
    """
    # 1) Initialize logger
    setup_logger(log_file_path=log_file_path, to_console=True)
    logging.info("\n\nStarting script stardist_predict\n *********************************************")

    # 2) Get all images for prediction
    gfp_images_paths = glob(os.path.join(dir_path_maxp_gfp, "*"))
    gfp_images_names = [os.path.basename(p)[:-4] for p in gfp_images_paths]

    if not gfp_images_names:
        logging.exception("No new embryos (GFP images) for stardist to predict.")
        return  # or sys.exit(1) if you prefer to exit the entire interpreter

    # 3) Load images into memory
    logging.info(f"Found {len(gfp_images_names)} images to predict.")
    gfp_images = [tif.imread(p) for p in gfp_images_paths]

    # 4) Normalize each image
    X = np.asarray(gfp_images, dtype=np.float32)
    X[X == 0] = np.nan

    for i, im in enumerate(X):
        min_val, max_val = np.nanmin(im), np.nanmax(im)
        normed_im = (im - min_val) / (max_val - min_val) if max_val != min_val else im
        X[i] = normed_im

    # Replace any NaNs with 0 after normalization
    X = np.nan_to_num(X)

    # 5) Downscale images
    logging.info(f"Resizing images down to {img_size_down}x{img_size_down} for StarDist prediction.")
    X_down = resize_data(X, img_size_down, anti_aliasing=True, order=1)

    # 6) Load a StarDist2D model (with user-specified name/basedir)
    logging.info(f"Initializing StarDist2D model (name={stardist_model_name}, basedir={stardist_model_basedir}).")
    model = StarDist2D(None, name=stardist_model_name, basedir=stardist_model_basedir)

    # 7) Run the model to predict instance segmentation
    logging.info("Starting mask predictions...")
    Y_down = []
    for x_img in X_down:
        y_inst, details = model.predict_instances(x_img)
        Y_down.append(y_inst)

    Y_down = np.asarray(Y_down, dtype=np.float32)

    # 8) Upscale predicted labels back to original size
    logging.info(f"Resizing predicted label masks back up to {img_size_up}x{img_size_up}.")
    Y_up = resize_data(Y_down, img_size_up, anti_aliasing=False, order=0)  # order=0 for nearest-neighbor on labels

    # 9) Save the predictions (labels) and image names
    logging.info(f"Saving masks and filenames to {predicted_npz_path}.")
    np.savez(predicted_npz_path, labels=Y_up, names=np.array(gfp_images_names))
    os.chmod(predicted_npz_path, 0o664)

    # 10) Optionally remove the dir of the GFP images
    if cleanup:
        logging.info(f"Removing directory {dir_path_maxp_gfp}...")
        rmtree(dir_path_maxp_gfp)

    # 11) Check for permission errors in the log
    if log_file_path and os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            curr_run_log = f.read().split('Starting script stardist_predict')[-1].split('\n')
        permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
        if len(permission_errors) > 0:
            nl = '\n'
            logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

    logging.info("Finished script, yay!\n ********************************************************************")
