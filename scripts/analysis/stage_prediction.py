import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import tifffile as tif

from glob import glob
from skimage.transform import resize
from keras.models import load_model
from keras import losses
from keras.preprocessing.image import ImageDataGenerator

##############################################################################
#                           LOGGER SETUP FUNCTION                             #
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
#                         HELPER FUNCTIONS                                   #
##############################################################################

def normalize_image(im):
    """
    Normalize a numpy array (any dimensionality) to [0,1] while treating zeros as NaNs.
    """
    im_float = im.astype(np.float32)
    im_float[im_float == 0] = np.nan
    min_val, max_val = np.nanmin(im_float), np.nanmax(im_float)
    if min_val == max_val:
        # avoid division-by-zero if the entire image is the same value
        return np.zeros_like(im_float)
    im_normed = (im_float - min_val) / (max_val - min_val)
    im_normed = np.nan_to_num(im_normed)
    return im_normed


def make_tiles(
    ims_names, 
    masked_cropped_20slices_dapi_normed_path, 
    embryos_normed_path,
    tile_size=128,
    n_augment_slices=5,
    batch_size=20
):
    """
    Given a list of 3D embryo DAPI stacks (already normalized), create augmented tiles 
    and save them to `embryos_normed_path` as multi-page TIFFs.

    :param ims_names: List of image filenames to process.
    :param masked_cropped_20slices_dapi_normed_path: Directory containing normalized 3D TIFFs.
    :param embryos_normed_path: Directory to save the multi-page TIFF tiles.
    :param tile_size: Size of each tile (tile_size x tile_size).
    :param n_augment_slices: How many times to draw from the ImageDataGenerator (for augmentation).
    :param batch_size: Batch size for the Keras ImageDataGenerator.
    """
    # Define data augmentation
    # data_gen_args = dict(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     rotation_range=180,
    #     shear_range=5,
    #     brightness_range=[0.85, 1],
    #     fill_mode='constant',
    #     cval=0,
    #     rescale=1./255.
    # )

    data_gen_args = dict(
    # no flips, no rotation, no shear
    horizontal_flip=False,
    vertical_flip=False,
    rotation_range=0,
    shear_range=0,
    brightness_range=None,
    fill_mode='constant',  # or reflect
    cval=0,
    rescale=1./255.
)

    datagen = ImageDataGenerator(**data_gen_args)

    for im_name in ims_names:
        out_path = os.path.join(embryos_normed_path, f'{im_name[:-4]}_tiles.tif')
        if os.path.exists(out_path):
            continue  # skip if tiles already exist

        in_path = os.path.join(masked_cropped_20slices_dapi_normed_path, im_name)
        if not os.path.exists(in_path):
            logging.warning(f"Normalized file not found for {im_name}, skipping tile creation.")
            continue

        # Load normalized 3D stack
        im_3d = tif.imread(in_path)  # shape: (Z, Y, X)
        logging.debug(f"shape of im3d {im_3d.shape}")
        im_tiles = []

        # Keras expects a 4D tensor for 2D images: (Z, Y, X, channels).
        # We'll treat each Z-slice as a separate "image".
        # shape after reshape: (Z, Y, X, 1)
        im_4d = im_3d[..., np.newaxis]  # add a channel dimension
        logging.debug(f"shape of im4d {im_4d.shape}")
        it = datagen.flow(im_4d, batch_size=batch_size)

        # We'll gather augmented slices from the generator
        for _ in range(n_augment_slices):
            batch = it.next()  # shape: (Z, Y, X, 1)

            # For each slice in this batch, subdivide into tiles
            for slice_2d in batch:
                # slice_2d shape: (Y, X, 1)
                y_size, x_size, _ = slice_2d.shape
                for i in range(math.ceil(y_size / tile_size)):
                    for j in range(math.ceil(x_size / tile_size)):
                        tile = slice_2d[
                            i * tile_size:(i + 1) * tile_size,
                            j * tile_size:(j + 1) * tile_size,
                        ]
                        # tile shape: (tile_size, tile_size, 1), if it fits fully
                        if tile.shape == (tile_size, tile_size, 1):
                            #logging.info(f"Number of tiles {len(tile)}")
                            # Keep tile if it isn't mostly zeros
                            n_zeros = np.count_nonzero(tile == 0)
                            logging.info(
                                f"Tile at i={i}, j={j}: shape={tile.shape}, n_zeros={n_zeros}, "
                                f"tile_size={tile.size}, keep={n_zeros*7 < tile.size}"
                            )

                            # E.g., discard tile if more than ~14% is zero (adjust logic as needed):
                            if n_zeros * 7 < tile.size: 
                                im_tiles.append(tile)
                                

        # Convert to numpy array, shape: (N, tile_size, tile_size, 1)
        im_tiles = np.asarray(im_tiles, dtype=np.float32)
        logging.info(f"Number of tiles at the end{len(im_tiles)}")
        if im_tiles.size > 0:
            tif.imsave(out_path, im_tiles)
            os.chmod(out_path, 0o664)


##############################################################################
#                  MAIN FUNCTION: run_stage_prediction                       #
##############################################################################

def run_stage_prediction(
    pipeline_dir,
    csv_path,
    log_file_path=None,
    stage_prediction_model_and_weights_path=None,
    new_nslices=20,
    pad_xy=40,
    tile_size=64,
    n_augment_slices=5,
    batch_size=20
):
    """
    Main driver function that:
      1) Sets up logger.
      2) Loads the CSV of embryos.
      3) Finds embryos that need stage prediction (based on certain columns).
      4) For each embryo, loads the 3D DAPI stack & mask, slices to `new_nslices` around center,
         zeroes out background according to mask, and saves the result.
      5) Normalizes the masked stack.
      6) Generates tiles from the normalized stack for model input.
      7) Runs the loaded Keras model to predict "stage bin" or "age bin" for each embryo.
      8) Writes results back to the CSV.
      9) Logs permission errors (if any) and finishes.

    :param pipeline_dir:  Path to your main pipeline directory, containing subfolders and the model.
    :param csv_path:      Path to the CSV file (embryos.csv).
    :param log_file_path: Path to the pipeline.log file (optional).
    :param stage_prediction_model_and_weights_path: Path to the Keras model for stage prediction.
    :param new_nslices:   Number of slices to keep in the center for each embryo stack (must be even).
    :param pad_xy:        How many pixels to crop from each side before slices (40 means skip 40 px border).
    :param tile_size:     Size of the tiles (e.g., 64x64).
    :param n_augment_slices: How many times to draw slices from ImageDataGenerator for augmentation.
    :param batch_size:    Batch size for the ImageDataGenerator.
    """
    # 1) Setup logger
    setup_logger(log_file_path=log_file_path, to_console=True)
    logging.info("\n\nStarting script stage_prediction\n *********************************************")

    # 2) Define needed paths
    dir_mask = os.path.join(pipeline_dir, 'masks')
    dir_dapi = os.path.join(pipeline_dir, 'dapi')
    masked_cropped_20slices_dapi_path = os.path.join(pipeline_dir, 'masked_cropped_20slices_dapi')
    masked_cropped_20slices_dapi_normed_path = os.path.join(pipeline_dir, 'masked_cropped_20slices_dapi_normed')
    embryos_normed_path = os.path.join(pipeline_dir, 'tiles_embryos')

    # Create output directories
    os.makedirs(masked_cropped_20slices_dapi_path, exist_ok=True)
    os.makedirs(masked_cropped_20slices_dapi_normed_path, exist_ok=True)
    os.makedirs(embryos_normed_path, exist_ok=True)

    # 3) Load the CSV
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file.reset_index(drop=True)

    # 4) Filter for rows/embryos that need stage prediction
    df_embryos_to_predict = csv_file[
        ((csv_file["status"] == 0) | (csv_file["status"] == 1)) 
        & (~csv_file["cropped_image_file"].isna()) 
        & (csv_file["#channels"] > 3)
        & (csv_file["predicted_bin"] == -1)
    ]
    if df_embryos_to_predict.empty:
        logging.exception("No embryos to predict stage. Exiting.")
        sys.exit(1)

    logging.info(f'Number of embryos to predict: {df_embryos_to_predict.shape[0]}')

    # Validate that corresponding dapi/mask files exist
    drop_indices = []
    for idx in df_embryos_to_predict.index:
        im_name = df_embryos_to_predict.at[idx, 'cropped_image_file']
        mask_name = df_embryos_to_predict.at[idx, 'cropped_mask_file']
        dapi_path = os.path.join(dir_dapi, im_name)
        mask_path = os.path.join(dir_mask, mask_name)
        if not os.path.exists(dapi_path):
            logging.info(f"{im_name} doesn't exist in {dir_dapi}")
            drop_indices.append(idx)
        if not os.path.exists(mask_path):
            logging.info(f"{mask_name} doesn't exist in {dir_mask}")
            drop_indices.append(idx)

    df_embryos_to_predict.drop(drop_indices, inplace=True)
    df_embryos_to_predict = df_embryos_to_predict.reset_index(drop=True)
    logging.info(f'Number of embryos to predict after checking dirs: {df_embryos_to_predict.shape[0]}')

    # 5) Create masked cropped 3D stacks with exactly `new_nslices` around the center
    for idx in df_embryos_to_predict.index:
        im_name = df_embryos_to_predict.at[idx, 'cropped_image_file']
        dapi_in_path = os.path.join(dir_dapi, im_name)
        out_path_3d = os.path.join(masked_cropped_20slices_dapi_path, im_name)

        if os.path.exists(out_path_3d):
            continue  # skip if already exists

        if not os.path.exists(dapi_in_path):
            # Should already be handled above, but just in case
            logging.warning(f"Missing DAPI stack for {im_name}, skipping.")
            continue

        # Load DAPI stack
        dapi_stack = tif.imread(dapi_in_path)  # shape: (Z, Y, X)
        # Crop XY border
        dapi_stack = dapi_stack[:, pad_xy:-pad_xy, pad_xy:-pad_xy]
        nslices = dapi_stack.shape[0]

        mid_idx = nslices // 2
        half_nslices = new_nslices // 2
        z_start = max(mid_idx - half_nslices, 0)
        z_end = min(mid_idx + half_nslices, nslices)
        dapi_stack = dapi_stack[z_start:z_end]  # shape: (new_nslices, Y-2pad_xy, X-2pad_xy)

        # Load mask
        mask_name = df_embryos_to_predict.at[idx, 'cropped_mask_file']
        mask_path = os.path.join(dir_mask, mask_name)
        mask_2d = tif.imread(mask_path)  # shape: (Y, X)
        mask_2d = mask_2d[pad_xy:-pad_xy, pad_xy:-pad_xy]

        # Zero out regions outside mask
        for z in range(dapi_stack.shape[0]):
            slice_2d = dapi_stack[z]
            slice_2d[mask_2d == 0] = 0
            dapi_stack[z] = slice_2d

        if dapi_stack.size > 0:
            # Save the new 3D stack
            tif.imsave(out_path_3d, dapi_stack)
            os.chmod(out_path_3d, 0o664)

    logging.info("Created masked DAPI images (3D stacks).")

    # 6) Normalize the 3D stacks
    ims_path = []
    embryos_stacks = []

    for idx in df_embryos_to_predict.index:
        im_name = df_embryos_to_predict.at[idx, 'cropped_image_file']
        in_3d_path = os.path.join(masked_cropped_20slices_dapi_path, im_name)
        out_normed_path = os.path.join(masked_cropped_20slices_dapi_normed_path, im_name)

        if os.path.exists(in_3d_path):
            stack_3d = tif.imread(in_3d_path)
            if stack_3d.size == 0:
                logging.info(f"Zero-size stack for {im_name}, removing.")
                os.remove(in_3d_path)
                continue

            # Normalize each voxel in the 3D stack
            normed_stack = normalize_image(stack_3d)
            tif.imsave(out_normed_path, normed_stack)
            os.chmod(out_normed_path, 0o664)

            ims_path.append(out_normed_path)
            embryos_stacks.append(normed_stack)

    logging.info("Normalized images and saved them.")

    # 7) Remove truly empty images
    to_remove = []
    for i, embryo_stack_im in enumerate(embryos_stacks):
        if embryo_stack_im.size == 0:
            to_remove.append(i)

    for idx in reversed(to_remove):
        logging.info(f"Removing empty image: {ims_path[idx]}")
        os.remove(ims_path[idx])
        del embryos_stacks[idx]
        del ims_path[idx]

    # 8) Create tiles from these normalized 3D stacks
    ims_names = [os.path.basename(p) for p in ims_path]
    make_tiles(
        ims_names=ims_names,
        masked_cropped_20slices_dapi_normed_path=masked_cropped_20slices_dapi_normed_path,
        embryos_normed_path=embryos_normed_path,
        tile_size=tile_size,
        n_augment_slices=n_augment_slices,
        batch_size=batch_size
    )
    logging.info("Created tiles and saved them.")

    # 9) Gather all tile stacks for inference
    names_final = []
    all_tiles = []

    for n in ims_names:
        tiles_tif_path = os.path.join(embryos_normed_path, f'{n[:-4]}_tiles.tif')
        if os.path.exists(tiles_tif_path) and os.path.getsize(tiles_tif_path) > 100_000:
            all_tiles.append(tif.imread(tiles_tif_path))
            names_final.append(n)

    logging.info(f"Have read tile images. Final number of embryos to predict: {len(names_final)}")

    if not names_final:
        logging.warning("No valid tile sets found for prediction.")
        return

    # 10) Load the model & predict
    if not stage_prediction_model_and_weights_path:
        logging.error("No stage prediction model provided; cannot proceed.")
        return

    model = load_model(stage_prediction_model_and_weights_path,
    custom_objects={
        'cosine_proximity': lambda y_true, y_pred: -tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    }
)
    predicted_probabilities = [model.predict(tiles) for tiles in all_tiles]
    logging.info("Ran predictions on tiles.")

    # 11) Compute majority vote for each embryo
    predicted_best_class = [np.argmax(p, axis=1) for p in predicted_probabilities]  # list of arrays
    majority_vote = [np.bincount(classes).argmax() for classes in predicted_best_class]

    # 12) Compute ratio of the majority vote (how many tiles voted for it)
    ratio_of_votes = [
        round(np.bincount(c)[majority_vote[i]] / c.size, 2) 
        for i, c in enumerate(predicted_best_class)
    ]
    # 13) Compute mean certainty
    mean_certainty = [
        round(np.mean(prob[:, majority_vote[i]]), 2)
        for i, prob in enumerate(predicted_probabilities)
    ]

    # 14) Assign results back to CSV
    for i, n in enumerate(names_final):
        csv_file.loc[csv_file["cropped_image_file"] == n, "predicted_bin"] = majority_vote[i]
        csv_file.loc[csv_file["cropped_image_file"] == n, "bin_confidence_count"] = ratio_of_votes[i]
        csv_file.loc[csv_file["cropped_image_file"] == n, "bin_confidence_mean"] = mean_certainty[i]

    # 15) Save updated CSV
    csv_file.to_csv(csv_path, index=False)
    os.chmod(csv_path, 0o664)
    logging.info("Saved new CSV with stage predictions.")

    # 16) Check for permission errors in the log
    if log_file_path and os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            curr_run_log = f.read().split('Starting script stage_prediction')[-1].split('\n')
        permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
        if permission_errors:
            nl = '\n'
            logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

    logging.info("Finished script, yay!\n ********************************************************************")
