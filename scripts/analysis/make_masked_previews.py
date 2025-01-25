import os
import sys
import logging
import shutil
import numpy as np
import pandas as pd
import tifffile as tif

from skimage import io, exposure
from scipy.ndimage import median_filter

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
    if log_file_path:
        handler_file = logging.FileHandler(log_file_path)
        handler_file.setLevel(logging.DEBUG)
        formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler_file.setFormatter(formatter_file)
        logger.addHandler(handler_file)

    logging.info("Logger initialized.")


##############################################################################
#                          HELPER FUNCTIONS                                  #
##############################################################################

def make_maxproj(im, channel_num):
    """
    Make a maximum projection of the specified channel in a 4D stack (z, c, y, x).
    Returns a float64 image normalized to [0,1].
    """
    # im shape: (z, c, y, x)
    channel_stack = im[:, channel_num, :, :]
    proj = np.max(channel_stack, axis=0).astype(np.float64)
    proj = exposure.rescale_intensity(proj, out_range=(0, 1))
    return proj

def make_meanproj(im, channel_num):
    """
    Make a mean projection of the specified channel in a 4D stack (z, c, y, x).
    Returns a float64 image normalized to [0,1] with intensity percentile cut.
    """
    # im shape: (z, c, y, x)
    channel_stack = im[:, channel_num, :, :]
    proj = np.mean(channel_stack, axis=0)

    # Clip intensities using percentiles
    mi, ma = np.percentile(proj, (40, 100))
    proj = exposure.rescale_intensity(proj, in_range=(mi, ma), out_range=(0,1))
    return proj

def make_medians(im, embryo_name, filter_size, dir_path_finaldata):
    """
    For a 4D stack (z, c, y, x) with c ~ 3..5, apply a median filter (size=filter_size)
    to each slice in the first three channels, then save the (c_im - med_c_im) result.
    """
    if im.ndim == 4 and im.shape[0] > 40 and 3 < im.shape[1] < 6:
        for ci in range(3):
            c_im = im[:, ci].astype(np.int16)
            med_c_im = np.zeros_like(c_im)

            for z_idx, slice_2d in enumerate(c_im):
                med_c_im[z_idx] = median_filter(slice_2d, size=filter_size, mode='nearest')

            final_im = c_im - med_c_im

            out_median_path = os.path.join(dir_path_finaldata, 'medians', f'c{ci}_{embryo_name}')
            tif.imsave(out_median_path, final_im)
            os.chmod(out_median_path, 0o664)

def make_final_tifs_and_preview(
    im_df, 
    dir_path_tif, 
    dir_path_finaldata,
    mask_full_im, 
    unique_labels, 
    dir_preview, 
    dir_dapi, 
    pipeline_dir,
    filter_size=19
):
    """
    For each row in im_df, create:
      1. Cropped embryo stack in 'finaldata/tifs'
      2. Cropped mask in 'finaldata/masks'
      3. DAPI stack in `dir_dapi`
      4. A preview .png in `dir_preview`
      5. Per-slice median subtracted images in 'finaldata/medians'
    
    Returns:
      is_dapi_stack (int): 1 if the first voxel in the DAPI channel is non-zero, else 0.
    """
    # Load the original full 4D TIFF stack
    filename = im_df.at[0, "filename"]
    full_tif_path = os.path.join(dir_path_tif, f'{filename}.tif')
    if not os.path.exists(full_tif_path):
        logging.warning(f"Could not find TIFF file for {filename}, skipping.")
        return 0

    im = tif.imread(full_tif_path)

    # Heuristic to check if there's actually DAPI signal
    dapi_channel_idx = int(im_df.at[0, 'DAPI channel'])
    is_dapi_stack = 0
    if im[0, dapi_channel_idx, 0, 0] != 0:
        is_dapi_stack = 1

    for i, idx in enumerate(im_df.index):
        # Coordinates to crop
        ellipse_str = im_df.at[idx, "ellipse"]
        if not isinstance(ellipse_str, str) or "Crop_end_coords--" not in ellipse_str:
            continue

        end_coords = list(map(int, ellipse_str.split('--')[1:]))
        y0 = int(im_df.at[idx, "crop_offset_y"])
        y1 = end_coords[0]
        x0 = int(im_df.at[idx, "crop_offset_x"])
        x1 = end_coords[1]

        # Crop the 4D stack
        embryo_tif = im[:, :, y0:y1, x0:x1]
        embryo_name = im_df.at[idx, "cropped_image_file"]

        # Save cropped 4D TIFF
        out_tif_path = os.path.join(dir_path_finaldata, 'tifs', embryo_name)
        tif.imsave(out_tif_path, embryo_tif)
        os.chmod(out_tif_path, 0o664)

        # Create mask for just this embryo
        embryo_mask = mask_full_im[y0:y1, x0:x1].copy()
        embryo_mask[embryo_mask == unique_labels[i]] = 255
        embryo_mask[embryo_mask != 255] = 0
        embryo_mask = embryo_mask.astype(np.uint8)

        # Save the cropped mask
        mask_filename = im_df.at[idx, "cropped_mask_file"]
        out_mask_path = os.path.join(dir_path_finaldata, 'masks', mask_filename)
        tif.imsave(out_mask_path, embryo_mask)
        os.chmod(out_mask_path, 0o664)

        # Also copy the mask to pipeline_dir/masks
        scratch_mask_dir = os.path.join(pipeline_dir, 'masks')
        os.makedirs(scratch_mask_dir, exist_ok=True)
        scratch_mask_path = os.path.join(scratch_mask_dir, mask_filename)
        shutil.copyfile(out_mask_path, scratch_mask_path)
        os.chmod(scratch_mask_path, 0o664)

        # Save DAPI stack in dir_dapi
        dapi_stack_path = os.path.join(dir_dapi, embryo_name)
        dapi_stack = embryo_tif[:, dapi_channel_idx]
        io.imsave(dapi_stack_path, dapi_stack)
        os.chmod(dapi_stack_path, 0o664)

        # Create preview
        dapi_im = make_maxproj(embryo_tif, dapi_channel_idx)
        fish_im0 = make_meanproj(embryo_tif, 0)
        fish_im2 = make_meanproj(embryo_tif, 2)

        preview_im = np.zeros((fish_im0.shape[0]*2, fish_im0.shape[1]*2), dtype=np.float32)
        preview_uint8 = (preview_im * 255).astype(np.uint8)
        preview_uint8[:fish_im0.shape[0], :fish_im0.shape[1]] = dapi_im
        preview_uint8[:fish_im0.shape[0], fish_im0.shape[1]:] = embryo_mask #/ 255.0
        preview_uint8[fish_im0.shape[0]:, :fish_im0.shape[1]] = fish_im0
        preview_uint8[fish_im0.shape[0]:, fish_im0.shape[1]:] = fish_im2

        # Save preview .png
        preview_png_path = os.path.join(dir_preview, f'{embryo_name[:-4]}.png')
        io.imsave(preview_png_path, preview_uint8)
        os.chmod(preview_png_path, 0o664)

        # Make median-filtered subtractions in finaldata/medians
        make_medians(embryo_tif, embryo_name, filter_size=filter_size, dir_path_finaldata=dir_path_finaldata)

    return is_dapi_stack


##############################################################################
#                 MAIN FUNCTION: run_make_masked_embryos_and_previews        #
##############################################################################

def run_make_masked_embryos_and_previews(
    pipeline_dir,
    csv_path,
    dir_path_tif,
    dir_path_finaldata,
    dir_dapi,
    dir_preview,
    predicted_npz_path,
    log_file_path=None,
    filter_size=19,
    pad_embryo_size=40,
    exit_if_no_npz=True
):
    """
    1) Sets up logger.
    2) Creates output directories.
    3) Loads predicted mask arrays & associated names from NPZ.
    4) Finds bounding boxes for each embryo in the mask, 
       and expands them by `pad_embryo_size` if they are fully within 0..1023.
    5) Updates the CSV to duplicate rows for each embryo found.
    6) Crops & saves final TIFs, masks, and DAPI stacks. Generates previews & median filters.
    7) Cleans up or modifies CSV accordingly, saves it, and removes the NPZ.
    8) Logs permission errors if any.

    :param pipeline_dir:      Path to the main pipeline directory.
    :param csv_path:          Path to the CSV file with embryo metadata.
    :param dir_path_tif:      Directory containing the original TIF stacks.
    :param dir_path_finaldata:Directory to store final cropped data (with subfolders).
    :param dir_dapi:          Directory to store DAPI stacks.
    :param dir_preview:       Directory to store previews.
    :param predicted_npz_path:Path to the NPZ file with stardist predictions (labels & names).
    :param log_file_path:     Path to the pipeline.log file.
    :param filter_size:       Size of median filter used in `make_medians`.
    :param pad_embryo_size:   How much to pad the bounding box of each embryo.
    :param exit_if_no_npz:    If True, exit if the NPZ is not found (default).
    """
    # 1) Setup logger
    setup_logger(log_file_path=log_file_path, to_console=True)
    logging.info("\n\nStarting script make_masked_embryos_and_previews\n *********************************************")

    # 2) Create output directories (clean or ensure they exist)
    os.makedirs(dir_preview, exist_ok=True)
    if os.path.exists(dir_path_finaldata):
        shutil.rmtree(dir_path_finaldata, ignore_errors=True)
    os.makedirs(dir_path_finaldata, mode=0o777)
    os.makedirs(os.path.join(dir_path_finaldata, 'tifs'), mode=0o777)
    os.makedirs(os.path.join(dir_path_finaldata, 'masks'), mode=0o777)
    os.makedirs(os.path.join(dir_path_finaldata, 'medians'), mode=0o777)

    # 3) Load the predicted images from NPZ
    logging.info(f"Loading NPZ predictions from {predicted_npz_path}")
    if not os.path.exists(predicted_npz_path):
        msg = f"No new stardist predictions found at {predicted_npz_path}"
        if exit_if_no_npz:
            logging.exception(msg)
            sys.exit(1)
        else:
            logging.warning(msg)
            return

    npzfile = np.load(predicted_npz_path)
    labels_images = npzfile['labels']       # was npzfile['arr_0']
    gfp_images_names = list(npzfile['names'])  # was npzfile['arr_1']


    # 4) Find bounding boxes of each label
    logging.info("Finding bounding boxes for predicted label masks...")
    embryo_labels_first_Ys_Xs_last_Ys_Xs = []
    images_unique_labels = []

    for labels_im in labels_images:
        # 1: Because 0 is background
        unique_labels = np.unique(labels_im)[1:]
        embryo_labels_idxs = [np.where(labels_im==u) for u in unique_labels]
        
        im_firsts_lasts = []
        for idxs in embryo_labels_idxs:
            # If embryo touches the border => we skip for now
            # (the code below uses a filter to avoid 'cut' embryos)
            if (np.min(idxs[0]) == 0 or np.min(idxs[1]) == 0 or 
                np.max(idxs[0]) == 1023 or np.max(idxs[1]) == 1023):
                continue
            # Otherwise, expand bounding box
            y_min = max(np.min(idxs[0]) - pad_embryo_size, 0)
            x_min = max(np.min(idxs[1]) - pad_embryo_size, 0)
            y_max = min(np.max(idxs[0]) + pad_embryo_size, 1023)
            x_max = min(np.max(idxs[1]) + pad_embryo_size, 1023)
            im_firsts_lasts.append([y_min, x_min, y_max, x_max])

        embryo_labels_first_Ys_Xs_last_Ys_Xs.append(im_firsts_lasts)
        images_unique_labels.append(unique_labels)

    # 5) Update the CSV with new rows for each embryo
    logging.info("Updating CSV to add cropped embryo rows.")
    csv_file = pd.read_csv(csv_path)
    csv_file = csv_file.reset_index(drop=True)

    for i, im_name in enumerate(gfp_images_names):
        im_row = csv_file[csv_file['filename'] == im_name]
        if im_row.shape[0] != 1:
            logging.warning(f"Filename {im_name} exists in DataFrame more than once or not at all.")

        # How many embryos did we detect in this mask?
        n_embryos = len(embryo_labels_first_Ys_Xs_last_Ys_Xs[i])
        if n_embryos > 0:
            # Duplicate the row n times
            im_df = pd.concat([im_row]*n_embryos, ignore_index=True)
            im_df["status"] = 0

            for j in im_df.index:
                embryo_im_name = f"{im_name}_cropped_{j}"
                y_min, x_min, y_max, x_max = embryo_labels_first_Ys_Xs_last_Ys_Xs[i][j]

                im_df.loc[j, "cropped_image_file"] = embryo_im_name + '.tif'
                im_df.loc[j, "cropped_mask_file"] = embryo_im_name + '.mask.tif'
                im_df.loc[j, "crop_offset_y"] = y_min
                im_df.loc[j, "crop_offset_x"] = x_min
                im_df.loc[j, "ellipse"] = f'Crop_end_coords--{y_max}--{x_max}'

            # Remove the original row(s) for im_name
            idx_orig = csv_file[csv_file['filename'] == im_name].index
            csv_file.drop(idx_orig, inplace=True)
            # Append new rows
            csv_file = pd.concat([csv_file, im_df], ignore_index=True)


        else:
            # If no embryos found, set status to -2
            csv_file.loc[csv_file["filename"] == im_name, "status"] = -2

    # Re-save CSV
    csv_file = csv_file.reset_index(drop=True)
    csv_file.to_csv(csv_path, index=False)

    # 6) Crop & save final TIFs, masks, previews, etc.
    logging.info("Creating final data images and previews for each embryo row...")
    for i, im_name in enumerate(gfp_images_names):
        im_rows = csv_file[(csv_file['filename'] == im_name) & (csv_file['status'] == 0)]
        im_rows = im_rows.reset_index(drop=True)

        if im_rows.shape[0] != 0:
            # Create final TIFs & previews
            is_dapi_stack = make_final_tifs_and_preview(
                im_df=im_rows,
                dir_path_tif=dir_path_tif,
                dir_path_finaldata=dir_path_finaldata,
                mask_full_im=labels_images[i],
                unique_labels=images_unique_labels[i],
                dir_preview=dir_preview,
                dir_dapi=dir_dapi,
                pipeline_dir=pipeline_dir,
                filter_size=filter_size
            )
            csv_file.loc[csv_file['filename'] == im_name, "is_dapi_stack"] = is_dapi_stack

    # Re-save CSV with final updates
    csv_file.to_csv(csv_path, index=False)
    os.chmod(csv_path, 0o664)

    # 7) Remove the NPZ file
    logging.info(f"Removing NPZ file {predicted_npz_path}.")
    #os.remove(predicted_npz_path)

    # 8) Check for permission errors in the log
    if log_file_path and os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            curr_run_log = f.read().split('Starting script make_masked_embryos_and_previews')[-1].split('\n')
        permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
        if permission_errors:
            nl = '\n'
            logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

    logging.info("Finished script, yay!\n ********************************************************************")
