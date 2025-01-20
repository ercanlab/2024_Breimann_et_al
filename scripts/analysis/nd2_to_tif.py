import os
import sys
import logging
import pandas as pd
import numpy as np
from glob import glob
from pims import ND2_Reader
from skimage import io
import tifffile as tif

##############################################################################
#                              HELPER FUNCTIONS                              #
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

    # File handler
    if log_file_path is not None:
        handler_file = logging.FileHandler(log_file_path)
        handler_file.setLevel(logging.DEBUG)
        formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        handler_file.setFormatter(formatter_file)
        logger.addHandler(handler_file)

    logging.info("Logger initialized.")


def find_latest_in_condition(filenames_in_csv, condition):
    """
    Given a list of filenames in the CSV and a condition string,
    returns the highest numeric value used so far in that condition.
    """
    condition_existing_files = [f for f in filenames_in_csv if f.startswith(condition)]
    if not condition_existing_files:
        return 0
    as_num = [int(f.split('_')[-1].split('.')[0]) for f in condition_existing_files]
    condition_max = max(as_num)
    return condition_max


def get_channels_info(meta):
    """
    Extract channel names and emission wavelengths from ND2 metadata.
    Pads with empty strings if <5 channels found.
    """
    # Channels come from meta keys like 'plane_0', 'plane_1', etc.
    channels_info = [value for key, value in meta.items() if "plane_" in key and isinstance(value, dict)]
    channels_names = [v["name"].lower() for v in channels_info]
    channels_emission_n = [v["emission_nm"] for v in channels_info]

    n_missing_channels = 5 - len(channels_info)
    if n_missing_channels > 0:
        channels_names.extend(["" for _ in range(n_missing_channels)])
        channels_emission_n.extend(["" for _ in range(n_missing_channels)])

    return channels_names, channels_emission_n


def get_channel_num(channels_names, channel_name):
    """
    Returns the index of channel_name in channels_names (case-sensitive).
    Returns -1 if not found.
    """
    if channel_name in channels_names:
        return channels_names.index(channel_name)
    return -1


def make_maxproj(frame, new_name, channel_num, output_path, save_im=True):
    """
    Produces a maximum projection of the Z-stack for the given channel
    and optionally saves it as a TIF image in `output_path` with `new_name`.
    """
    im_channel = frame[:, channel_num, :, :]
    im_max = np.max(im_channel, axis=0)

    if save_im:
        outpath = os.path.join(output_path, new_name)
        io.imsave(outpath, im_max)
        os.chmod(outpath, 0o664)

    return im_max


def fill_additional_df_cols(csv_file, new_filenames, channels_names, channels_emission_n,
                            dapi_num, gfp_num, z_size, nd2_basename):
    """
    Updates columns in the CSV with channel info, z-size, is_male_batch, etc.
    """
    idxs = list(csv_file[csv_file["filename"].isin(new_filenames)].index)

    if len(idxs) != len(new_filenames):
        logging.warning(f'{nd2_basename} - mismatch in number of CSV rows vs. new filenames. Check for duplicates.')
    else:
        # Attempt to standardize channel name formatting
        formatted_channel_names = ["DAPI", "GFP", "mCherry", "Cy5", "GoldFISH"]
        # Re-map each channel to the "canonical" name if possible:
        # e.g. if channels_names[i] contains "dapi", rename it to "DAPI"
        corrected_channels = []
        for c in channels_names:
            matched = None
            c_lower = c.lower()
            for fc in formatted_channel_names:
                if fc.lower() in c_lower:
                    matched = fc
                    break
            if matched is not None:
                corrected_channels.append(matched)
            else:
                corrected_channels.append(c)

        # Update CSV
        for i in range(len(channels_names)):
            csv_file.loc[idxs, f'c{i}'] = corrected_channels[i]
            csv_file.loc[idxs, f'c{i}_lambda'] = channels_emission_n[i]

        # Fill other columns
        csv_file.loc[idxs, "#channels"] = len([c for c in corrected_channels if c != ""])
        csv_file.loc[idxs, "DAPI channel"] = dapi_num
        csv_file.loc[idxs, "GFP channel"] = gfp_num
        csv_file.loc[idxs, "num_z_planes"] = z_size
        csv_file.loc[idxs, "is_male_batch"] = 1 if "male" in nd2_basename.lower() else 0

    return csv_file


def readND2_saveTIFF(
    images, 
    output_path,
    dir_path_maxp_gfp,
    csv_file,
    failing_nd2_list_file,
    csv_path
):
    """
    Reads an ND2 file, converts it to .tif (including max-proj for the GFP channel),
    appends metadata rows to the CSV, and handles exceptions/logging.
    """
    from pims import ND2_Reader  # to ensure ND2_Reader is locally available if needed

    new_filenames = []
    nd2_basename = os.path.basename(images)

    try:
        with ND2_Reader(images) as frames:
            # Setup axes
            if 'm' in frames.axes:
                frames.iter_axes = 'm'
            frames.bundle_axes = 'zcyx'
            meta = frames.metadata

            # Clean any unusual chars in metadata
            if 'objective' in meta and 'λ' in meta['objective']:
                meta['objective'] = meta['objective'].replace("λ", "lambda")

            # Get channel info
            channels_names, channels_emission_n = get_channels_info(meta)

            # skip if fewer than 4 channels or something unexpected
            if channels_names[3] == "" or "5" in channels_names:
                logging.info(f'Skipping file (missing channels) - {images}')
                with open(failing_nd2_list_file, "a+") as f:
                    f.write(f'{nd2_basename}\n')
                return csv_file

            # Indices for DAPI and GFP
            dapi_num = get_channel_num(channels_names, 'dapi_andor')
            gfp_num = get_channel_num(channels_names, 'gfp_andor')

            if dapi_num == -1 or gfp_num == -1:
                logging.info(f'Skipping file (no dapi/gfp channel) - {images}')
                with open(failing_nd2_list_file, "a+") as f:
                    f.write(f'{nd2_basename}\n')
                return csv_file

            # Derive "condition" from filename (example logic below, adjust as needed)
            if "rnai" in nd2_basename.lower():
                # e.g. Something like: "mySample_RNAi_something"
                # Start indexing from something like "RNAi_[...]" 
                # Adjust the slice accordingly to your naming convention
                condition = f"RNAi_{nd2_basename.split('_')[2][5:]}"
            else:
                # Default: second token = condition
                # e.g. "prefix_CONDITION_something.nd2"
                condition = nd2_basename.split("_")[1].upper()

            # Convert all frames in the ND2
            for i, frame in enumerate(frames):
                condition_max = find_latest_in_condition(csv_file["filename"].dropna().tolist(), condition)
                new_name = f'{condition}_{condition_max+1}'

                # If shape is (z, c, y, x) but channel <-> y mismatch, adjust
                if frame.shape[1] > frame.shape[3]:
                    frame = np.swapaxes(frame, 1, 3)

                # Save the entire stack as TIF
                out_stack_name = f'{new_name}.tif'
                full_out_stack_path = os.path.join(output_path, out_stack_name)
                tif.imsave(full_out_stack_path, frame, imagej=True, metadata=meta)
                os.chmod(full_out_stack_path, 0o664)

                # Make a note of the original filename (ND2 + series)
                original_filename = f'{nd2_basename[:-4]} series{i+1}'

                # Create a new row in the CSV
                new_row = {
                    "original filename": [original_filename],
                    "filename": [new_name]
                }

                # Make GFP max proj and save
                make_maxproj(frame, out_stack_name, gfp_num, dir_path_maxp_gfp)

                # Concatenate to CSV
                df = pd.DataFrame(new_row)
                csv_file = pd.concat([csv_file, df], ignore_index=True)

                new_filenames.append(new_name)
                z_size = frame.shape[0]

                logging.info(f'Success nd2_to_tif {original_filename}')

        # After finishing reading all frames
        frames.close()

        # Remove the ND2 file to save space
        os.remove(images)

        # Fill additional columns in CSV
        csv_file = fill_additional_df_cols(
            csv_file, 
            new_filenames, 
            channels_names, 
            channels_emission_n, 
            dapi_num, 
            gfp_num, 
            z_size, 
            nd2_basename
        )

        # Update CSV on disk
        csv_file.to_csv(csv_path, index=False)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.warning(f'\nException ND2_to_tif {images}\n{e}')
        logging.warning(f'{exc_type} {exc_tb.tb_lineno}\n')

        with open(failing_nd2_list_file, "a+") as f:
            f.write(f'{nd2_basename}\n')

    return csv_file


##############################################################################
#                              MAIN PIPELINE                                 #
##############################################################################

def run_nd2_to_tif(
    pipeline_dir,
    log_file_path,
    channels_csv_path,
    failing_nd2_list_file,
    csv_path,
    dir_path_nd2,
    dir_path_tif,
    dir_path_maxp_gfp
):
    """
    Main driver function that ties together the ND2 -> TIF pipeline.
    
    :param pipeline_dir:             Base directory of the pipeline (for reference)
    :param log_file_path:            Path to the .log file
    :param channels_csv_path:        Path to the channel-to-type CSV
    :param failing_nd2_list_file:    Path to the 'failing' ND2 log
    :param csv_path:                 Path to the main `embryos.csv`
    :param dir_path_nd2:             Directory containing ND2 files
    :param dir_path_tif:             Directory for output TIF stacks
    :param dir_path_maxp_gfp:        Directory for output max-projected GFP TIFs
    """
    ############################################################################
    #  1. Setup logger
    ############################################################################
    setup_logger(log_file_path=log_file_path, to_console=True)
    logging.info("\n\nStarting script nd2_to_tif\n *********************************************")

    ############################################################################
    #  2. Find ND2 files & load CSV
    ############################################################################
    nd2_files = glob(os.path.join(dir_path_nd2, "*"))
    csv_file = pd.read_csv(csv_path)
    
    ############################################################################
    #  3. Process each ND2 file
    ############################################################################
    for f in nd2_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        if size_mb > 1:
            csv_file = readND2_saveTIFF(
                images=f,
                output_path=dir_path_tif,
                dir_path_maxp_gfp=dir_path_maxp_gfp,
                csv_file=csv_file,
                failing_nd2_list_file=failing_nd2_list_file,
                csv_path=csv_path
            )

    logging.info('Finished ND2 -> TIF conversion step')

    ############################################################################
    #  4. Fill default values in CSV 
    ############################################################################
    csv_file = csv_file.fillna(value={
        'c3_lambda': -1,
        'c4_lambda': -1, 
        '#c0_smfish': -1,
        '#c1_smfish': -1,
        '#c2_smfish': -1, 
        'c0_saturation': -1,
        'c1_saturation': -1,
        'c2_saturation': -1,
        'c3_saturation': -1,
        'c4_saturation': -1, 
        '#c0_smfish_adj': -1,
        '#c1_smfish_adj': -1,
        '#c2_smfish_adj': -1,
        'crop_offset_x': -1,
        'crop_offset_y': -1,
        'is_male_batch': 0,
        'is_male': -1,
        'is_z_cropped': -1,
        'num_z_planes': -1,
        'is_too_bleached': -1,
        'tx': -1,
        'signal': -1,
        'is_valid_final': -1,
        'is_dapi_stack': -1,
        'status': -1,
        'first_slice': -1,
        'last_slice': -1,
        'stage_bin': -1,
        'predicted_bin': -1,
        'bin_confidence_count': -1,
        'bin_confidence_mean': -1,
        'unique_id': -1,
    })
    csv_file.to_csv(csv_path, index=False)
    logging.info('Added default values to CSV (if new data was processed)')

    ############################################################################
    #  5. Fill channel type info 
    ############################################################################
    if os.path.exists(channels_csv_path):
        df_channel_type = pd.read_csv(channels_csv_path)
        # Convert to dict
        channels_types_dict = df_channel_type.groupby('c')["c_type"].apply(list).to_dict()

        # For columns c{i}_type, fill if empty
        for i in range(3):
            ctype_col = f'c{i}_type'
            if ctype_col not in csv_file.columns:
                csv_file[ctype_col] = ""
            
            # Indices where c{i}_type is NaN or empty
            empties_mask = csv_file[ctype_col].isna() | (csv_file[ctype_col] == "")
            empties_in_c_types = csv_file.index[empties_mask]

            for idx in empties_in_c_types:
                c_name = csv_file.at[idx, f'c{i}']
                if pd.isna(c_name) or c_name == "":
                    continue

                # possible channel types for that channel
                if c_name in channels_types_dict:
                    possible_types = channels_types_dict[c_name]
                    # glean from original filename if possible
                    original_fn_tokens = str(csv_file.at[idx,"original filename"]).split("_")[2:-1]
                    # see if exactly one type from possible_types appears in the tokens
                    found_types = [token for token in original_fn_tokens if token in possible_types]
                    
                    if len(found_types) > 1:
                        logging.warning(
                            f'Row {idx}: filename {csv_file.at[idx,"original filename"]} '
                            f'has more than one possible channel type ({found_types}). '
                            'Leaving cell empty.'
                        )
                    elif len(found_types) == 1:
                        csv_file.at[idx, ctype_col] = found_types[0]
                    else:
                        # no single match found
                        csv_file.at[idx, ctype_col] = ""
    else:
        logging.warning(f"No channel type CSV found at {channels_csv_path}; skipping channel-type assignment.")
    
    logging.info('Added channel types to CSV (if new data was processed)')

    ############################################################################
    #  6. Correct lambda if needed
    ############################################################################
    # For example, if emission lambda is 0 or 590, correct them:
    for i in range(5):
        c_name_col = f'c{i}'
        c_lambda_col = f'c{i}_lambda'
        if c_name_col not in csv_file.columns or c_lambda_col not in csv_file.columns:
            continue

        for idx, row in csv_file.iterrows():
            if row[c_lambda_col] in [0, 590]:
                if row[c_name_col] == "DAPI":
                    csv_file.at[idx, c_lambda_col] = 405
                elif row[c_name_col] == "GFP":
                    csv_file.at[idx, c_lambda_col] = 488
                elif row[c_name_col] == "mCherry":
                    csv_file.at[idx, c_lambda_col] = 610

    # Save the CSV
    csv_file = csv_file.reset_index(drop=True)
    csv_file.to_csv(csv_path, index=False)
    # Adjust permission
    os.chmod(csv_path, 0o664)

    logging.info('Corrected lambda values (if needed)')

    logging.info("Finished script, yay!\n ********************************************************************")

