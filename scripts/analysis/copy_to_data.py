### Original code from Ella Bahry @bellonet, modified by Laura Breimann

import os
import sys
import logging
import shutil
from glob import glob


##############################################################################
#                           LOGGER SETUP FUNCTION                             #
##############################################################################

def setup_logger(log_file_path=None, to_console=True):
    """
    Sets up a root logger. Optionally writes to file (log_file_path) and/or to console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplication:
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler (console)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        console_handler.setFormatter(formatter_console)
        logger.addHandler(console_handler)

    # Optional file handler
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s - %(message)s')
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    logging.info("Logger initialized.")


##############################################################################
#                           MAIN FUNCTION                                    #
##############################################################################

def run_copy_to_data(
    pipeline_dir,
    scratch_csv_path,
    dir_path_scratch_finaldata,
    dir_path_scratch_preview,
    dir_path_new_tif,
    dir_path_new_nd2,
    analysis_path,
    log_file_path=None
):
    """
    This function:
     1) Sets up logging.
     2) Copies the CSV file from scratch to analysis directory.
     3) Copies any new final TIFs, masks, previews, and median-filtered images
        from scratch to the analysis directory.
     4) Copies any original (uncropped) TIFs.
     5) Removes temporary folders containing processed TIF/ND2 files.
     6) Logs any permission errors found in the pipeline log.

    :param pipeline_dir:               Path to the main pipeline directory.
    :param scratch_csv_path:           Path to the CSV in scratch.
    :param dir_path_scratch_finaldata: Path to the temporary "finaldata" on scratch.
    :param dir_path_scratch_preview:   Path to the preview images on scratch.
    :param dir_path_new_tif:           Path to the new TIFs from scratch.
    :param dir_path_new_nd2:           Path to the new ND2s from scratch.
    :param analysis_path:              Path to the base analysis directory.
    :param log_file_path:              Path to pipeline.log file for logging (optional).
    """

    setup_logger(log_file_path=log_file_path, to_console=True)
    logging.info("\n\nStarting script copy_to_data\n *********************************************")

    # Prepare relevant paths in analysis folder
    data_csv_path = os.path.join(analysis_path, 'embryos_csv', 'embryos.csv')
    dir_path_data_finaldata = os.path.join(analysis_path, "finaldata")
    dir_path_data_preview = os.path.join(analysis_path, "preview_embryos")
    dir_path_data_tifs = os.path.join(analysis_path, "tifs")
    log_file_path = os.path.join(pipeline_dir, 'pipeline.log') if (not log_file_path) else log_file_path

    # 1) Copy CSV file
    logging.info(f"Copying CSV from {scratch_csv_path} to {data_csv_path}")
    shutil.copyfile(scratch_csv_path, data_csv_path)
    os.chmod(data_csv_path, 0o664)

    # 2) Copy final TIFs, masks, previews, etc.
    new_tifs_paths = glob(os.path.join(dir_path_scratch_finaldata, 'tifs', '*'))
    all_old_tifs_names = [os.path.basename(f) for f in glob(os.path.join(dir_path_data_finaldata, 'tifs', '*'))]
    tifs_names = [os.path.basename(f) for f in new_tifs_paths]

    for i, tif_name in enumerate(tifs_names):
        if tif_name not in all_old_tifs_names:
            scratch_tif_path = new_tifs_paths[i]
            data_tif_path = os.path.join(dir_path_data_finaldata, 'tifs', tif_name)

            # Copy TIF
            shutil.copyfile(scratch_tif_path, data_tif_path)
            os.chmod(data_tif_path, 0o664)

            # Copy mask
            scratch_mask_path = os.path.join(dir_path_scratch_finaldata, 'masks', f'{tif_name[:-4]}.mask.tif')
            data_mask_path = os.path.join(dir_path_data_finaldata, 'masks', f'{tif_name[:-4]}.mask.tif')
            shutil.copyfile(scratch_mask_path, data_mask_path)
            os.chmod(data_mask_path, 0o664)

            # Copy preview
            scratch_preview_path = os.path.join(dir_path_scratch_preview, f'{tif_name[:-4]}.png')
            data_preview_path = os.path.join(dir_path_data_preview, f'{tif_name[:-4]}.png')
            shutil.copyfile(scratch_preview_path, data_preview_path)
            os.chmod(data_preview_path, 0o664)

            # Copy median-filter files for channels c0, c1, c2
            for c in range(3):
                scratch_median_path = os.path.join(dir_path_scratch_finaldata, 'medians', f'c{c}_{tif_name}')
                data_median_path = os.path.join(dir_path_data_finaldata, 'medians', f'c{c}_{tif_name}')
                shutil.copyfile(scratch_median_path, data_median_path)
                os.chmod(data_median_path, 0o664)

        else:
            logging.warning(f'{tif_name} is already in data, double name - skipping copy')

    # 3) Copy original TIFs (uncropped) from scratch
    org_tifs_scratch_paths = glob(os.path.join(dir_path_new_tif, '*'))
    all_org_old_tifs_names = [os.path.basename(f) for f in glob(os.path.join(dir_path_data_tifs, '*'))]
    tifs_names = [os.path.basename(f) for f in org_tifs_scratch_paths]

    for i, tif_name in enumerate(tifs_names):
        if tif_name not in all_org_old_tifs_names:
            new_tif_src = org_tifs_scratch_paths[i]
            new_tif_dest = os.path.join(dir_path_data_tifs, tif_name)

            shutil.copyfile(new_tif_src, new_tif_dest)
            os.chmod(new_tif_dest, 0o664)
        else:
            logging.warning(f'{tif_name} is already in data, double name')

    # 4) Cleanup: remove new TIF, ND2, and finaldata directories from scratch
    logging.info("Removing temporary directories on scratch...")
    shutil.rmtree(dir_path_new_tif, ignore_errors=True)
    shutil.rmtree(dir_path_new_nd2, ignore_errors=True)
    shutil.rmtree(dir_path_scratch_finaldata, ignore_errors=True)

    # NOTE: If you want to do a 'git add/commit/push' on the analysis path,
    # you could do that here (requires a Git Python library or subprocess calls).
    # e.g.
    # from git import Repo
    # repo = Repo(analysis_path)
    # repo.git.add(A=True)
    # repo.index.commit("Update with new TIFs, masks, previews, etc.")
    # origin = repo.remotes.origin
    # origin.push()

    # 5) Check for permission errors in pipeline log
    if os.path.exists(log_file_path):
        with open(log_file_path,'r') as f:
            curr_run_log = f.read().split('Starting script copy_to_data')[-1].split('\n')
        permission_errors = [l.split("<class")[0] for l in curr_run_log if "Permission" in l]
        if len(permission_errors) > 0:
            nl = '\n'
            logging.warning(f'AY YAY YAY, permission errors: \n {nl.join(permission_errors)}')

    logging.info("Finished script, yay!\n ********************************************************************")
