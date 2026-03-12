import pandas as pd
import os
import shutil
from datetime import datetime

def extract_stem_and_channel(filename):
    """ Converts 'c0_N2_123_cropped_0.csv' -> channel='c0', embryo_tif='N2_123_cropped_0.tif' """
    name = filename.replace('.csv', '').replace('.tif', '') # handle double extensions just in case
    channel = name[:2]   
    stem = name[3:]      
    return channel, f"{stem}.tif"

def run_database_update(master_db_path, normalized_csv_dir, tx_threshold=2.0):
    
    # 1. Create a safe backup of the database before doing anything!
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = master_db_path.replace('.csv', f'_backup_{timestamp}.csv')
    shutil.copy2(master_db_path, backup_path)
    print(f"Created safety backup: {os.path.basename(backup_path)}")

    # 2. Map columns
    col_map = {
        'c0': {'total_spots': 'channel_1_counts_total', 'total_int': 'channel_1_counts_total_adj', 
               'tss_spots': 'channel_1_counts',         'tss_int': 'sum_of_intensity_1'},
        'c1': {'total_spots': 'channel_2_counts_total', 'total_int': 'channel_2_counts_total_adj', 
               'tss_spots': 'channel_2_counts',         'tss_int': 'sum_of_intensity_2'},
        'c2': {'total_spots': 'channel_3_counts_total', 'total_int': 'channel_3_counts_total_adj', 
               'tss_spots': 'channel_3_counts',         'tss_int': 'sum_of_intensity_3'}
    }

    print(f"Loading master database: {os.path.basename(master_db_path)}")
    db = pd.read_csv(master_db_path)

    # Ensure target columns exist
    for ch_cols in col_map.values():
        for col in ch_cols.values():
            if col not in db.columns:
                db[col] = -1

    csv_files = [f for f in os.listdir(normalized_csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} normalized CSV files to process.\n")

    updated_count = 0
    skipped_existing = 0

    for file in csv_files:
        channel, embryo_tif = extract_stem_and_channel(file)
        csv_path = os.path.join(normalized_csv_dir, file)
        
        # Find embryo in DB
        idx = db.index[db['cropped_image_file'] == embryo_tif]
        if len(idx) == 0:
            continue # Do not add a row if the embryo doesn't already exist
            
        if channel not in col_map:
            continue
        c_names = col_map[channel]
        
        # Check overwrite rule: Only update if the current value is NaN, -1, 0, or -2
        current_val = db.loc[idx[0], c_names['total_spots']]
        if pd.notna(current_val) and current_val > 0:
            skipped_existing += 1
            continue
        
        # Calculate metrics
        try:
            if os.path.getsize(csv_path) == 0:
                total_spots, total_int, tss_spots, tss_int = 0, 0.0, 0, 0.0
            else:
                df_spots = pd.read_csv(csv_path)
                int_col = 'normalized_intensity' if 'normalized_intensity' in df_spots.columns else 'intensity'
                
                if int_col not in df_spots.columns or len(df_spots) == 0:
                    total_spots, total_int, tss_spots, tss_int = 0, 0.0, 0, 0.0
                else:
                    total_spots = len(df_spots)
                    total_int = df_spots[int_col].sum()
                    
                    tx_sites = df_spots[df_spots[int_col] >= tx_threshold]
                    tss_spots = len(tx_sites)
                    tss_int = tx_sites[int_col].sum()
            
            # Update DB
            db.loc[idx, c_names['total_spots']] = total_spots
            db.loc[idx, c_names['total_int']] = total_int
            db.loc[idx, c_names['tss_spots']] = tss_spots
            db.loc[idx, c_names['tss_int']] = tss_int
            
            updated_count += 1
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save OVER the original file (since we backed it up)
    db.to_csv(master_db_path, index=False)

    print("\n==========================================")
    print("✅ DATABASE UPDATE COMPLETE")
    print(f"Successfully updated: {updated_count} channel records.")
    print(f"Skipped (Already had >0 count): {skipped_existing}")