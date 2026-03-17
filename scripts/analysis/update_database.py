import pandas as pd
import os
import shutil
import re
from datetime import datetime

def extract_stem_and_channel(filename):
    clean_name = filename.replace('.tif.csv', '').replace('.csv', '')
    channel = clean_name[:2].lower()
    stem = clean_name[3:]
    return channel, f"{stem}.tif"

def run_database_update(master_db_path, normalized_csv_dir, tx_threshold=2.0, force_overwrite=True):
    
    # 1. Create a safe backup ONLY if we are forcing overwrite (the first run)
    if force_overwrite:
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

    db = pd.read_csv(master_db_path)

    # Ensure target columns exist
    for ch_cols in col_map.values():
        for col in ch_cols.values():
            if col not in db.columns:
                db[col] = -1

    csv_files = [f for f in os.listdir(normalized_csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSVs in {os.path.basename(normalized_csv_dir)}. Processing...")

    updated_count = 0
    skipped_existing = 0
    unmatched_files = []  # <-- List to collect CSVs with no matching embryo

    for file in csv_files:
        channel, embryo_tif = extract_stem_and_channel(file)
        csv_path = os.path.join(normalized_csv_dir, file)
        
        idx = db.index[db['cropped_image_file'] == embryo_tif]
        
        # Check if embryo is missing from database
        if len(idx) == 0:
            unmatched_files.append({'csv_file': file, 'searched_embryo_name': embryo_tif})
            continue
            
        if channel not in col_map:
            continue
            
        c_names = col_map[channel]
        
        # --- OVERWRITE LOGIC ---
        if not force_overwrite:
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

    # Save Master Database
    db.to_csv(master_db_path, index=False)

    print(f"✅ Updated: {updated_count} | Skipped (Protected): {skipped_existing}")
    
    # Save Unmatched Log
    if len(unmatched_files) > 0:
        df_unmatched = pd.DataFrame(unmatched_files)
        # Name the log file dynamically based on the folder name so strict/weak don't overwrite each other
        log_name = f"unmatched_csvs_{os.path.basename(normalized_csv_dir)}.csv"
        df_unmatched.to_csv(log_name, index=False)
        print(f"⚠️ WARNING: {len(unmatched_files)} CSVs had no matching embryo in the database!")
        print(f"   -> Saved details to '{log_name}'")