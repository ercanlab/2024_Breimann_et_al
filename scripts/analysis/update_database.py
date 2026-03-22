import pandas as pd
import os
import shutil
from datetime import datetime

def get_csv_metrics(csv_path, tx_threshold):
    """ Reads an RS-FISH CSV and calculates total spots and TSS metrics. """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0, 0.0, 0, 0.0
    
    try:
        df_spots = pd.read_csv(csv_path)
        int_col = 'normalized_intensity' if 'normalized_intensity' in df_spots.columns else 'intensity'
        
        if int_col not in df_spots.columns or len(df_spots) == 0:
            return 0, 0.0, 0, 0.0
            
        total_spots = len(df_spots)
        total_int = df_spots[int_col].sum()
        
        tx_sites = df_spots[df_spots[int_col] >= tx_threshold]
        tss_spots = len(tx_sites)
        tss_int = tx_sites[int_col].sum()
        
        return total_spots, total_int, tss_spots, tss_int
    except Exception:
        return 0, 0.0, 0, 0.0

def run_database_update_max(master_db_path, strict_dir, weak_dir, tx_threshold=2.0, force_overwrite=True):
    
    # 1. Create a safe backup
    if force_overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = master_db_path.replace('.csv', f'_max_spots_update_{timestamp}.csv')
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

    updated_count = 0
    skipped_count = 0
    strict_wins = 0
    weak_wins = 0
    zero_ties = 0

    print("Evaluating channels independently to find maximum transcripts...")

    # Iterate through every embryo in the database
    for idx, row in db.iterrows():
        embryo_tif = row['cropped_image_file']
        if pd.isna(embryo_tif):
            continue
            
        base_stem = embryo_tif.replace('.tif', '')
        
        # Evaluate each channel independently
        for ch, c_names in col_map.items():
            
            # Check overwrite protection
            if not force_overwrite:
                curr_val = row.get(c_names['total_spots'], -1)
                if pd.notna(curr_val) and curr_val > 0:
                    skipped_count += 1
                    continue
            
            # Construct possible filenames
            csv_name_1 = f"{ch}_{base_stem}.csv"
            csv_name_2 = f"{ch}_{base_stem}.tif.csv" # The double-extension fallback
            
            # Resolve Strict Path
            s_path = os.path.join(strict_dir, csv_name_1)
            if not os.path.exists(s_path): s_path = os.path.join(strict_dir, csv_name_2)
                
            # Resolve Weak Path
            w_path = os.path.join(weak_dir, csv_name_1)
            if not os.path.exists(w_path): w_path = os.path.join(weak_dir, csv_name_2)
            
            # Calculate metrics for both
            s_tot, s_int, s_tss, s_tss_int = get_csv_metrics(s_path, tx_threshold)
            w_tot, w_int, w_tss, w_tss_int = get_csv_metrics(w_path, tx_threshold)
            
            # Compare and pick the winner!
            # If there's a tie, we bias towards Strict (which is scientifically safer)
            if s_tot >= w_tot:
                best_tot, best_int, best_tss, best_tss_int = s_tot, s_int, s_tss, s_tss_int
                if s_tot > 0:
                    strict_wins += 1
                else:
                    zero_ties += 1
            else:
                best_tot, best_int, best_tss, best_tss_int = w_tot, w_int, w_tss, w_tss_int
                weak_wins += 1
                
            # Write the winner to the database
            db.at[idx, c_names['total_spots']] = best_tot
            db.at[idx, c_names['total_int']] = best_int
            db.at[idx, c_names['tss_spots']] = best_tss
            db.at[idx, c_names['tss_int']] = best_tss_int
            
            updated_count += 1

    # Save
    db.to_csv(master_db_path, index=False)

    print("\n==========================================")
    print("✅ DATABASE MAX-SPOT UPDATE COMPLETE")
    print("==========================================")
    print(f"Total channels updated     : {updated_count}")
    print(f"  -> STRICT settings chosen: {strict_wins} channels")
    print(f"  -> WEAK settings chosen  : {weak_wins} channels")
    print(f"  -> No spots in either    : {zero_ties} channels")
    print(f"Channels skipped (>0 protected): {skipped_count}")