#%%
import os
import pandas as pd
from PIL import Image
import numpy as np
import re

#%%

# Paths to folders
localization_folder = '/Users/laurabreimann/Documents/Postdoc/Colabs/PLA_Christine/Duolink assay 2023 Nov-Dec/puro-PLA/DoG_filtered/EpHA4'  
mask_folder = '/Users/laurabreimann/Documents/Postdoc/Colabs/PLA_Christine/Duolink assay 2023 Nov-Dec/masks/soma_masks'                  
output_folder = '/Users/laurabreimann/Documents/Postdoc/Colabs/PLA_Christine/Duolink assay 2023 Nov-Dec/puro-PLA/test'               

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


#%%

# Define a function to extract the common stem from a filename    
def extract_common_stem(filename):
    """
    Extract the common stem from the given filename based on the provided pattern.
    """
    match = re.match(r"(.*?)(?:_C\d+\.tif|_GFP_max_segmentation).*", filename)
    if match:
        return match.group(1)
    else:
        return None    



#%%

 # Iterate over the CSV files in the localization folder
for csv_filename in os.listdir(localization_folder):
    if csv_filename.endswith('.csv'):
        # Construct the full path to the CSV file
        csv_path = os.path.join(localization_folder, csv_filename)
        
        # Extract the stem of the filename using the pattern
        stem = extract_common_stem(csv_filename)
        
        if stem is None:
            print(f"Could not determine the common stem for {csv_filename}")
            continue
        
        # Find the corresponding mask image
        mask_filename = None
        for potential_mask in os.listdir(mask_folder):
            if potential_mask.startswith(stem) and "_GFP_max_segmentation" in potential_mask:
                mask_filename = potential_mask
                break
        
        if mask_filename:
            # Load the mask image
            mask_path = os.path.join(mask_folder, mask_filename)
            mask_image = Image.open(mask_path)
            mask_array = np.array(mask_image) == 1  # Assuming binary mask
            
            # Read the CSV file into a DataFrame
            localizations = pd.read_csv(csv_path)
            
            # Apply the mask and filter the localizations
            before_filtering_count = len(localizations)
            filtered_localizations = localizations[
                localizations.apply(lambda row: mask_array[int(row['y']), int(row['x'])] if 0 <= int(row['y']) < mask_array.shape[0] and 0 <= int(row['x']) < mask_array.shape[1] else False, axis=1)
            ]
            after_filtering_count = len(filtered_localizations)

            # Check if any localizations remain after filtering
            if after_filtering_count == 0:
                print(f"All localizations were filtered out for {csv_filename}. Check mask alignment.")

            # Save the filtered DataFrame to a new CSV with only the stem in the filename
            output_csv_filename = stem + '_filtered_localizations.csv'
            output_csv_path = os.path.join(output_folder, output_csv_filename)
            filtered_localizations.to_csv(output_csv_path, index=False)
        else:
            print(f"No matching mask found for {csv_filename}")
# %%
