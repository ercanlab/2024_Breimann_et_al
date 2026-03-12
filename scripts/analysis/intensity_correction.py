import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import warnings

# Suppress scipy warnings for sparse gamma fits
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def process_embryo_intensity(csv_path, out_csv_path, plot_path=None):
    """
    Reads an RS-FISH CSV, applies Z-correction and Gamma normalization,
    and optionally saves a diagnostic plot.
    """
    if os.path.getsize(csv_path) == 0:
        pd.DataFrame(columns=['x', 'y', 'z', 'intensity', 'normalized_intensity']).to_csv(out_csv_path, index=False)
        return False

    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist and if there are enough spots
    if 'intensity' not in df.columns or 'z' not in df.columns or len(df) == 0:
        df.to_csv(out_csv_path, index=False)
        return False

    # ---------------------------------------------------------
    # STEP 1: Z-Correction (Quadratic Fit)
    # ---------------------------------------------------------
    if len(df) > 15 and df['z'].nunique() > 3:
        # Get median intensity per Z slice to avoid transcription sites skewing the fit
        z_medians = df.groupby('z')['intensity'].median().reset_index()
        
        # Fit quadratic (Degree 2 polynomial)
        coeffs = np.polyfit(z_medians['z'], z_medians['intensity'], 2)
        poly_func = np.poly1d(coeffs)
        
        # Calculate expected intensity for every spot based on its Z position
        df['expected_intensity'] = poly_func(df['z'])
        
        # Prevent division by zero or negative values if the curve drops too low
        min_allowed = df['intensity'].median() * 0.1
        df['expected_intensity'] = df['expected_intensity'].clip(lower=min_allowed)
        
        # Normalize to the maximum of the expected curve so raw values stay roughly in the same numerical range
        correction_factor = df['expected_intensity'] / df['expected_intensity'].max()
        df['z_corrected_intensity'] = df['intensity'] / correction_factor
    else:
        # Fallback for embryos with almost no spots
        df['z_corrected_intensity'] = df['intensity']
        poly_func = None

    # ---------------------------------------------------------
    # STEP 2: Gamma Fit & Normalization
    # ---------------------------------------------------------
    if len(df) > 15:
        # Fit Gamma distribution to the z-corrected intensities
        # (Using data < 95th percentile to prevent massive transcription sites from ruining the fit)
        p95 = df['z_corrected_intensity'].quantile(0.95)
        fit_data = df[df['z_corrected_intensity'] <= p95]['z_corrected_intensity']
        
        shape, loc, scale = gamma.fit(fit_data)
        
        # Calculate the Mode (Maximum of the PDF curve)
        # Formula for Gamma mode: loc + (shape - 1) * scale (if shape > 1)
        if shape > 1:
            mode_intensity = loc + (shape - 1) * scale
        else:
            mode_intensity = loc  # Fallback
            
        # Safety catch: if mode is somehow <= 0, fallback to median
        if mode_intensity <= 0:
            mode_intensity = df['z_corrected_intensity'].median()
    else:
        # Fallback for sparse data
        mode_intensity = df['z_corrected_intensity'].median()
        shape, loc, scale = None, None, None

    # Apply final normalization
    # Now, a spot with intensity 1.0 equals 1 transcript!
    df['normalized_intensity'] = df['z_corrected_intensity'] / mode_intensity

    # Save CSV
    df.to_csv(out_csv_path, index=False)

    # ---------------------------------------------------------
    # STEP 3: Diagnostic Plot (Optional but highly recommended)
    # ---------------------------------------------------------
    if plot_path and len(df) > 15:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{os.path.basename(csv_path)} | Mode = {mode_intensity:.1f}")
        
        # Plot 1: Z-Correction
        axes[0].scatter(df['z'], df['intensity'], alpha=0.2, color='gray', label='Raw Spots')
        axes[0].scatter(z_medians['z'], z_medians['intensity'], color='red', label='Z-Medians')
        if poly_func:
            z_range = np.linspace(df['z'].min(), df['z'].max(), 50)
            axes[0].plot(z_range, poly_func(z_range), color='blue', linewidth=2, label='Quadratic Fit')
        axes[0].set_title("Z-Dependent Signal Loss")
        axes[0].set_xlabel("Z-slice")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()
        
        # Plot 2: Gamma Fit
        x = np.linspace(0, p95, 100)
        pdf_gamma = gamma.pdf(x, shape, loc, scale)
        axes[1].hist(fit_data, bins=30, density=True, alpha=0.6, color='purple', label='Spot Intensities')
        axes[1].plot(x, pdf_gamma, 'k-', lw=2, label='Gamma Fit')
        axes[1].axvline(mode_intensity, color='red', linestyle='dashed', label='Mode (Set to 1.0)')
        axes[1].set_title("Intensity Histogram & Gamma Fit")
        axes[1].set_xlabel("Z-Corrected Intensity")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)

    return True

def run_intensity_correction(input_folder, output_folder, plot_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    if plot_folder:
        os.makedirs(plot_folder, exist_ok=True)
        
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"Starting intensity correction on {len(csv_files)} files...")
    
    processed = 0
    for file in csv_files:
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        plot_path = os.path.join(plot_folder, file.replace('.csv', '.png')) if plot_folder else None
        
        success = process_embryo_intensity(in_path, out_path, plot_path)
        if success:
            processed += 1
            
        if processed > 0 and processed % 500 == 0:
            print(f"Processed {processed} files...")
            
    print(f"✅ Completed. Corrected data saved to: {output_folder}")
    if plot_folder:
        print(f"📈 Diagnostic plots saved to: {plot_folder}")