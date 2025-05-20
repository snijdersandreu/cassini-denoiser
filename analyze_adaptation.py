#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import pandas as pd
from pds import parse_header, read_image

def load_image_pair(npz_path):
    """Load a clean/noisy image pair from an NPZ file"""
    data = np.load(npz_path)
    return data['clean'], data['noisy']

def find_image_file(filename, data_dir="data"):
    """
    Find an image file (LBL or IMG) in the data directory structure.
    
    Parameters:
        filename (str): Filename pattern to search for
        data_dir (str): Base directory to search in
        
    Returns:
        tuple: (header_path, image_path) or (None, None) if not found
    """
    from pathlib import Path
    
    # Extract base name without extension (in case input has extension)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # First, try to find the LBL file
    lbl_patterns = [
        f"**/{base_name}*.LBL",  # Label files with any suffix
        f"**/{base_name}_CALIB.LBL"   # Calibrated label files
    ]
    
    header_path = None
    for pattern in lbl_patterns:
        matches = list(Path(data_dir).glob(pattern))
        if matches:
            header_path = str(matches[0])
            break
    
    if not header_path:
        return None, None
    
    # Now find the corresponding IMG file
    base_img_path = os.path.splitext(header_path)[0]
    image_path = f"{base_img_path}.IMG"
    
    # Check if file exists (case sensitive)
    if not os.path.exists(image_path):
        image_path = f"{base_img_path}.img"  # Try lowercase extension
        if not os.path.exists(image_path):
            return header_path, None  # LBL found but IMG not found
    
    return header_path, image_path

def load_real_image_patches(csv_path="noise_characterization_patch_logs.csv", data_dir="data", max_patches=10):
    """
    Load patches from real Cassini images.
    
    Parameters:
        csv_path (str): Path to the CSV file with patch coordinates
        data_dir (str): Directory containing the Cassini data
        max_patches (int): Maximum number of patches to load
        
    Returns:
        list: List of image patches from real Cassini images
    """
    patches = []
    
    try:
        # Read the CSV file
        import csv
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                if i >= max_patches:
                    break
                    
                filename = row['filename']
                
                # Extract patch coordinates
                coords = (
                    int(row['orig_x0']),
                    int(row['orig_x1']),
                    int(row['orig_y0']),
                    int(row['orig_y1'])
                )
                
                # Find the actual image file
                header_path, img_path = find_image_file(filename, data_dir)
                
                if header_path and img_path:
                    try:
                        # Load the image
                        image = read_image(header_file_path=header_path, image_file_path=img_path, keep_float=True)
                        
                        # Extract the patch
                        x0, x1, y0, y1 = coords
                        patch = image[y0:y1, x0:x1]
                        
                        # Add to patches list
                        patches.append(patch)
                        print(f"Loaded patch from {filename}")
                    except Exception as e:
                        print(f"Error loading patch from {filename}: {e}")
            
    except Exception as e:
        print(f"Error loading real image patches: {e}")
    
    return patches

def analyze_histogram_adaptation(dataset_dir="simulation_dataset", index=0):
    """
    Analyze the histogram adaptation of a simulated image pair
    
    Parameters:
        dataset_dir (str): Directory with histogram-matched images
        index (int): Image pair index to analyze
    """
    # Paths to the image file
    image_path = os.path.join(dataset_dir, f"pair_{index:03d}.npz")
    
    # Load image pair
    if os.path.exists(image_path):
        clean, noisy = load_image_pair(image_path)
        
        # Load real image patches for comparison
        real_patches = load_real_image_patches(max_patches=5)
        if not real_patches:
            print("Warning: No real image patches loaded for comparison")
            return False
        
        # Combine patches for histogram analysis
        real_pixels = np.concatenate([patch.flatten() for patch in real_patches])
        
        # Create figure for comparison
        plt.figure(figsize=(15, 10))
        
        # Plot clean image
        plt.subplot(2, 3, 1)
        plt.imshow(clean, cmap='gray')
        plt.title('Clean Image')
        plt.axis('off')
        
        # Plot noisy image
        plt.subplot(2, 3, 2)
        plt.imshow(noisy, cmap='gray')
        plt.title('Noisy Image')
        plt.axis('off')
        
        # Plot real image patch example
        plt.subplot(2, 3, 3)
        plt.imshow(real_patches[0], cmap='gray')
        plt.title('Real Cassini Patch Example')
        plt.axis('off')
        
        # Plot clean image histogram
        plt.subplot(2, 3, 4)
        plt.hist(clean.flatten(), bins=50, alpha=0.7)
        plt.title('Clean Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Plot real image pixels histogram
        plt.subplot(2, 3, 5)
        plt.hist(real_pixels, bins=50, alpha=0.7)
        plt.title('Real Cassini Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Plot overlay of both histograms
        plt.subplot(2, 3, 6)
        plt.hist(clean.flatten(), bins=50, alpha=0.5, label='Simulation')
        plt.hist(real_pixels, bins=50, alpha=0.5, label='Real Cassini')
        plt.title('Histogram Comparison')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'histogram_analysis_{index:03d}.png', dpi=300)
        plt.close()
        
        # Print some statistics
        print(f"Image Pair {index}:")
        print(f"  Simulation - Min: {clean.min():.6f}, Max: {clean.max():.6f}, Mean: {clean.mean():.6f}, Std: {clean.std():.6f}")
        
        # Calculate RMS contrast
        sim_rms_contrast = clean.std() / clean.mean() if clean.mean() != 0 else 0
        real_pixel_mean = np.mean(real_pixels)
        real_pixel_std = np.std(real_pixels)
        real_rms_contrast = real_pixel_std / real_pixel_mean if real_pixel_mean != 0 else 0
        
        print(f"  Simulation RMS Contrast: {sim_rms_contrast:.6f}")
        print(f"  Real Cassini RMS Contrast: {real_rms_contrast:.6f}")
        
        # Calculate histogram similarity using Kullback-Leibler divergence
        from scipy.stats import entropy
        
        # Use histogram with fixed bins for comparison
        bins = np.linspace(min(clean.min(), real_pixels.min()), max(clean.max(), real_pixels.max()), 100)
        sim_hist, _ = np.histogram(clean.flatten(), bins=bins, density=True)
        real_hist, _ = np.histogram(real_pixels, bins=bins, density=True)
        
        # Replace zeros with small values to avoid division by zero in KL divergence
        sim_hist = np.where(sim_hist == 0, 1e-10, sim_hist)
        real_hist = np.where(real_hist == 0, 1e-10, real_hist)
        
        # Calculate KL divergence (lower is better)
        kl_div = entropy(real_hist, sim_hist)
        print(f"  Histogram KL Divergence: {kl_div:.6f}")
        
        return True
    else:
        print(f"Error: Image pair {index} not found")
        return False

def main():
    # Analyze histogram adaptation for several images
    for i in range(3):  # Analyze first 3 image pairs
        print(f"\n--- Analyzing Image Pair {i} ---")
        analyze_histogram_adaptation("simulation_dataset", i)

if __name__ == "__main__":
    main() 