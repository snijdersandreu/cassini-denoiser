#!/usr/bin/env python
# Import external modules
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path

# Import Cassini PDS image parser
from pds import parse_header, read_image


def compute_rms_contrast(image):
    """
    Compute RMS (Root Mean Square) contrast of an image.
    
    RMS contrast is defined as the standard deviation of pixel values 
    divided by the mean (for non-zero mean)
    
    Parameters:
        image (ndarray): 2D array representing the image.
        
    Returns:
        float: RMS contrast value
    """
    # Ensure we have a valid image
    if image.size == 0:
        return 0.0
    
    # Calculate RMS contrast: standard deviation / mean (for non-zero mean)
    mean_val = np.mean(image)
    
    # Avoid division by zero
    if abs(mean_val) < 1e-10:
        return 0.0
    
    # Standard deviation / mean
    rms_contrast = np.std(image) / mean_val
    
    return rms_contrast


def compute_michelson_contrast(image):
    """
    Compute Michelson contrast of an image.
    
    Michelson contrast is defined as (Imax - Imin) / (Imax + Imin)
    
    Parameters:
        image (ndarray): 2D array representing the image.
        
    Returns:
        float: Michelson contrast value
    """
    # Ensure we have a valid image
    if image.size == 0:
        return 0.0
    
    # Get min and max pixel values
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Avoid division by zero
    if (max_val + min_val) < 1e-10:
        return 0.0
    
    # Calculate Michelson contrast
    michelson_contrast = (max_val - min_val) / (max_val + min_val)
    
    return michelson_contrast


def find_image_file(filename, data_dir="data"):
    """
    Find an image file (LBL or IMG) in the data directory structure.
    
    Parameters:
        filename (str): Filename pattern to search for
        data_dir (str): Base directory to search in
        
    Returns:
        tuple: (header_path, image_path) or (None, None) if not found
    """
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


def analyze_image_contrast(header_path, img_path, region_coords=None):
    """
    Analyze the contrast of an image or specific region.
    
    Parameters:
        header_path (str): Path to the LBL header file
        img_path (str): Path to the IMG image file
        region_coords (tuple): Optional (x0, x1, y0, y1) to specify a region
        
    Returns:
        dict: Dictionary with RMS and Michelson contrast values
    """
    # Load the image using PDS module
    try:
        image = read_image(header_file_path=header_path, image_file_path=img_path, keep_float=True)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return {
            "rms_contrast": None,
            "michelson_contrast": None
        }
    
    # If region coordinates provided, extract the region
    if region_coords is not None:
        x0, x1, y0, y1 = region_coords
        region = image[y0:y1, x0:x1]
    else:
        region = image
    
    # Compute contrasts
    rms = compute_rms_contrast(region)
    michelson = compute_michelson_contrast(region)
    
    return {
        "rms_contrast": rms,
        "michelson_contrast": michelson
    }


def main():
    # Path to the noise characterization CSV
    csv_path = "noise_characterization_patch_logs.csv"
    
    # Create results file
    results_file = "contrast_results.csv"
    
    # List to store all contrast values
    rms_contrasts = []
    michelson_contrasts = []
    
    # Create CSV writer
    with open(results_file, 'w', newline='') as outfile:
        fieldnames = ['filename', 'orig_x0', 'orig_x1', 'orig_y0', 'orig_y1', 
                      'rms_contrast', 'michelson_contrast']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Read the noise characterization CSV
        with open(csv_path, 'r') as infile:
            reader = csv.DictReader(infile)
            
            for row in reader:
                filename = row['filename']
                
                # Extract region coordinates
                coords = (
                    int(row['orig_x0']),
                    int(row['orig_x1']),
                    int(row['orig_y0']),
                    int(row['orig_y1'])
                )
                
                # Find the actual image file
                header_path, img_path = find_image_file(filename)
                
                if header_path and img_path:
                    print(f"Processing {filename}...")
                    
                    # Analyze contrast
                    contrast_results = analyze_image_contrast(header_path, img_path, coords)
                    
                    # Add to our lists if valid
                    if contrast_results['rms_contrast'] is not None:
                        rms_contrasts.append(contrast_results['rms_contrast'])
                    
                    if contrast_results['michelson_contrast'] is not None:
                        michelson_contrasts.append(contrast_results['michelson_contrast'])
                    
                    # Write to CSV
                    writer.writerow({
                        'filename': filename,
                        'orig_x0': row['orig_x0'],
                        'orig_x1': row['orig_x1'],
                        'orig_y0': row['orig_y0'],
                        'orig_y1': row['orig_y1'],
                        'rms_contrast': contrast_results['rms_contrast'],
                        'michelson_contrast': contrast_results['michelson_contrast']
                    })
                else:
                    print(f"Could not find image file for {filename}")
    
    # Calculate statistics
    rms_stats = {
        "mean": np.mean(rms_contrasts),
        "median": np.median(rms_contrasts),
        "min": np.min(rms_contrasts),
        "max": np.max(rms_contrasts)
    }
    
    michelson_stats = {
        "mean": np.mean(michelson_contrasts),
        "median": np.median(michelson_contrasts),
        "min": np.min(michelson_contrasts),
        "max": np.max(michelson_contrasts)
    }
    
    # Print statistics
    print("\nRMS Contrast Statistics:")
    print(f"Mean: {rms_stats['mean']:.6f}")
    print(f"Median: {rms_stats['median']:.6f}")
    print(f"Range: {rms_stats['min']:.6f} to {rms_stats['max']:.6f}")
    
    print("\nMichelson Contrast Statistics:")
    print(f"Mean: {michelson_stats['mean']:.6f}")
    print(f"Median: {michelson_stats['median']:.6f}")
    print(f"Range: {michelson_stats['min']:.6f} to {michelson_stats['max']:.6f}")
    
    # Create and save histogram of all pixel values (reference histogram)
    create_reference_histogram(rms_contrasts, michelson_contrasts)


def create_reference_histogram(rms_values, michelson_values):
    """
    Create and save a reference histogram of contrast values.
    
    Parameters:
        rms_values (list): List of RMS contrast values
        michelson_values (list): List of Michelson contrast values
    """
    # Convert to numpy arrays
    rms_array = np.array(rms_values)
    michelson_array = np.array(michelson_values)
    
    # Save raw histogram data as NPZ file
    np.savez(
        'reference_histogram_full_range.npz',
        rms_contrast=rms_array,
        michelson_contrast=michelson_array
    )
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot RMS contrast histogram
    plt.subplot(1, 2, 1)
    plt.hist(rms_array, bins=20, alpha=0.7)
    plt.title('RMS Contrast Distribution')
    plt.xlabel('RMS Contrast')
    plt.ylabel('Frequency')
    
    # Plot Michelson contrast histogram
    plt.subplot(1, 2, 2)
    plt.hist(michelson_array, bins=20, alpha=0.7)
    plt.title('Michelson Contrast Distribution')
    plt.xlabel('Michelson Contrast')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('contrast_distribution.png', dpi=300)
    print("Created contrast distribution visualization: contrast_distribution.png")


if __name__ == "__main__":
    main() 