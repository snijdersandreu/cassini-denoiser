#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def load_image_pair(npz_path):
    """Load a clean/noisy image pair from an NPZ file"""
    data = np.load(npz_path)
    return data['clean'], data['noisy']

def compare_histograms(original_dir, new_dir, index=0):
    """
    Compare histograms between original and new histogram-matched images
    
    Parameters:
        original_dir (str): Directory with original images
        new_dir (str): Directory with new histogram-matched images
        index (int): Image pair index to compare
    """
    # Paths to the image files
    original_path = os.path.join(original_dir, f"pair_{index:03d}.npz")
    new_path = os.path.join(new_dir, f"pair_{index:03d}.npz")
    
    # Load image pairs
    if os.path.exists(original_path) and os.path.exists(new_path):
        orig_clean, orig_noisy = load_image_pair(original_path)
        new_clean, new_noisy = load_image_pair(new_path)
        
        # Create figure for comparison
        plt.figure(figsize=(15, 10))
        
        # Plot original clean image
        plt.subplot(2, 3, 1)
        plt.imshow(orig_clean, cmap='gray')
        plt.title('Original Clean')
        plt.axis('off')
        
        # Plot new clean image
        plt.subplot(2, 3, 2)
        plt.imshow(new_clean, cmap='gray')
        plt.title('Histogram-Matched Clean')
        plt.axis('off')
        
        # Plot difference
        plt.subplot(2, 3, 3)
        plt.imshow(new_clean - orig_clean, cmap='coolwarm')
        plt.title('Difference (New - Original)')
        plt.axis('off')
        
        # Plot original histogram
        plt.subplot(2, 3, 4)
        plt.hist(orig_clean.flatten(), bins=50, alpha=0.7)
        plt.title('Original Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Plot new histogram
        plt.subplot(2, 3, 5)
        plt.hist(new_clean.flatten(), bins=50, alpha=0.7)
        plt.title('New Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Plot overlay of both histograms
        plt.subplot(2, 3, 6)
        plt.hist(orig_clean.flatten(), bins=50, alpha=0.5, label='Original')
        plt.hist(new_clean.flatten(), bins=50, alpha=0.5, label='New')
        plt.title('Histogram Comparison')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'histogram_comparison_{index:03d}.png', dpi=300)
        plt.close()
        
        # Print some statistics
        print(f"Image Pair {index}:")
        print(f"  Original Clean - Min: {orig_clean.min():.6f}, Max: {orig_clean.max():.6f}, Mean: {orig_clean.mean():.6f}, Std: {orig_clean.std():.6f}")
        print(f"  New Clean      - Min: {new_clean.min():.6f}, Max: {new_clean.max():.6f}, Mean: {new_clean.mean():.6f}, Std: {new_clean.std():.6f}")
        
        # Calculate RMS contrast for both
        orig_rms_contrast = orig_clean.std() / orig_clean.mean()
        new_rms_contrast = new_clean.std() / new_clean.mean()
        print(f"  Original RMS Contrast: {orig_rms_contrast:.6f}")
        print(f"  New RMS Contrast: {new_rms_contrast:.6f}")
        
        return True
    else:
        print(f"Error: Image pair {index} not found in both directories")
        return False

def main():
    # Create backup of original dataset
    if not os.path.exists("simulation_dataset_original"):
        print("Creating backup of original simulation dataset...")
        os.makedirs("simulation_dataset_original", exist_ok=True)
        for file in glob.glob("simulation_dataset/pair_*.npz"):
            # Check if this file was generated before the histogram matching was applied
            file_time = os.path.getmtime(file)
            current_time = os.path.getmtime("adapt_simulation.py")
            if file_time < current_time:
                # Copy it to the original directory using numpy load/save
                basename = os.path.basename(file)
                data = np.load(file)
                np.savez(os.path.join("simulation_dataset_original", basename), 
                        clean=data['clean'], noisy=data['noisy'])
                print(f"  Backed up {basename}")
    
    # Compare histograms for several images
    for i in range(5):  # Compare first 5 image pairs
        compare_histograms("simulation_dataset_original", "simulation_dataset", i)

if __name__ == "__main__":
    main() 