#!/usr/bin/env python3
"""
SNR Analysis Script for NPZ Image Pairs

This script loads clean/noisy image pairs from an NPZ file,
and calculates SNR metrics from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==================== LOAD IMAGE PAIR ====================
NPZ_PATH = "simulation_dataset/pair_000.npz"

def load_npz_pair(npz_path):
    """Load a clean/noisy image pair from an NPZ file."""
    try:
        data = np.load(npz_path)
        clean_image = data['clean']
        noisy_image = data['noisy']
        return clean_image, noisy_image
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None, None

# ==================== SNR CALCULATIONS ====================
def calculate_snr_basic(input_image, clean_image):
    """
    Calculate SNR from scratch using the basic formula.
    
    SNR = 10 * log10(signal_power / noise_power)
    where:
    - signal_power = mean(clean_image^2)
    - noise_power = mean((input_image - clean_image)^2)
    """
    # Extract noise
    noise = input_image - clean_image
    
    # Calculate signal power (using clean image as true signal)
    signal_power = np.mean(clean_image ** 2)
    
    # Calculate noise power
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR
    if noise_power < 1e-10:  # Avoid division by zero
        snr_linear = float('inf')
        snr_db = float('inf')
    else:
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
    
    # Calculate PSNR
    data_range = np.max(clean_image) - np.min(clean_image)
    rmse = np.sqrt(noise_power)
    if rmse < 1e-10 or data_range < 1e-10:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(data_range / rmse)
    
    return {
        'signal_power': signal_power,
        'noise_power': noise_power,
        'snr_linear': snr_linear,
        'snr_db': snr_db,
        'rmse': rmse,
        'psnr': psnr,
        'min_clean': np.min(clean_image),
        'max_clean': np.max(clean_image),
        'min_noise': np.min(noise),
        'max_noise': np.max(noise),
        'mean_noise': np.mean(noise),
        'std_noise': np.std(noise)
    }

# ==================== MAIN FUNCTION ====================
def main():
    # Check if file exists
    if not os.path.exists(NPZ_PATH):
        print(f"File not found: {NPZ_PATH}")
        return
    
    # Load the image pair
    clean_image, noisy_image = load_npz_pair(NPZ_PATH)
    if clean_image is None or noisy_image is None:
        return
    
    print(f"Clean image shape: {clean_image.shape}, dtype: {clean_image.dtype}")
    print(f"Noisy image shape: {noisy_image.shape}, dtype: {noisy_image.dtype}")
    print(f"Min/Max values - Clean: [{np.min(clean_image)}, {np.max(clean_image)}], "
          f"Noisy: [{np.min(noisy_image)}, {np.max(noisy_image)}]")
    
    # Calculate SNR for noisy image
    metrics = calculate_snr_basic(noisy_image, clean_image)
    
    print("\nSNR Metrics for Noisy Image:")
    print(f"Signal Power: {metrics['signal_power']:.6e}")
    print(f"Noise Power: {metrics['noise_power']:.6e}")
    print(f"SNR (linear): {metrics['snr_linear']:.6f}")
    print(f"SNR (dB): {metrics['snr_db']:.6f} dB")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.6f} dB")
    print(f"Noise Stats - Min: {metrics['min_noise']:.6f}, Max: {metrics['max_noise']:.6f}, "
          f"Mean: {metrics['mean_noise']:.6e}, Std: {metrics['std_noise']:.6e}")
    
    # Visualize the images and the noise
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Clean image
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(clean_image, cmap='gray')
    ax1.set_title("Clean Image")
    plt.colorbar(im1, ax=ax1)
    
    # Noisy image
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(noisy_image, cmap='gray')
    ax2.set_title("Noisy Image")
    plt.colorbar(im2, ax=ax2)
    
    # Noise (difference)
    noise = noisy_image - clean_image
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(noise, cmap='coolwarm')
    ax3.set_title("Noise (Noisy - Clean)")
    plt.colorbar(im3, ax=ax3)
    
    # Histograms
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(clean_image.ravel(), bins=50, alpha=0.7, label="Clean Image")
    ax4.set_title("Clean Image Histogram")
    ax4.set_xlabel("Pixel Value")
    ax4.set_ylabel("Frequency")
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(noisy_image.ravel(), bins=50, alpha=0.7, label="Noisy Image")
    ax5.set_title("Noisy Image Histogram")
    ax5.set_xlabel("Pixel Value")
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(noise.ravel(), bins=50, alpha=0.7, label="Noise", color='red')
    ax6.set_title(f"Noise Histogram (Mean={np.mean(noise):.2e}, Std={np.std(noise):.2e})")
    ax6.set_xlabel("Noise Value")
    
    plt.tight_layout()
    plt.savefig("snr_analysis.png", dpi=150)
    print("\nAnalysis saved to snr_analysis.png")
    plt.show()

if __name__ == "__main__":
    main() 