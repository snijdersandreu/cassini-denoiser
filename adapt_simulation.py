#!/usr/bin/env python
# Import external modules
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
import glob
from pathlib import Path


def load_reference_histogram():
    """
    Load the reference histogram data for contrast matching.
    
    Returns:
        dict: Dictionary containing RMS and Michelson contrast arrays
    """
    try:
        ref_data = np.load('reference_histogram_full_range.npz')
        return {
            'rms_contrast': ref_data['rms_contrast'],
            'michelson_contrast': ref_data['michelson_contrast']
        }
    except Exception as e:
        print(f"Error loading reference histogram: {e}")
        return None


def downscale_image(image, bin_size=5):
    """
    Downscale an image using bin averaging.
    
    Parameters:
        image (ndarray): Input image
        bin_size (int): Size of the bin for averaging
        
    Returns:
        ndarray: Downscaled image
    """
    # Get original dimensions
    height, width = image.shape
    
    # Calculate new dimensions
    new_height = height // bin_size
    new_width = width // bin_size
    
    # Prepare output array
    downscaled = np.zeros((new_height, new_width), dtype=np.float32)
    
    # Perform binning
    for i in range(new_height):
        for j in range(new_width):
            # Extract bin
            bin_data = image[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]
            # Average values in bin
            downscaled[i, j] = np.mean(bin_data)
    
    return downscaled


def extract_sections(image, size=100, stride=50):
    """
    Extract overlapping sections from an image.
    
    Parameters:
        image (ndarray): Input image
        size (int): Size of each square section
        stride (int): Step size between sections
        
    Returns:
        list: List of extracted sections and their coordinates (section, (y, x))
    """
    height, width = image.shape
    sections = []
    
    # Calculate how many complete sections can fit
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            # Extract section
            section = image[y:y+size, x:x+size]
            if section.shape == (size, size):  # Ensure section is complete
                sections.append((section, (y, x)))
    
    return sections


def resize_to_target(image, target_size=1024):
    """
    Resize an image to a target size using bilinear interpolation.
    
    Parameters:
        image (ndarray): Input image
        target_size (int): Target size for both dimensions
        
    Returns:
        ndarray: Resized image
    """
    # Convert to PIL Image for resizing
    pil_img = Image.fromarray(image)
    
    # Resize to target size
    resized_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert back to numpy array
    return np.array(resized_img, dtype=np.float32)


def match_contrast(image, target_contrast):
    """
    Adjust the contrast of an image to match a target contrast.
    
    Parameters:
        image (ndarray): Input image
        target_contrast (float): Target RMS contrast value
        
    Returns:
        ndarray: Contrast-adjusted image
    """
    # Compute current contrast
    current_contrast = np.std(image) / np.mean(image)
    
    # Calculate scaling factor to match target contrast
    scaling_factor = target_contrast / current_contrast
    
    # Preserve mean while adjusting contrast
    mean_val = np.mean(image)
    adjusted = mean_val + (image - mean_val) * scaling_factor
    
    return adjusted


def add_gaussian_noise(image, std_dev=0.001224):
    """
    Add Additive Gaussian White Noise (AGWN) to an image.
    
    Parameters:
        image (ndarray): Input image
        std_dev (float): Standard deviation of the noise
        
    Returns:
        ndarray: Noisy image, clean image (for reference)
    """
    # Create noise with the specified standard deviation
    noise = np.random.normal(0, std_dev, image.shape)
    
    # Add noise to the image
    noisy_image = image + noise
    
    return noisy_image, image


def save_image_pair(clean, noisy, output_dir, index):
    """
    Save a pair of clean and noisy images.
    
    Parameters:
        clean (ndarray): Clean image
        noisy (ndarray): Noisy image
        output_dir (str): Output directory
        index (int): Pair index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as high-precision NPZ files
    np.savez(
        os.path.join(output_dir, f'pair_{index:03d}.npz'),
        clean=clean,
        noisy=noisy
    )
    
    # Save as PNG for visualization
    plt.figure(figsize=(20, 10))
    
    # Display clean image
    plt.subplot(1, 2, 1)
    plt.imshow(clean, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')
    
    # Display noisy image
    plt.subplot(1, 2, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pair_{index:03d}.png'), dpi=150)
    plt.close()


def process_simulation(input_path, output_dir, bin_size=5, section_size=100, stride=50, noise_std=0.001224):
    """
    Process a simulation image according to the adaptation pipeline.
    
    Parameters:
        input_path (str): Path to the input simulation image
        output_dir (str): Output directory for processed image pairs
        bin_size (int): Size of bins for downscaling
        section_size (int): Size of image sections for extraction
        stride (int): Stride between sections
        noise_std (float): Noise standard deviation
    """
    # 1. Load the reference histogram
    ref_histogram = load_reference_histogram()
    if ref_histogram is None:
        print("Cannot proceed without reference histogram.")
        return
    
    # Calculate target contrast as the median of RMS contrast values
    target_contrast = np.median(ref_histogram['rms_contrast'])
    print(f"Target RMS contrast: {target_contrast:.6f}")
    
    # 2. Load the simulation image
    try:
        raw_image = np.array(Image.open(input_path).convert('F'))
    except Exception as e:
        print(f"Error loading image {input_path}: {e}")
        return
    
    print(f"Loaded image with shape: {raw_image.shape}")
    
    # 3. Downscale the image
    print(f"Downscaling with bin size {bin_size}...")
    downscaled = downscale_image(raw_image, bin_size)
    print(f"Downscaled image shape: {downscaled.shape}")
    
    # 4. Extract sections
    print(f"Extracting {section_size}x{section_size} sections with stride {stride}...")
    sections = extract_sections(downscaled, section_size, stride)
    print(f"Extracted {len(sections)} sections")
    
    # 5. Process each section
    for i, (section, coords) in enumerate(sections):
        print(f"Processing section {i+1}/{len(sections)} at coordinates {coords}...")
        
        # Match contrast
        print("  Matching contrast...")
        contrast_matched = match_contrast(section, target_contrast)
        
        # Add noise
        print("  Adding noise...")
        noisy_section, clean_section = add_gaussian_noise(contrast_matched, noise_std)
        
        # Save the pair
        save_image_pair(clean_section, noisy_section, output_dir, i)
    
    print(f"All sections processed and saved to {output_dir}")


def main():
    # Input and output paths
    input_path = os.path.join("StormAlley", "raw_pvort.png")
    output_dir = "simulation_dataset"
    
    # Process parameters
    bin_size = 5
    section_size = 100  # Smaller section size to fit downscaled dimensions
    stride = 50  # Smaller stride for more sections
    noise_std = 0.001224
    
    # Run the processing pipeline
    process_simulation(input_path, output_dir, bin_size, section_size, stride, noise_std)
    
    print("\nSummary of dataset creation:")
    print("----------------------------")
    print(f"Source image: {input_path}")
    print(f"Downscale bin size: {bin_size}x{bin_size}")
    print(f"Section size: {section_size}x{section_size}")
    print(f"Section stride: {stride}")
    print(f"Noise standard deviation: {noise_std}")
    
    # Count the generated pairs
    num_pairs = len(glob.glob(os.path.join(output_dir, "pair_*.npz")))
    print(f"Total image pairs generated: {num_pairs}")


if __name__ == "__main__":
    main() 