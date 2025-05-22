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


def histogram_matching(source_image, reference_patches):
    """
    Match the histogram of a source image to a reference distribution derived from patches.
    
    Parameters:
        source_image (ndarray): Source image to transform
        reference_patches (ndarray): Array of reference patches or values 
        
    Returns:
        ndarray: Image with histogram matched to reference distribution
    """
    # Flatten source image for histogram computation
    src_flat = source_image.flatten()
    
    # Flatten reference patches into a single array of values
    ref_flat = reference_patches.flatten()
    
    # Get the sorted unique values and their indices
    src_values, src_indices = np.unique(src_flat, return_inverse=True)
    
    # Calculate the normalized cumulative histograms
    src_quantiles = np.zeros(len(src_values))
    for i in range(len(src_values)):
        src_quantiles[i] = np.sum(src_flat <= src_values[i]) / len(src_flat)
    
    # Create a mapping from source to reference quantiles
    interp_values = np.interp(src_quantiles, 
                             np.linspace(0, 1, len(ref_flat)),
                             np.sort(ref_flat))
    
    # Map each pixel in source image
    matched_flat = interp_values[src_indices]
    
    # Reshape back to original image dimensions
    return matched_flat.reshape(source_image.shape)


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
    plt.figure(figsize=(10, 20)) # Changed figure size for vertical layout
    # Display clean image (top)
    plt.subplot(2, 1, 1)
    plt.imshow(clean, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')
    # Display noisy image (bottom)
    plt.subplot(2, 1, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pair_{index:03d}.png'), dpi=150)
    plt.close()


def load_real_image_patches(csv_path="noise_characterization_patch_logs.csv", data_dir="data"):
    """
    Load patches from real Cassini images for histogram reference.
    
    Parameters:
        csv_path (str): Path to the CSV file with patch coordinates
        data_dir (str): Directory containing the Cassini data
        
    Returns:
        list: List of image patches from real Cassini images
    """
    from pds import parse_header, read_image
    
    patches = []
    
    try:
        # Read the CSV file
        import csv
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
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
                    except Exception as e:
                        print(f"Error loading patch from {filename}: {e}")
            
    except Exception as e:
        print(f"Error loading real image patches: {e}")
    
    return patches


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


def process_simulation(input_path, output_dir, bin_size=5, noise_std=0.001224):
    """
    Process a simulation image according to the adaptation pipeline.
    
    Parameters:
        input_path (str): Path to the input simulation image
        output_dir (str): Output directory for processed image pairs
        bin_size (int): Size of bins for downscaling
        noise_std (float): Noise standard deviation
    """
    # Use mean RMS contrast from contrast_analysis.py
    target_contrast = 0.060011  # Mean RMS contrast from contrast_analysis.py
    print(f"Target RMS contrast: {target_contrast:.6f}")
    
    # 2. Load real image patches for histogram matching
    print("Loading real image patches for histogram matching...")
    real_patches = load_real_image_patches()
    real_patches_array = None # Initialize to None
    if not real_patches:
        print("Warning: No real image patches loaded for histogram matching.")
    else:
        # Combine all patches into a single array for histogram reference
        real_patches_array = np.concatenate([patch.flatten() for patch in real_patches])
        print(f"Loaded {len(real_patches)} patches, total of {len(real_patches_array)} pixels for histogram reference.")
    
    # 3. Load the simulation image
    try:
        raw_image = np.array(Image.open(input_path).convert('F'))
    except Exception as e:
        print(f"Error loading image {input_path}: {e}")
        return
    
    print(f"Loaded image with shape: {raw_image.shape}")
    
    # 4. Downscale the image
    print(f"Downscaling with bin size {bin_size}...")
    downscaled_image = downscale_image(raw_image, bin_size)
    print(f"Downscaled image shape: {downscaled_image.shape}")
    
    # 5. Process the full downscaled image
    print("Processing full downscaled image...")
    
    processed_image = downscaled_image
    
    # Match histogram if we have real patches
    if real_patches_array is not None:
        print("  Matching histogram...")
        processed_image = histogram_matching(processed_image, real_patches_array)
    
    # Match contrast
    print("  Matching contrast...")
    contrast_matched_image = match_contrast(processed_image, target_contrast)
    
    # Add noise
    print("  Adding noise...")
    noisy_image, clean_image = add_gaussian_noise(contrast_matched_image, noise_std)
    
    # Save the pair
    # Use a fixed index or derive from filename if multiple inputs are processed later
    # For now, using index 0 as main calls this once.
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_pair_filename = f"{base_filename}_adapted" 

    # Modify save_image_pair to accept a base filename instead of an index for clarity
    # Or, ensure the output_dir makes it unique if multiple simulations are run.
    # For now, we will save with a generic name "full_adapted_pair" and index 0.
    # If you plan to process multiple simulation images, this naming needs to be more dynamic.
    
    print(f"Saving processed image pair...")
    # Save the pair using index 0, assuming one main image processed by this script run
    save_image_pair(clean_image, noisy_image, output_dir, 0) 
    
    print(f"Full image processed and saved to {output_dir}")


def main():
    # Input and output paths
    input_path = os.path.join("StormAlley", "raw_pvort.png")
    output_dir = "simulation_dataset_full" # Changed output directory to avoid overwriting sectioned data
    
    # Process parameters
    bin_size = 5
    # section_size = 100  # Smaller section size to fit downscaled dimensions - REMOVED
    # stride = 50  # Smaller stride for more sections - REMOVED
    noise_std = 0.001224
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the processing pipeline
    process_simulation(input_path, output_dir, bin_size, noise_std)
    
    print("\nSummary of dataset creation:")
    print("----------------------------")
    print(f"Source image: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Downscale bin size: {bin_size}x{bin_size}")
    # print(f"Section size: {section_size}x{section_size}") # REMOVED
    # print(f"Section stride: {stride}") # REMOVED
    print(f"Noise standard deviation: {noise_std}")
    
    # Count the generated pairs
    # This will count pair_000.npz, pair_001.npz etc. We are saving only one with index 0.
    # Adjust if naming convention in save_image_pair changes.
    num_pairs = len(glob.glob(os.path.join(output_dir, "pair_000.npz"))) 
    print(f"Total image pairs generated: {num_pairs}")


if __name__ == "__main__":
    main() 