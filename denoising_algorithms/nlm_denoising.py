import numpy as np

def nlm_denoise(image, patch_size, patch_distance, h, progress_callback=None, debug=False):
    """
    Applies Non-Local Means denoising to an image.

    Args:
        image (np.ndarray): The input noisy image.
        patch_size (int): The size of the patches to compare (e.g., 7 for a 7x7 patch).
        patch_distance (int): The radius of the search window for similar patches (e.g., 10).
        h (float): The degree of filtering, which controls the decay of weights as a function of Euclidean distances.
        progress_callback (function, optional): A function to call with progress updates (percentage).
        debug (bool, optional): If True, prints debugging information.

    Returns:
        np.ndarray: The denoised image.
    """
    if debug:
        print(f"[NLM DEBUG] Starting NLM denoise on image of shape {image.shape}")
        print(f"[NLM DEBUG] Parameters: patch_size={patch_size}, patch_distance={patch_distance}, h={h}")

    # Pad the image to handle borders
    pad_width = patch_size // 2 + patch_distance
    padded_image = np.pad(image, pad_width, mode='reflect')
    denoised_image = np.zeros_like(image)
    
    # Precompute h*h for efficiency
    h2 = h * h

    total_rows = image.shape[0]
    # Iterate over each pixel in the original image
    for r_idx in range(total_rows):
        if debug:
            print(f"[NLM DEBUG] Processing row {r_idx + 1}/{total_rows}")
        for c_idx in range(image.shape[1]):
            # Current pixel's actual coordinates in the padded image
            r_padded = r_idx + pad_width
            c_padded = c_idx + pad_width

            # Extract the reference patch centered at (r_padded, c_padded)
            # The patch itself has dimensions patch_size x patch_size
            # We need to account for the patch_size // 2 offset to get the top-left corner
            patch_r_start = r_padded - (patch_size // 2)
            patch_r_end = r_padded + (patch_size // 2) + 1
            patch_c_start = c_padded - (patch_size // 2)
            patch_c_end = c_padded + (patch_size // 2) + 1
            reference_patch = padded_image[patch_r_start:patch_r_end, patch_c_start:patch_c_end]

            # Initialize variables for weighted sum and normalization factor
            weighted_sum = 0.0
            normalization_factor = 0.0

            # Iterate over the search window
            # The search window is centered at (r_padded, c_padded) and has radius patch_distance
            search_win_r_start = r_padded - patch_distance
            search_win_r_end = r_padded + patch_distance + 1
            search_win_c_start = c_padded - patch_distance
            search_win_c_end = c_padded + patch_distance + 1
            
            for search_r in range(search_win_r_start, search_win_r_end):
                for search_c in range(search_win_c_start, search_win_c_end):
                    # Avoid comparing the patch with itself in a trivial way (though it's usually included)
                    # if search_r == r_padded and search_c == c_padded:
                    #     continue

                    # Extract the comparison patch centered at (search_r, search_c)
                    comp_patch_r_start = search_r - (patch_size // 2)
                    comp_patch_r_end = search_r + (patch_size // 2) + 1
                    comp_patch_c_start = search_c - (patch_size // 2)
                    comp_patch_c_end = search_c + (patch_size // 2) + 1
                    comparison_patch = padded_image[comp_patch_r_start:comp_patch_r_end, comp_patch_c_start:comp_patch_c_end]
                    
                    # Calculate squared Euclidean distance between patches
                    # Ensure patches are of the same size, which they should be by construction
                    if reference_patch.shape != comparison_patch.shape:
                        # This should ideally not happen with correct padding and windowing
                        if debug:
                            print(f"[NLM DEBUG] Patch shape mismatch: ref {reference_patch.shape}, comp {comparison_patch.shape} at pixel ({r_idx},{c_idx}) for search ({search_r},{search_c})")
                        continue 
                    
                    dist_sq = np.sum((reference_patch - comparison_patch)**2)
                    
                    # Calculate weight
                    weight = np.exp(-dist_sq / h2)
                    
                    # Accumulate
                    normalization_factor += weight
                    weighted_sum += weight * padded_image[search_r, search_c] # Use the central pixel of the search patch

            # Compute the denoised pixel value
            if normalization_factor > 0:
                denoised_image[r_idx, c_idx] = weighted_sum / normalization_factor
            else:
                # Should not happen in practice with a reflective padding and h > 0
                # but as a fallback, keep the original pixel value
                if debug:
                    print(f"[NLM DEBUG] Normalization factor is zero for pixel ({r_idx},{c_idx}). Using original value.")
                denoised_image[r_idx, c_idx] = image[r_idx, c_idx]
        
        if progress_callback:
            progress_percentage = ((r_idx + 1) / total_rows) * 100
            progress_callback(progress_percentage)
                
    if debug:
        print("[NLM DEBUG] NLM denoise finished.")
    return denoised_image

if __name__ == '__main__':
    # Example Usage (assuming you have an image to test)
    # from skimage.util import random_noise
    # from skimage import img_as_float, io
    # import matplotlib.pyplot as plt

    # def example_progress_update(percentage):
    #     print(f"NLM Progress: {percentage:.2f}%")

    # # Load a sample image (e.g., skimage's camera)
    # try:
    #     original_image = img_as_float(io.imread('path_to_your_image.png', as_gray=True)) # Replace with your image
    # except FileNotFoundError:
    #     print("Test image not found. Using a synthetic image.")
    #     original_image = np.zeros((64,64)) # Smaller for faster testing
    #     original_image[16:48, 16:48] = 0.5
    #     original_image[24:40, 24:40] = 1.0


    # # Add some noise
    # noisy_image = random_noise(original_image, mode='gaussian', var=0.01)

    # # Denoise
    # patch_s = 5  # Patch size
    # patch_d = 7 # Search window radius (patch_distance)
    # h_param = 0.05 # Denoising parameter

    # print(f"Denoising with patch_size={patch_s}, patch_distance={patch_d}, h={h_param}...")
    # denoised_nlm = nlm_denoise(noisy_image, patch_size=patch_s, patch_distance=patch_d, h=h_param, progress_callback=example_progress_update, debug=True)
    # print("Denoising complete.")

    # # Display results
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
    # ax[0].set_title('Original Image')

    # ax[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
    # ax[1].set_title('Noisy Image')
    
    # ax[2].imshow(denoised_nlm, cmap='gray', vmin=0, vmax=1)
    # ax[2].set_title(f'NLM Denoised (h={h_param})')

    # for a in ax:
    #     a.axis('off')

    # plt.tight_layout()
    # plt.show()
    pass # Placeholder for example usage 