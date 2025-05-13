import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

def bm3d_denoise(img, sigma=None):
    """
    Apply a non-local means denoising algorithm as an improved
    alternative to BM3D for a 2D image.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array)
    sigma : float or None
        Noise standard deviation. If None, it will be estimated.
        
    Returns:
    --------
    denoised_img : ndarray
        Denoised image
    """
    # Convert to float if needed
    img_float = np.asarray(img, dtype=np.float64)
    
    # Estimate noise if not provided
    if sigma is None:
        # Estimate the noise standard deviation from the image
        sigma_est = np.mean(estimate_sigma(img_float))
        sigma = max(0.01, sigma_est)  # Ensure positive sigma
    
    # Apply Non-Local Means denoising
    # This is a good alternative to BM3D for many applications
    h = 0.8 * sigma  # filtering parameter
    patch_size = 5   # 5×5 patches
    patch_distance = 6  # 13×13 search area
    
    denoised = denoise_nl_means(
        img_float,
        h=h,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        preserve_range=True
    )
    
    # Ensure output has same range as input
    return np.clip(denoised, np.min(img), np.max(img))
