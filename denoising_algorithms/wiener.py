import numpy as np
from scipy import fftpack
from scipy.ndimage import gaussian_filter

def wiener_filter(img, kernel_size=3, noise_std=0.01):
    """
    Apply Wiener filter to a 2D image.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array)
    kernel_size : int
        Size of the local window
    noise_std : float
        Noise standard deviation
        
    Returns:
    --------
    filtered_img : ndarray
        Filtered image
    """
    # Make sure image is float
    img = np.asarray(img, dtype=np.float64)
    
    # Calculate local mean
    local_mean = gaussian_filter(img, kernel_size/2)
    
    # Calculate local variance
    local_var = gaussian_filter(img**2, kernel_size/2) - local_mean**2
    
    # Ensure local_var is positive
    local_var = np.maximum(local_var, 1e-10)
    
    # Calculate noise variance
    noise_var = noise_std**2
    
    # Apply Wiener filter formula
    filtered_img = local_mean + ((local_var - noise_var) / local_var) * (img - local_mean)
    
    # Clip to ensure values are within the same range as the input
    return np.clip(filtered_img, np.min(img), np.max(img)) 