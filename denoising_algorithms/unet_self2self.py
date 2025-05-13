import numpy as np
from scipy.ndimage import gaussian_filter

def unet_self2self_denoise(img, sigma=0.03):
    """
    Apply UNET-Self2Self denoising algorithm to a 2D image.
    This is a simplified placeholder implementation since the actual
    UNET-Self2Self requires training a neural network.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D numpy array)
    sigma : float
        Noise level for smoothing
        
    Returns:
    --------
    denoised_img : ndarray
        Denoised image
    """
    # This is a simple placeholder using edge-preserving smoothing
    # Real UNET-Self2Self requires a trained neural network
    
    # Make sure image is float
    img = np.asarray(img, dtype=np.float64)
    
    # Apply bilateral-like filtering (simplified)
    smoothed = gaussian_filter(img, sigma=sigma)
    
    # Edge preservation approximation
    edges = np.abs(img - smoothed)
    edge_weight = np.exp(-edges/0.1)
    
    # Combine smoothed image and original based on edge weights
    result = edge_weight * smoothed + (1-edge_weight) * img
    
    # Ensure output range is the same as input
    return np.clip(result, np.min(img), np.max(img))
