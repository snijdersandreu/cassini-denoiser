import numpy as np
from scipy.ndimage import convolve1d

def apply_starlet_denoising(image, n_scales=4, k=3, sigma=None):
    """
    Perform Starlet (Isotropic Undecimated Wavelet Transform) denoising on a 2D image.

    This function implements the denoising algorithm described by Starck, Murtagh, and Bertero,
    using a B3-spline scaling function and hard thresholding of detail coefficients.

    Parameters
    ----------
    image : np.ndarray
        2D input image (float or convertible to float).
    n_scales : int, optional
        Number of wavelet scales to use (default: 4). More scales can better separate structures.
    k : float, optional
        Threshold multiplier for noise standard deviation (default: 3).
    sigma : float or None, optional
        Known noise standard deviation. If provided, internal noise estimation is skipped and
        the threshold is computed using this value (default: None).

    Returns
    -------
    denoised : np.ndarray
        The denoised 2D image, same shape as input.

    Process
    -------
    1. Decompose the image into n_scales detail images (wavelet coefficients) and a coarse residual
       using the B3-spline filter [1, 4, 6, 4, 1]/16 with separable convolution.
    2. Estimate the noise sigma from the finest scale detail coefficients (robust MAD estimator),
       or use the provided sigma.
    3. Apply hard thresholding to detail coefficients at each scale, using k * sigma.
    4. Reconstruct the denoised image by summing the thresholded details and final coarse scale.
    """
    image = np.asarray(image, dtype=np.float64)
    h = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    # Store the smoothed images at each scale (c_j)
    c = [image]
    for j in range(n_scales):
        # Convolve rows then columns (separable)
        smooth = convolve1d(c[-1], h, axis=0, mode='mirror')
        smooth = convolve1d(smooth, h, axis=1, mode='mirror')
        c.append(smooth)
    # Detail coefficients: w_j = c_{j} - c_{j+1}
    details = [c[j] - c[j+1] for j in range(n_scales)]
    # Use provided sigma or estimate noise sigma
    if sigma is not None:
        noise_sigma = sigma
    else:
        w0 = details[0]
        noise_sigma = np.median(np.abs(w0)) / 0.6745
    # Threshold and reconstruct
    thresholded_details = []
    for w in details:
        w_thr = np.where(np.abs(w) > k * noise_sigma, w, 0)
        thresholded_details.append(w_thr)
    denoised = np.sum(thresholded_details, axis=0) + c[-1]
    return denoised