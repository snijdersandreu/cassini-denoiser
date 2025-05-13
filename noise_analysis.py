# Import external modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from PIL import Image
from scipy.stats import norm, skew, kurtosis, kstest
from scipy.signal import wiener



def linear_plane(coords, a, b, c):
    """
    Defines a linear plane function for fitting.

    Parameters:
        coords (tuple): Tuple of (X, Y) coordinate arrays.
        a (float): Coefficients of the plane.
        b (float): Coefficients of the plane.
        c (float): Coefficients of the plane.

    Returns:
        Z (ndarray): Calculated Z values for the plane at given X, Y.
    """
    X, Y = coords
    return a * X + b * Y + c


def fit_trend(surface):
    """
    Fits a linear plane to the given 2D surface to model illumination trend.

    Parameters:
        surface (ndarray): 2D array representing the image sub-region.

    Returns:
        trend (ndarray): Fitted linear trend surface.
    """
    n_rows, n_cols = surface.shape
    X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = surface.ravel()

    # Initial guess for the parameters
    initial_guess = (0, 0, np.mean(Z_flat))

    # Fit the linear plane
    # noinspection PyTupleAssignmentBalance
    popt, _ = curve_fit(linear_plane, (X_flat, Y_flat), Z_flat, p0=initial_guess)

    # Generate the fitted trend surface
    trend = linear_plane((X, Y), *popt)
    return trend


def estimate_noise(sub_region):
    trend = fit_trend(sub_region)
    residuals = sub_region - trend
    residuals = residuals.astype(np.float32)
    residuals_streched = residuals.ravel()  

    noise_mean = np.mean(residuals_streched)
    noise_std = np.std(residuals_streched)
    noise_skewness = skew(residuals_streched, bias=False)
    noise_kurtosis = kurtosis(residuals_streched, fisher=False)
    noise_ks_stat, noise_ks_pval = kstest((residuals_streched - noise_mean) / (noise_std + 1e-12), 'norm')
    return noise_mean, noise_std, residuals, noise_skewness, noise_kurtosis, noise_ks_stat, noise_ks_pval, trend



# Lapshenkov & Anikeeva SNR estimation by harmonic analysis
def estimate_snr_lapshenkov(image_2d, reference_trend=None):
    """
    Estimate SNR using Lapshenkov & Anikeeva's harmonic analysis method.
    This method estimates signal and noise RMS by fitting a 2D trend and
    analyzing residuals.

    Parameters:
        image_2d (ndarray): A 2D float numpy array representing the image.
        reference_trend (ndarray, optional): Pre-computed signal trend to use instead
                                          of estimating a new one. When provided,
                                          this trend will be used to calculate residuals.

    Returns:
        dict: Dictionary with 'snr_linear', 'snr_db', 'signal_rms', 'noise_rms', 'trend'
              The 'trend' key contains the estimated signal component, useful for
              reusing in subsequent calls for consistent SNR comparisons.
    """
    if image_2d.ndim != 2:
        raise ValueError("Input must be a 2D image.")

    from scipy.optimize import curve_fit

    def harmonic_model(coords, a, b, c, d, e, f):
        X, Y = coords
        return (a + b * X + c * Y +
                d * np.cos(2 * np.pi * X / X.max()) +
                e * np.sin(2 * np.pi * Y / Y.max()) +
                f * np.cos(2 * np.pi * (X + Y) / max(X.max(), Y.max())))

    h, w = image_2d.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    try:
        # If reference_trend is provided, use it as the signal component
        if reference_trend is not None:
            if reference_trend.shape != image_2d.shape:
                raise ValueError("Reference trend must have the same shape as the input image")
            trend = reference_trend
        else:
            # Otherwise, fit the model to estimate the signal component
            X_flat = X.ravel()
            Y_flat = Y.ravel()
            Z_flat = image_2d.ravel()
            p0 = [0, 0, 0, 1, 1, 1]  # initial guess
            popt, _ = curve_fit(harmonic_model, (X_flat, Y_flat), Z_flat, p0=p0, maxfev=5000)
            trend = harmonic_model((X, Y), *popt)
        
        # Calculate residuals and metrics
        residuals = image_2d - trend
        signal_rms = np.sqrt(np.mean(trend ** 2))
        noise_rms = np.sqrt(np.mean(residuals ** 2))

        snr_linear = signal_rms / noise_rms if noise_rms != 0 else np.inf
        snr_db = 20 * np.log10(snr_linear) if snr_linear != 0 else -np.inf

        return {
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "signal_rms": signal_rms,
            "noise_rms": noise_rms,
            "trend": trend  # Return the trend so it can be reused
        }
    except Exception as e:
        print("SNR estimation failed:", e)
        return {
            "snr_linear": None,
            "snr_db": None,
            "signal_rms": None,
            "noise_rms": None,
            "trend": None
        }
        
def estimate_snr_psd(image_2d):
    """
    Estimates Signal-to-Noise Ratio (SNR) using a PSD-based blind method.

    This method is based on noise-floor subtraction in the frequency domain,
    as described in texts like Gonzalez & Woods ("Digital Image Processing")
    and Kay ("Fundamentals of Statistical Signal Processing").

    Methodology:
    1. Compute the 2D Fourier Transform and then the Power Spectral Density (PSD).
    2. Estimate the noise floor (S_n_hat) from a high-frequency annulus in the PSD.
       The annulus is defined by frequencies >= 0.8 * (0.5 * Nyquist_radius).
    3. Subtract the noise floor from the PSD to recover the signal PSD (S_s_uv).
    4. Integrate signal and noise power to compute SNR:
       P_s = sum(S_s_uv)
       P_n = S_n_hat * M * N
       SNR_linear = P_s / P_n

    Parameters:
        image_2d (np.ndarray): 2D numpy array representing the image.

    Returns:
        dict: A dictionary containing:
            'snr_linear' (float): The linear SNR (P_s / P_n).
            'snr_db' (float): The SNR in decibels (10 * log10(snr_linear)).
            'signal_rms' (float): Estimated RMS of the signal component,
                                  sqrt(sum(S_s_uv) / (M*N)).
            'noise_rms' (float): Estimated RMS of the noise component, sqrt(S_n_hat).
            'P_s' (float): Total estimated signal power (sum(S_s_uv)).
            'P_n' (float): Total estimated noise power (S_n_hat * M * N).
    """
    if not isinstance(image_2d, np.ndarray) or image_2d.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    M, N = image_2d.shape
    if M < 2 or N < 2: # Check for minimal dimensions for meaningful PSD analysis
        return {
            "snr_linear": np.nan, "snr_db": np.nan,
            "signal_rms": np.nan, "noise_rms": np.nan,
            "P_s": np.nan, "P_n": np.nan
        }

    # 1. 2D Fourier Transform and Periodogram
    F = np.fft.fft2(image_2d)
    F_shifted = np.fft.fftshift(F)
    # Raw power spectral density (periodogram)
    P_uv = np.abs(F_shifted)**2

    # 2. Noise-Floor Estimation
    # Create frequency coordinates (indices for the shifted PSD)
    ky_indices, kx_indices = np.indices((M, N))
    center_y, center_x = M // 2, N // 2
    
    # Distance from center in terms of indices
    dist_from_center = np.sqrt((ky_indices - center_y)**2 + (kx_indices - center_x)**2)
    
    # Nyquist radius in terms of pixel indices from center
    # For a dimension D, max frequency is at D/2 index from center.
    nyquist_radius_pixels = min(M / 2.0, N / 2.0)
    
    # Threshold frequency for annulus (0.8 * half the Nyquist radius)
    f_th_pixels = 0.4 * nyquist_radius_pixels # 0.8 * 0.5 * nyquist_radius_pixels
    
    # Select pixels in the high-frequency annulus
    annulus_mask = dist_from_center >= f_th_pixels
    
    noise_psd_values_in_annulus = P_uv[annulus_mask]
    
    if noise_psd_values_in_annulus.size == 0:
        # Fallback if annulus is empty (e.g., very small image or f_th_pixels too large)
        # This might also happen if high-frequency content is exactly zero.
        # Assume noise floor is 0 in such cases, or a very small number to avoid division by zero.
        S_n_hat = 0.0
    else:
        S_n_hat = np.median(noise_psd_values_in_annulus)

    # 3. Signal PSD Recovery
    # S_s_uv = max{P(u,v) - S_n_hat, 0}
    S_s_uv = np.maximum(P_uv - S_n_hat, 0)

    # 4. Power Integration (as per user's formulas in the description)
    # P_s = sum over u,v of S_s_uv
    P_s_total = np.sum(S_s_uv)
    # P_n = S_n_hat * (M*N)
    P_n_total = S_n_hat * M * N

    # SNR Computation
    if P_n_total < 1e-12: # Effectively zero noise power
        if P_s_total > 1e-12: # Effectively non-zero signal power
            snr_linear = np.inf
        else: # Both signal and noise are effectively zero
            snr_linear = np.nan # Or 0 or 1, NaN indicates undefined 0/0
    else:
        snr_linear = P_s_total / P_n_total

    # SNR in dB
    if np.isnan(snr_linear):
        snr_db = np.nan
    elif np.isinf(snr_linear):
        snr_db = np.inf
    elif snr_linear == 0: # Handles P_s_total = 0, P_n_total > 0
        snr_db = -np.inf 
    elif snr_linear < 0: # Should not happen if S_s_uv >= 0 and S_n_hat >=0
        snr_db = np.nan # Undefined for negative linear SNR
    else: # snr_linear > 0 and finite
        snr_db = 10 * np.log10(snr_linear)

    # RMS values calculation:
    # Average signal power per pixel = P_s_total / (M*N)
    # Average noise power per pixel = S_n_hat (noise floor estimate per frequency component)
    avg_signal_power = P_s_total / (M * N)
    avg_noise_power = S_n_hat # S_n_hat is already average power if PSD is scaled as |F|^2

    signal_rms = np.sqrt(avg_signal_power) if avg_signal_power >= 0 else np.nan
    noise_rms = np.sqrt(avg_noise_power) if avg_noise_power >= 0 else np.nan

    return {
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "signal_rms": signal_rms,
        "noise_rms": noise_rms,
        "P_s": P_s_total,
        "P_n": P_n_total
    }

def apply_denoising(image, sigma=None, method="wiener"):
    """
    Apply denoising to an image using the specified method.

    Parameters:
        image (ndarray): Input 2D image array.
        sigma (float, optional): Estimated noise standard deviation. Used by some methods.
        method (str): Denoising method to use. One of "wiener", "starlet", "bm3d", "unet-self2self".

    Returns:
        ndarray: Denoised image.

    Raises:
        NotImplementedError: If the selected method is not implemented.
        ValueError: If the method is not recognized.
    """
    if method == "wiener":
        from scipy.signal import wiener
        if sigma is not None:
            return wiener(image, noise=sigma**2)
        else:
            return wiener(image)
    elif method == "starlet":
        raise NotImplementedError("Starlet denoising is not implemented yet.")
    elif method == "bm3d":
        raise NotImplementedError("BM3D denoising is not implemented yet.")
    elif method == "unet-self2self":
        raise NotImplementedError("U-Net Self2Self denoising is not implemented yet.")
    else:
        raise ValueError(f"Unknown denoising method: {method}")

def calculate_snr_with_ground_truth(denoised_image, clean_image):
    """
    Calculate Signal-to-Noise Ratio (SNR) using a clean ground truth image as reference.
    
    This calculates:
    1. MSE (Mean Squared Error) between input image and ground truth
    2. RMSE (Root Mean Squared Error)
    3. PSNR (Peak Signal-to-Noise Ratio)
    4. SNR (Signal-to-Noise Ratio in dB)
    
    Parameters:
        denoised_image (ndarray): The input image (could be denoised or noisy)
        clean_image (ndarray): The ground truth clean image
        
    Returns:
        dict: Dictionary containing 'mse', 'rmse', 'psnr', 'snr_db'
    """
    if denoised_image.shape != clean_image.shape:
        raise ValueError("Input and ground truth images must have the same shape")
    
    # Calculate noise as the difference between the input image and clean reference
    noise = denoised_image - clean_image
    
    # Calculate error metrics
    mse = np.mean(noise ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate signal power (from ground truth)
    signal_power = np.mean(clean_image ** 2)
    
    # Calculate noise power from the extracted noise
    noise_power = np.mean(noise ** 2)  # Same as mse
    
    # Calculate PSNR (using max value of clean image as peak)
    data_range = np.max(clean_image) - np.min(clean_image)
    if data_range < 1e-10 or rmse < 1e-10:  # Avoid division by zero
        psnr = np.inf
    else:
        psnr = 20 * np.log10(data_range / rmse)
    
    # Calculate SNR in dB
    if noise_power < 1e-10:  # Avoid division by zero
        snr_db = np.inf
    else:
        snr_db = 10 * np.log10(signal_power / noise_power)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "snr_db": snr_db
    }