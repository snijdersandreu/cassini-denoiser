# Import external modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from scipy.stats import norm, skew, kurtosis, kstest, probplot


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
def estimate_snr_lapshenkov(image_2d):
    """
    Estimate SNR using Lapshenkov & Anikeeva's harmonic analysis method.
    This method estimates signal and noise RMS by fitting a 2D trend and
    analyzing residuals.

    Parameters:
        image_2d (ndarray): A 2D float numpy array representing the image.

    Returns:
        dict: Dictionary with 'snr_linear', 'snr_db', 'signal_rms', 'noise_rms'
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
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = image_2d.ravel()

    p0 = [0, 0, 0, 1, 1, 1]  # initial guess
    try:
        popt, _ = curve_fit(harmonic_model, (X_flat, Y_flat), Z_flat, p0=p0, maxfev=5000)
        trend = harmonic_model((X, Y), *popt)
        residuals = image_2d - trend

        signal_rms = np.sqrt(np.mean(trend ** 2))
        noise_rms = np.sqrt(np.mean(residuals ** 2))

        snr_linear = signal_rms / noise_rms if noise_rms != 0 else np.inf
        snr_db = 20 * np.log10(snr_linear) if snr_linear != 0 else -np.inf

        return {
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "signal_rms": signal_rms,
            "noise_rms": noise_rms
        }
    except Exception as e:
        print("SNR estimation failed:", e)
        return {
            "snr_linear": None,
            "snr_db": None,
            "signal_rms": None,
            "noise_rms": None
        }