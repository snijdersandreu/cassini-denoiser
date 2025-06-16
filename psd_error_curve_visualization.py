import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.fft import fft2, fftshift

# --- Utility functions ---
def generate_gaussian_noise_image(shape, std_dev):
    return np.random.normal(0, std_dev, shape)

def calculate_radial_psd(image_patch, num_bins):
    """
    Calculates the 1D Power Spectral Density (PSD) by radially averaging the 2D PSD.
    """
    if image_patch is None or image_patch.size == 0:
        return np.array([]), np.array([])
    if image_patch.ndim != 2:
        if image_patch.ndim == 3 and image_patch.shape[2] in [1, 3, 4]:
            image_patch = image_patch[:, :, 0]
        else:
            return np.array([]), np.array([])
            
    h, w = image_patch.shape
    if h <= 0 or w <= 0:
        return np.array([]), np.array([])
        
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    patch_windowed = image_patch * window
    
    f_transform = np.fft.fft2(patch_windowed)
    f_transform_shifted = np.fft.fftshift(f_transform)
    psd_2d = np.abs(f_transform_shifted)**2
    
    center_x, center_y = psd_2d.shape[1] // 2, psd_2d.shape[0] // 2
    y_idx, x_idx = np.indices(psd_2d.shape)
    r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
    max_r = np.sqrt(center_x**2 + center_y**2)
    
    radial_bins = np.linspace(0, max_r, num_bins + 1)
    psd_sum_in_bin = np.zeros(num_bins)
    pixels_in_bin = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
        psd_sum_in_bin[i] = np.sum(psd_2d[mask])
        pixels_in_bin[i] = np.sum(mask)
        
    psd_1d_mean = np.zeros_like(psd_sum_in_bin)
    non_empty_bins = pixels_in_bin > 0
    psd_1d_mean[non_empty_bins] = psd_sum_in_bin[non_empty_bins] / pixels_in_bin[non_empty_bins]
    
    if num_bins > 0:
        spatial_freqs = np.arange(num_bins) / (2.0 * num_bins)
    else:
        spatial_freqs = np.array([])
        
    return spatial_freqs, psd_1d_mean

# --- Main script ---
try:
    data = np.load('simulation_dataset/pair_000.npz')
    clean_image = data['clean']
except FileNotFoundError:
    print("Error: 'simulation_dataset/pair_000.npz' not found. Make sure the path is correct.")
    exit()
except KeyError:
    print("Error: 'clean' key not found in 'pair_000.npz'. Check the file structure.")
    exit()

noise_shape = clean_image.shape
noise_std_dev = 0.0012
noise_image = generate_gaussian_noise_image(noise_shape, noise_std_dev)
noisy_image = clean_image + noise_image

# Increase number of bins significantly
num_bins = min(clean_image.shape) // 2
radii_clean, rap_clean = calculate_radial_psd(clean_image, num_bins=num_bins)
radii_noisy, rap_noisy = calculate_radial_psd(noisy_image, num_bins=num_bins)
psd_error_curve = np.abs(rap_noisy - rap_clean)

# Plotting: 2 rows, 3 columns
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('PSD and PSD Error Curve Visualization', fontsize=16)

# Column 1: Images
# Left Top: Clean Image
axs[0, 0].imshow(clean_image, cmap='gray')
axs[0, 0].set_title('Clean Image')
axs[0, 0].axis('off')

# Left Bottom: Noisy Image
axs[1, 0].imshow(noisy_image, cmap='gray')
axs[1, 0].set_title('Noisy Image')
axs[1, 0].axis('off')

# Column 2: 2D PSDs
# Center Top: PSD of Clean image
_psd_clean_for_plot = np.abs(fftshift(fft2(clean_image)))**2
axs[0, 1].imshow(np.log10(_psd_clean_for_plot + 1e-9), cmap='viridis')
axs[0, 1].set_title('PSD of Clean Image (log scale)')
axs[0, 1].axis('off')

# Center Bottom: PSD of Noisy image
_psd_noisy_for_plot = np.abs(fftshift(fft2(noisy_image)))**2
axs[1, 1].imshow(np.log10(_psd_noisy_for_plot + 1e-9), cmap='viridis')
axs[1, 1].set_title('PSD of Noisy Image (log scale)')
axs[1, 1].axis('off')

# Column 3: 1D PSD plots
# Right Top: Radially Averaged PSDs
axs[0, 2].plot(radii_clean, rap_clean, label='Clean PSD', color='blue', linewidth=1.5, alpha=0.9)
axs[0, 2].plot(radii_noisy, rap_noisy, label='Noisy PSD', color='red', linewidth=1.5, alpha=0.8)
axs[0, 2].set_title('Radially Averaged PSDs')
axs[0, 2].set_xlabel('Spatial Frequency (cycles/pixel)')
axs[0, 2].set_ylabel('Avg. Power (log scale)')
axs[0, 2].set_yscale('log')
axs[0, 2].set_xlim(0, 0.55)
axs[0, 2].grid(True, linestyle='--', alpha=0.7)
axs[0, 2].legend()

# Right Bottom: PSD Error Curve
axs[1, 2].plot(radii_clean, psd_error_curve, label='PSD Error Curve (Noisy vs. Clean)', color='purple', linewidth=1.5)
axs[1, 2].set_title('PSD Error Curve')
axs[1, 2].set_xlabel('Spatial Frequency (cycles/pixel)')
axs[1, 2].set_ylabel('Avg. Power Difference (log scale)')
axs[1, 2].set_yscale('log')
axs[1, 2].set_xlim(0, 0.55)
axs[1, 2].grid(True, linestyle='--', alpha=0.7)
axs[1, 2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_filename = 'psd_error_curve_explanation.png'
plt.savefig(output_filename)
print(f"Plot saved as {output_filename}")
plt.show() 