import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import for drawing circles
from scipy.fft import fft2, fftshift

def generate_gaussian_noise_image(shape, std_dev):
    """Generates a white Gaussian noise image."""
    return np.random.normal(0, std_dev, shape)

# def compute_psd(image):
#     """Computes the Power Spectral Density (PSD) of an image."""
#     f_transform = fft2(image)
#     f_transform_shifted = fftshift(f_transform)
#     psd = np.abs(f_transform_shifted)**2
#     return psd

def calculate_radial_psd(image_patch):
    """ 
    Calculates the 1D Power Spectral Density (PSD) by radially averaging the 2D PSD.
    Applies a Hanning window before FFT to reduce edge artifacts.
    Returns: freqs, psd_1d_mean
    """
    if image_patch is None or image_patch.size == 0:
        return np.array([]), np.array([])

    # Ensure it's a 2D patch
    if image_patch.ndim != 2:
        if image_patch.ndim == 3 and image_patch.shape[2] in [1, 3, 4]:
            image_patch = image_patch[:, :, 0]
        else:
            return np.array([]), np.array([])

    h, w = image_patch.shape

    # Apply a 2D Hanning window to reduce edge artifacts
    # Ensure h and w are positive before creating Hanning window
    if h <= 0 or w <= 0:
        return np.array([]), np.array([]) # Or handle error appropriately
        
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    patch_windowed = image_patch * window

    # Compute 2D PSD
    f_transform = np.fft.fft2(patch_windowed) # Use windowed patch
    f_transform_shifted = np.fft.fftshift(f_transform)
    psd_2d = np.abs(f_transform_shifted)**2

    # Get image dimensions and center (use original dimensions for PSD indexing if needed, but psd_2d.shape is fine here)
    center_x, center_y = psd_2d.shape[1] // 2, psd_2d.shape[0] // 2

    # Create a grid of coordinates
    y_idx, x_idx = np.indices(psd_2d.shape) # Changed variable names to avoid conflict if x,y are used later

    # Calculate radial distances from the center
    r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)

    max_r = np.sqrt(center_x**2 + center_y**2)
    num_bins = min(psd_2d.shape[0], psd_2d.shape[1]) // 2 
    num_bins = max(1, num_bins)

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
# 1. Load the clean image
try:
    data = np.load('simulation_dataset/pair_000.npz')
    clean_image = data['clean'] 
except FileNotFoundError:
    print("Error: 'simulation_dataset/pair_000.npz' not found. Make sure the path is correct.")
    exit()
except KeyError:
    print("Error: 'clean' key not found in 'pair_000.npz'. Check the file structure.")
    exit()


# 2. Generate Gaussian noise image
# noise_shape = clean_image.shape # Use the shape of the clean image
# noise_shape = (1024, 1024) # Use a much larger shape for the noise image
noise_shape = clean_image.shape # Use the same shape as clean image for proper combination
noise_std_dev = 0.0012
noise_image = generate_gaussian_noise_image(noise_shape, noise_std_dev)

# 3. Create noisy image by combining clean image and noise
noisy_image = clean_image + noise_image

# # 3. Compute PSDs (Now done within calculate_radial_psd)
# psd_clean = compute_psd(clean_image)
# psd_noise = compute_psd(noise_image)

# 4. Compute Radially Averaged PSDs
radii_clean, rap_clean = calculate_radial_psd(clean_image) 
radii_noise, rap_noise = calculate_radial_psd(noise_image)
radii_noisy, rap_noisy = calculate_radial_psd(noisy_image)


# 5. Plotting
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Radially Averaged PSD Visualization', fontsize=16)

# Clean Image Column
axs[0, 0].imshow(clean_image, cmap='gray')
axs[0, 0].set_title('Clean Image')
axs[0, 0].axis('off')

_psd_clean_for_plot = np.abs(fftshift(fft2(clean_image)))**2
axs[0, 1].imshow(np.log10(_psd_clean_for_plot + 1e-9), cmap='viridis') 
axs[0, 1].set_title('PSD of Clean Image (log scale)')
axs[0, 1].axis('off')

# Add concentric circles to clean PSD plot
h_clean, w_clean = _psd_clean_for_plot.shape
center_x_clean, center_y_clean = w_clean // 2, h_clean // 2
max_radius_clean = min(center_x_clean, center_y_clean)
circle_radii_clean = [max_radius_clean * r for r in [0.15, 0.3, 0.45, 0.6, 0.75]]
for radius in circle_radii_clean:
    circle = patches.Circle((center_x_clean, center_y_clean), radius, fill=False, edgecolor='white', linestyle='--', alpha=0.7, linewidth=1.5)
    axs[0, 1].add_patch(circle)

axs[0, 2].plot(radii_clean, rap_clean, label='Clean Image PSD', color='blue')
axs[0, 2].set_title('Radially Averaged PSD (Clean)')
axs[0, 2].set_xlabel('Spatial Frequency (cycles/pixel)')
axs[0, 2].set_ylabel('Avg. Power (log)')
axs[0, 2].set_yscale('log')
axs[0, 2].set_xlim(0, 0.55)
axs[0, 2].grid(True, linestyle='--', alpha=0.7)
axs[0, 2].legend()


# Noise Image Column
axs[1, 0].imshow(noise_image, cmap='gray')
axs[1, 0].set_title(f'Gaussian Noise (std={noise_std_dev})')
axs[1, 0].axis('off')

_psd_noise_for_plot = np.abs(fftshift(fft2(noise_image)))**2
axs[1, 1].imshow(np.log10(_psd_noise_for_plot + 1e-9), cmap='viridis') 
axs[1, 1].set_title('PSD of Noise Image (log scale)')
axs[1, 1].axis('off')

# Add concentric circles to noise PSD plot
h_noise, w_noise = _psd_noise_for_plot.shape
center_x_noise, center_y_noise = w_noise // 2, h_noise // 2
max_radius_noise = min(center_x_noise, center_y_noise)
circle_radii_noise = [max_radius_noise * r for r in [0.15, 0.3, 0.45, 0.6, 0.75]]
for radius in circle_radii_noise:
    circle = patches.Circle((center_x_noise, center_y_noise), radius, fill=False, edgecolor='white', linestyle='--', alpha=0.7, linewidth=1.5)
    axs[1, 1].add_patch(circle)

axs[1, 2].plot(radii_noise, rap_noise, label='Noise Image PSD', color='red')
axs[1, 2].set_title('Radially Averaged PSD (Noise)')
axs[1, 2].set_xlabel('Spatial Frequency (cycles/pixel)')
axs[1, 2].set_ylabel('Avg. Power (log)')
axs[1, 2].set_yscale('log')
axs[1, 2].set_xlim(0, 0.55)
axs[1, 2].grid(True, linestyle='--', alpha=0.7)
axs[1, 2].legend()


# Noisy Image Column (Clean + Noise)
axs[2, 0].imshow(noisy_image, cmap='gray')
axs[2, 0].set_title('Noisy Image (Clean + Noise)')
axs[2, 0].axis('off')

_psd_noisy_for_plot = np.abs(fftshift(fft2(noisy_image)))**2
axs[2, 1].imshow(np.log10(_psd_noisy_for_plot + 1e-9), cmap='viridis') 
axs[2, 1].set_title('PSD of Noisy Image (log scale)')
axs[2, 1].axis('off')

# Add concentric circles to noisy PSD plot
h_noisy, w_noisy = _psd_noisy_for_plot.shape
center_x_noisy, center_y_noisy = w_noisy // 2, h_noisy // 2
max_radius_noisy = min(center_x_noisy, center_y_noisy)
circle_radii_noisy = [max_radius_noisy * r for r in [0.15, 0.3, 0.45, 0.6, 0.75]]
for radius in circle_radii_noisy:
    circle = patches.Circle((center_x_noisy, center_y_noisy), radius, fill=False, edgecolor='white', linestyle='--', alpha=0.7, linewidth=1.5)
    axs[2, 1].add_patch(circle)

# Combined Radially Averaged PSDs comparison
axs[2, 2].plot(radii_clean, rap_clean, label='Clean Image', color='blue', linewidth=2)
axs[2, 2].plot(radii_noise, rap_noise, label='Noise Only', color='red', linewidth=2)
axs[2, 2].plot(radii_noisy, rap_noisy, label='Noisy Image (Clean + Noise)', color='purple', linewidth=2)
axs[2, 2].set_title('Radially Averaged PSD Comparison')
axs[2, 2].set_xlabel('Spatial Frequency (cycles/pixel)')
axs[2, 2].set_ylabel('Avg. Power (log)')
axs[2, 2].set_yscale('log')
axs[2, 2].set_xlim(0, 0.55)
axs[2, 2].grid(True, linestyle='--', alpha=0.7)
axs[2, 2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

# 6. Save the plot
output_filename = 'radially_averaged_psd_explanation.png'
plt.savefig(output_filename)
print(f"Plot saved as {output_filename}")

plt.show() 