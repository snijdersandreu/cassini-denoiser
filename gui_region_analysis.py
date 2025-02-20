# Import system modules
import argparse
import csv
import os
import sys
from pathlib import Path

# Import external modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RectangleSelector, RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm
from tkinter import filedialog, messagebox

# Import local modules
from noise_analysis import estimate_noise, denoise_image, high_contrast_image
from pds import read_image


class ImageRegionNoiseAnalysis:
    """
    A GUI tool for selecting regions in an image and performing noise analysis.
    """

    def __init__(self, image_array, output_folder):
        """
        Initializes the GUI with the provided image array.

        Args:
            image_array (np.ndarray): The grayscale image array of shape (rows, columns).
            output_folder (str): Directory where output results will be saved.
        """

        # User input parameters
        self.image_array = image_array  # Original image array
        self.output_folder = output_folder   # Directory where to store images

        # Create output folders if not exist
        os.makedirs(self.output_folder, exist_ok=True)

        # CSV to store noise analyses
        self.csv_path = os.path.join(self.output_folder, 'region_analysis.csv')

        # Disable Matplotlib internal save keypress event functionality
        plt.rcParams['keymap.save'] = ''  # Disable default 'S' key for save-as functionality

        # Store coordinates and analysis data of the last selected region
        self.selected_region = None
        self.selected_region_data = None

        # Initialize historical data storage
        self.historical_data = []  # List to store dictionaries of analyses

        # Initialize de-noising methods
        self.de_noise_methods = ['median', 'gaussian', 'tv_chambolle']

        # Initialize variables to hold insets
        self.inset_axes_left = None
        self.inset_axes_right = None

        # Create the figure and axes
        self.fig = plt.figure(figsize=(18, 6))

        # Set the window title and icon
        self.set_window_title_and_icon("Image Noise Analysis", "icons/sound_wave.ico")

        # Create a GridSpec layout for side-by-side plotting
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

        # Left side: Image display with RectangleSelector
        self.ax_image = self.fig.add_subplot(gs[0, 0])
        self.ax_image.imshow(self.image_array, cmap='gray')
        self.ax_image.set_title('Select region by dragging the mouse.')

        # Overlay a fine grid
        self.ax_image.set_xticks(np.arange(0, self.image_array.shape[1], 100), minor=True)
        self.ax_image.set_yticks(np.arange(0, self.image_array.shape[0], 100), minor=True)
        self.ax_image.grid(which='minor', color='y', linestyle='-', linewidth=0.1)

        # Middle: Histogram and sub-images
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.ax_hist.set_title('Residuals Histogram')
        self.ax_hist.set_xlabel('Residual Value')
        self.ax_hist.set_ylabel('Frequency')

        # Right side: Denoising comparison with 2x(1+N) grid
        # Define the number of denoising methods
        self.num_methods = len(self.de_noise_methods)

        # Create a 2x(1 + num_methods) grid for original and denoised images
        # First column: Original Sub-region and Residuals
        # Next columns: Denoised Sub-regions and Residuals for each method
        gs_denoise = self.fig.add_gridspec(2, 1 + self.num_methods, wspace=0.3, hspace=0.3, left=0.66, right=0.98)

        # Original Sub-region Image
        self.ax_orig_image = self.fig.add_subplot(gs_denoise[0, 0])
        self.ax_orig_image.set_title('Sub-region')
        self.ax_orig_image.axis('off')

        # Original Residuals
        self.ax_orig_residuals = self.fig.add_subplot(gs_denoise[1, 0])
        self.ax_orig_residuals.set_title('Residuals')
        self.ax_orig_residuals.axis('off')

        # Dictionaries to hold denoised axes for images and residuals
        self.ax_denoised_images = {}
        self.ax_denoised_residuals = {}

        # Iterate over denoising methods to create subplots
        for idx, method in enumerate(self.de_noise_methods):
            # Denoised Sub-region Image
            ax_img = self.fig.add_subplot(gs_denoise[0, idx + 1])
            ax_img.set_title(f'{method}')
            ax_img.axis('off')
            self.ax_denoised_images[method] = ax_img

            # Denoised Residuals Image
            ax_res = self.fig.add_subplot(gs_denoise[1, idx + 1])
            ax_res.set_title(f'{method}')
            ax_res.axis('off')
            self.ax_denoised_residuals[method] = ax_res

        # Initialize the RectangleSelector
        self.rectangle_selector = RectangleSelector(
            self.ax_image, self.on_select,
            useblit=True,
            button=MouseButton.LEFT,  # Left mouse button only
            minspanx=1, minspany=1,
            spancoords='pixels',
            interactive=True
        )

        # Connect the key press event
        # noinspection PyTypeChecker
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

        # Show the plot
        plt.show()

    def set_window_title_and_icon(self, title, icon_path):
        """
        Sets the window title and icon for the Matplotlib figure.

        Args:
            title (str): The desired window title.
            icon_path (str): Path to the icon file (e.g., 'icon.ico' or 'icon.png').
        """
        backend = matplotlib.get_backend()
        if backend.lower() == 'tkagg':
            try:
                # import tkinter as Tk
                window = self.fig.canvas.manager.window
                window.title(title)
                window.iconbitmap(icon_path)
            except Exception as e:
                print(f"Failed to set window icon for TkAgg backend: {e}")
        else:
            # For other backends, setting window title and icon might not be supported
            print(f"Setting window title and icon is not supported for the '{backend}' backend.")

    def on_select(self, eclick, erelease):
        """
        Callback function when a region is selected.

        Args:
            eclick: MouseEvent when mouse button is pressed.
            erelease: MouseEvent when mouse button is released.
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Calculate start and end coordinates
        start_col, end_col = sorted([x1, x2])
        start_row, end_row = sorted([y1, y2])

        # Safety-check: if the region is degenerated, return
        if start_row == end_row or start_col == end_col:
            return

        # Draw area and process the selected sub-region
        self.draw_and_process_subregion(start_row, end_row, start_col, end_col)

    def on_keypress(self, event):
        """
        Handles key press events for saving and resetting.

        Args:
            event: The key press event (of type matplotlib.backend_bases.KeyEvent).
        """
        if event.key == 'up' or event.key == 'down' or event.key == 'left' or event.key == 'right':
            self.move_subregion(event)
        elif event.key == 'a':  # Add current analysis to historical data
            if self.selected_region_data:
                self.historical_data.append(self.selected_region_data.copy())
                print("Current analysis added to history.")
                self.update_image_canvas()
        elif event.key == 'd':  # Delete last analysis from historical data
            if self.historical_data:
                _ = self.historical_data.pop()
                print("Last analysis removed from history.")
                self.update_image_canvas()
            else:
                print("No historical data to remove.")
        elif event.key == 's':  # Save current analysis
            self.save_analysis_results()
            print("Current analysis saved.")
        elif event.key == 'e':  # Export historical results to a file
            self.export_historical_data()
            print("Historical data exported.")
        elif event.key == 'r':  # Reset all historical analyses
            self.reset_all()
            print("Current and historical data reset.")

    def move_subregion(self, event):
        """
        Event handler for key presses to move the selected rectangle.

        Args:
            event: The key press event (of type matplotlib.backend_bases.KeyEvent).
        """
        if self.selected_region is None:
            return  # No region to move

        # Get the current region coordinates
        start_row, end_row, start_col, end_col = self.selected_region

        # Move region based on arrow key
        if event.key == 'up':
            start_row, end_row = start_row - 1, end_row - 1
        elif event.key == 'down':
            start_row, end_row = start_row + 1, end_row + 1
        elif event.key == 'left':
            start_col, end_col = start_col - 1, end_col - 1
        elif event.key == 'right':
            start_col, end_col = start_col + 1, end_col + 1

        # Ensure region stays within image boundaries
        start_row = max(0, start_row)
        end_row = min(self.image_array.shape[0], end_row)
        start_col = max(0, start_col)
        end_col = min(self.image_array.shape[1], end_col)

        # Draw area and process the selected sub-region
        self.draw_and_process_subregion(start_row, end_row, start_col, end_col)

    def draw_and_process_subregion(self, start_row, end_row, start_col, end_col):

        # Update the selected region
        self.selected_region = [start_row, end_row, start_col, end_col]

        # Draw current image canvas with subregion patches
        self.update_image_canvas(region=self.selected_region)

        # Process the selected sub-region
        self.process_subregion(r_start=start_row, r_end=end_row, c_start=start_col, c_end=end_col)

        # Update the histogram plot with current residuals
        self.update_histogram()

        # Update the denoising comparison
        self.update_denoising_images()

        # Update the figure
        self.fig.canvas.draw_idle()

    def update_image_canvas(self, region=None):
        """
        Updates the main image canvas to show historical (and current) subregions.
        """

        # Clear previous rectangles by removing each patch individually
        self.clear_image_canvas()

        # Draw historical subregions in semi-transparent grey
        for analysis in self.historical_data:
            rect = plt.Rectangle(
                (analysis['start_col'], analysis['start_row']),
                analysis['end_col'] - analysis['start_col'],
                analysis['end_row'] - analysis['start_row'],
                fill=False, color=None, alpha=0.3, edgecolor='green'
            )
            self.ax_image.add_patch(rect)

        # If no region given as input, try to recover current region
        if not region:
            region = self.selected_region

        # Draw the current selection rectangle in blue
        if region:
            r_start, r_end, c_start, c_end = region  # Recover current region
            rect = plt.Rectangle(
                (c_start, r_start), c_end - c_start, r_end - r_start,
                fill=False, edgecolor='blue', linewidth=2
            )
            self.ax_image.add_patch(rect)

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def update_histogram(self):
        """
        Updates the histogram plot with the residuals.
        """

        # If no data available, clear all and return
        if not self.selected_region or not self.selected_region_data:
            self.clear_axes_histogram()
            return

        # Recover current region
        [r_start, r_end, c_start, c_end] = self.selected_region
        noise_mean = self.selected_region_data['noise_mean']
        noise_std = self.selected_region_data['noise_std']
        residuals = self.selected_region_data['residuals']
        sub_region_high = self.selected_region_data['sub_region_high']
        residuals_high = self.selected_region_data['residuals_high']
        noise_overall_std = self.selected_region_data['noise_overall_std']

        # Clear previous histogram and insets
        self.clear_axes_histogram()

        # Normalize the histogram
        self.ax_hist.hist(residuals.ravel(), bins=20, color='gray', edgecolor='black', density=True)
        self.ax_hist.set_title(f'Residuals Histogram | Region '
                               f'{r_start}:{r_end} ({r_end-r_start}), '
                               f'{c_start}:{c_end} ({c_end-c_start})')
        self.ax_hist.set_xlabel('Residual Value')
        self.ax_hist.set_ylabel('Probability Density')

        # Plot the current normal distribution
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = norm.pdf(x, noise_mean, noise_std)
        self.ax_hist.plot(x, p, 'b-', lw=4, label='Current Fit')

        # Plot historical normal distributions
        for analysis in self.historical_data:
            p_hist = norm.pdf(x, analysis['noise_mean'], analysis['noise_std'])
            self.ax_hist.plot(x, p_hist, color='green', lw=1, alpha=0.5)

        # Add an inset with the high-contrast sub-region image
        self.inset_axes_left = inset_axes(self.ax_hist, width="30%", height="30%", loc="upper left")
        self.inset_axes_left.imshow(sub_region_high, cmap='gray')
        self.inset_axes_left.axis('off')
        self.inset_axes_left.set_title('Region\n(contrast)', y=-0.1, loc='center', verticalalignment='top')

        # Add an inset with the residuals image
        self.inset_axes_right = inset_axes(self.ax_hist, width="30%", height="30%", loc="upper right")
        self.inset_axes_right.imshow(residuals_high, cmap='gray')
        self.inset_axes_right.axis('off')
        self.inset_axes_right.set_title('Residuals', y=-0.1, loc='center', verticalalignment='top')

        # Display noise stats in the histogram
        text_str = (f"Mean: {noise_mean:.4f}\n"
                    f"Std: {noise_std:.4f}\n"
                    f"*Std: {noise_overall_std:.4f}")
        self.ax_hist.text(0.95, 0.55, text_str, transform=self.ax_hist.transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(facecolor='white', alpha=0.8))

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def update_denoising_images(self):
        """
        Updates the denoising images in the third column's 2xN grid.
        """
        # Display Original Sub-region Image
        self.ax_orig_image.imshow(self.selected_region_data['sub_region_high'], cmap='gray')

        # Display Original Residuals Image
        self.ax_orig_residuals.imshow(self.selected_region_data['residuals_high'], cmap='gray')

        # Iterate over denoising methods to display images
        for method in self.de_noise_methods:
            denoised_data = self.selected_region_data['denoised'][method]

            # Display Denoised Sub-region Image
            ax_img = self.ax_denoised_images[method]
            ax_img.imshow(denoised_data['denoised_sub_region_high'], cmap='gray')

            # Display Denoised Residuals Image
            ax_res = self.ax_denoised_residuals[method]
            ax_res.imshow(denoised_data['denoised_residuals_high'], cmap='gray')

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def process_subregion(self, r_start, r_end, c_start, c_end):
        """
        Processes the selected sub-region: performs noise analysis and updates the plots.

        Args:
            r_start (int): Starting row index.
            r_end (int): Ending row index.
            c_start (int): Starting column index.
            c_end (int): Ending column index.
        """

        # Safety-check: if the region is degenerated, return
        if r_start == r_end or c_start == c_end:
            return

        # Extract the current sub-region
        sub_region = self.image_array[r_start:r_end, c_start:c_end]

        # Estimate noise for the sub-region
        noise_mean, noise_std, residuals = estimate_noise(sub_region=sub_region)

        # Create high-contrast sub-region images
        sub_region_high = high_contrast_image(image=sub_region)
        residuals_high = high_contrast_image(image=residuals)

        # Compute overall standard deviation from historical data
        if self.historical_data:
            # Concatenate all residuals from historical analyses
            all_residuals = np.concatenate([analysis['residuals'].ravel() for analysis in self.historical_data])
            noise_overall_std = np.std(all_residuals)
        else:  # No historical data available; use current one
            noise_overall_std = noise_std

        # Store data
        self.selected_region = [r_start, r_end, c_start, c_end]
        self.selected_region_data = {
            'start_row': r_start,
            'end_row': r_end,
            'start_col': c_start,
            'end_col': c_end,
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'residuals': residuals,
            'sub_region_high': sub_region_high,
            'residuals_high': residuals_high,
            'noise_overall_std': noise_overall_std,
            'denoised': {}
        }

        # String to print the denoised statistics
        str_denoised = ""

        # Loop over all denoising methods
        for method in self.de_noise_methods:

            # De-noise the sub-region
            if method == 'median':  # Apply median filter
                denoised_sub_region = denoise_image(sub_region, method=method, size=3)
            elif method == 'gaussian':  # Apply Gaussian filter
                denoised_sub_region = denoise_image(sub_region, method=method, sigma=noise_overall_std)
            elif method == 'tv_chambolle':  # Apply total variation denoising
                denoised_sub_region = denoise_image(sub_region, method=method, weight=0.001)
            else:
                raise ValueError(f"Unknown de-noising method: {method}")

            # Estimate noise for the de-noised sub-region
            denoised_noise_mean, denoised_noise_std, denoised_residuals = estimate_noise(denoised_sub_region)

            # Create high-contrast images for de-noised sub-region and residuals
            denoised_sub_region_high = high_contrast_image(image=denoised_sub_region)
            denoised_residuals_high = high_contrast_image(image=denoised_residuals)

            # Store denoised analysis results
            self.selected_region_data['denoised'][method] = {
                'denoised_sub_region': denoised_sub_region,
                'denoised_noise_mean': denoised_noise_mean,
                'denoised_noise_std': denoised_noise_std,
                'denoised_residuals': denoised_residuals,
                'denoised_sub_region_high': denoised_sub_region_high,
                'denoised_residuals_high': denoised_residuals_high
            }

            # Add statistics info
            str_denoised += f" | {denoised_noise_mean:.4f}, {denoised_noise_std:.4f}"

        # Print noise statistics
        print(f"{r_start}:{r_end}, {c_start}:{c_end} | "
              f"{noise_mean:.4f}, {noise_std:.4f}"
              f"{str_denoised}")

    def save_analysis_results(self):
        """
        Saves the selected region's noise analysis results to a CSV file and exports the figure as a PNG.
        """
        if self.selected_region_data is None:
            print("No region selected to save.")
            return

        # Extract only relevant data
        data = {
            'start_row': self.selected_region_data['start_row'],
            'end_row': self.selected_region_data['end_row'],
            'start_col': self.selected_region_data['start_col'],
            'end_col': self.selected_region_data['end_col'],
            'noise_mean': self.selected_region_data['noise_mean'],
            'noise_std': self.selected_region_data['noise_std']
        }

        # Append data to the CSV file
        df = pd.DataFrame([data])
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)

        # Generate a unique filename prefix based on the coordinates
        start_row, end_row, start_col, end_col = self.selected_region
        region_id = f"{start_row}_{end_row}_{start_col}_{end_col}"

        # Save the current figure as a PNG
        output_image_path = os.path.join(self.output_folder, f"{region_id}.png")
        self.fig.savefig(output_image_path)
        print(f"Results saved to CSV at {self.csv_path} and figure saved as {output_image_path}")

    def export_historical_data(self):
        """
        Exports the historical data to a file for future post-processing.
        """
        if self.historical_data is None:
            print("No historical data available to save.")
            return

        # Define the headers for the CSV file
        headers = ['start_row', 'end_row', 'start_col', 'end_col', 'noise_mean', 'noise_std']

        # Open the CSV file and write the data
        with open(self.csv_path, mode='w', newline='') as csv_file:
            # noinspection PyTypeChecker
            writer = csv.DictWriter(csv_file, fieldnames=headers)

            # Write the headers
            writer.writeheader()

            # Write each analysis entry in historical_data
            for analysis in self.historical_data:
                # Extract only relevant fields for CSV export
                row = {key: analysis[key] for key in headers}
                writer.writerow(row)

        print(f"Historical data exported to {self.csv_path}.")

    def reset_all(self):
        """
        Resets the GUI state.
        """
        # Store coordinates and analysis data of the last selected region
        self.selected_region = None
        self.selected_region_data = None

        # Initialize historical data storage
        self.historical_data = []  # List to store dictionaries of analyses

        # Clear plot elements
        self.clear_image_canvas()
        self.clear_axes_histogram()

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def clear_image_canvas(self):
        """
        Cleans up the main image canvas from all historical (and current) subregions.
        """
        # Clear previous rectangles by removing each patch individually
        for patch in self.ax_image.patches:
            patch.remove()

    def clear_axes_histogram(self):
        """
        Clears GUI axes.
        """
        # Clear previous histogram and insets
        self.ax_hist.clear()
        if self.inset_axes_left:
            self.inset_axes_left.remove()
            self.inset_axes_left = None
        if self.inset_axes_right:
            self.inset_axes_right.remove()
            self.inset_axes_right = None


def get_input_parameters():
    """
    Retrieves input parameters for the application.
    If provided via command-line arguments, use them.
    Else, open file dialogs to let the user select the required files and output directory.

    Returns:
        tuple: (header_file_path, image_file_path, output_dir)

    Raises:
        SystemExit: If any parameter is missing or invalid after prompting.
    """
    # Initialize the argument parser without requiring arguments
    parser = argparse.ArgumentParser(description='Image Region Noise Analysis')
    parser.add_argument('--header', type=str, help='Path to the header file')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--output', type=str, help='Path to the output directory')

    # Parse the known arguments
    args, _ = parser.parse_known_args()

    header_path = args.header
    image_path = args.image
    output_dir = args.output

    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Function to validate file existence
    def validate_file(path, file_description):
        if not path or not os.path.isfile(path):
            messagebox.showerror(
                "Invalid File",
                f"The {file_description} file is missing or invalid.")
            return False
        return True

    # Function to validate or create output directory
    def validate_or_create_directory(path):
        if not path:
            messagebox.showerror(
                "Invalid Directory", "The output directory is missing.")
            return False
        if os.path.isdir(path):
            if os.access(path, os.W_OK):
                return True
            else:
                messagebox.showerror(
                    "Permission Error", f"The output directory '{path}' is not writable.")
                return False
        else:
            try:
                os.makedirs(path, exist_ok=True)
                return True
            except Exception as e:
                messagebox.showerror(
                    "Directory Creation Failed", f"Failed to create output directory '{path}'.\nError: {e}")
                return False

    # If header_path not provided, open file dialog
    if not header_path:
        header_path = filedialog.askopenfilename(
            title="Select Header File",
            filetypes=[("Header Files", "*.LBL"), ("All Files", "*.*")]
        )

    # If image_path not provided, open file dialog
    if not image_path:
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.IMG"), ("All Files", "*.*")]
        )

    # If output_dir not provided, open folder dialog
    if not output_dir:
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            mustexist=False  # Allow selecting a non-existing directory to create it
        )

    # Validate the selected files
    valid_header = validate_file(header_path, "header") if header_path else False
    valid_image = validate_file(image_path, "image") if image_path else False

    # Validate or create the output directory
    valid_output = validate_or_create_directory(output_dir) if output_dir else False

    # If any file is invalid, show error and exit
    if not (valid_header and valid_image and valid_output):
        messagebox.showerror(
            "Missing Parameters",
            "Required files are missing or invalid. The program will exit.")
        sys.exit(1)

    return header_path, image_path, output_dir


def main():

    # Retrieve input parameters
    header_file_path, image_file_path, output_folder = get_input_parameters()

    # Build output path folder
    output_folder = os.path.join(output_folder, Path(image_file_path).stem)

    # Load the image
    image_array = read_image(header_file_path=header_file_path, image_file_path=image_file_path)

    # Create the image region noise analysis GUI
    ImageRegionNoiseAnalysis(image_array=image_array, output_folder=output_folder)


if __name__ == '__main__':
    main()
