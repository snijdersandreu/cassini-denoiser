import os
import pds
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from scipy.signal import wiener
import noise_analysis
from denoising_algorithms import starlet
from denoising_algorithms import bm3d
from denoising_algorithms import unet_self2self
from denoising_algorithms import wiener as custom_wiener
from denoising_algorithms import nlm_denoising # Add NLM import
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates # For radial profile
from scipy.stats import norm, skew, kurtosis, kstest, probplot # Added skew, kurtosis, kstest, probplot
import tkinter.messagebox # Added for popups
import threading # Added for NLM threading
import queue     # Added for NLM threading

# Need laplace for new metrics
from scipy.ndimage import laplace

# Import the new utility function
from image_utils import create_display_image


class DenoiseWindow(tk.Toplevel):
    def __init__(self, master, image_data=None, clean_image_data=None):
        super().__init__(master)
        self.title("Denoise & Compare")
        # image_data is a numpy array: float64 for calibrated (32-bit) images,
        # or uint8/uint16 promoted to float64 for uncalibrated images
        self.image_data = image_data
        self.clean_image_data = clean_image_data  # Clean reference image (if available)
        self.preview_size = 256
        # self.hist_size = 100  # Height of the histogram - Now using Matplotlib for PSD
        self.controls_visible = True  # State for the control panel visibility
        self.scrollable_canvas = None
        self.h_scrollbar = None
        self.results_frame_id = None
        self.results_frame = None # Add this to ensure it's initialized
        self.create_widgets()

        # State for region selection
        self.select_start_pos = None # Tuple: (canvas_widget, start_x, start_y)
        self.selection_coords = None # Tuple: (x0, y0, x1, y1) in preview coordinates
        self.result_widgets = [] # List of dicts, each containing info for one result panel
                                 # Keys: 'canvas', 'data', 'title', 'original_data', 'rect_id',
                                 #       'hist_widget', 'psd_fig', 'psd_ax', 'psd_canvas',
                                 #       'vol_label', 'snr_psd_label'

        # Initial display of results with just the original image
        self.apply_denoise()
        self.result_image = None

        # state for show_image
        self.region_rect_id = None
        self.zoom_state = None
        self.file_label_var = tk.StringVar(value="")

    def create_widgets(self):
        # Create main container frame
        main_container = ttk.Frame(self)
        main_container.pack(expand=True, fill='both', padx=10, pady=10)

        # Button to toggle the control panel
        self.toggle_button = ttk.Button(main_container, text="Hide Controls", command=self.toggle_controls)
        self.toggle_button.pack(side='top', anchor='w', pady=(0, 5))

        # Left side: Control panel (initially visible)
        self.control_frame = ttk.LabelFrame(main_container, text="Denoising Options")
        self.control_frame.pack(side='left', fill='y', padx=(0, 10)) # Initial packing

        # Right side: Image preview area
        self.preview_frame = ttk.Frame(main_container)
        self.preview_frame.pack(side='left', expand=True, fill='both')

        # Frame for algorithm result previews (including original)
        self.alg_frame = ttk.Frame(self.preview_frame) # Parent for canvas and scrollbar
        self.alg_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create a canvas for scrollable content
        self.scrollable_canvas = tk.Canvas(self.alg_frame)
        self.scrollable_canvas.pack(side='top', fill='both', expand=True)

        # Create a horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self.alg_frame, orient='horizontal', command=self.scrollable_canvas.xview)
        # self.h_scrollbar.pack(side='bottom', fill='x') # Packed conditionally in apply_denoise

        # Configure canvas
        self.scrollable_canvas.configure(xscrollcommand=self.h_scrollbar.set)

        # --- Control Frame Content ---
        # Algorithm selection checkboxes in control frame
        self.wiener_var = tk.BooleanVar(value=False)
        wiener_frame = ttk.Frame(self.control_frame)
        wiener_frame.pack(fill='x', padx=10, pady=5)

        ttk.Checkbutton(
            wiener_frame,
            text="Wiener Filter",
            variable=self.wiener_var
        ).pack(anchor="w")

        # Add parameters for Wiener filter
        wiener_params_frame = ttk.Frame(wiener_frame)
        wiener_params_frame.pack(fill='x', padx=20, pady=2)

        # Noise STD input
        ttk.Label(wiener_params_frame, text="Noise STD:").grid(row=0, column=0, sticky="w")
        self.wiener_noise_std_var = tk.StringVar(value="0.01")
        ttk.Entry(wiener_params_frame, textvariable=self.wiener_noise_std_var, width=8).grid(row=0, column=1, padx=5)

        # Kernel size input
        ttk.Label(wiener_params_frame, text="Kernel size:").grid(row=1, column=0, sticky="w")
        self.wiener_kernel_size_var = tk.StringVar(value="3")
        ttk.Entry(wiener_params_frame, textvariable=self.wiener_kernel_size_var, width=8).grid(row=1, column=1, padx=5)

        self.starlet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.control_frame,
            text="Starlet Transform",
            variable=self.starlet_var
        ).pack(anchor="w", padx=10, pady=5)

        # Add parameters for Starlet transform
        starlet_params_frame = ttk.Frame(self.control_frame)
        starlet_params_frame.pack(fill='x', padx=20, pady=2)

        # n_scales input
        ttk.Label(starlet_params_frame, text="n_scales:").grid(row=0, column=0, sticky="w")
        self.starlet_n_scales_var = tk.StringVar(value="4")
        ttk.Entry(starlet_params_frame, textvariable=self.starlet_n_scales_var, width=8).grid(row=0, column=1, padx=5)

        # k (threshold multiplier) input
        ttk.Label(starlet_params_frame, text="k (threshold):").grid(row=1, column=0, sticky="w")
        self.starlet_k_var = tk.StringVar(value="3.0")
        ttk.Entry(starlet_params_frame, textvariable=self.starlet_k_var, width=8).grid(row=1, column=1, padx=5)

        self.bm3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.control_frame,
            text="BM3D",
            variable=self.bm3d_var
        ).pack(anchor="w", padx=10, pady=5)

        # Add parameters for BM3D
        bm3d_params_frame = ttk.Frame(self.control_frame)
        bm3d_params_frame.pack(fill='x', padx=20, pady=2)

        # Sigma input (noise standard deviation)
        ttk.Label(bm3d_params_frame, text="Sigma:").grid(row=0, column=0, sticky="w")
        self.bm3d_sigma_var = tk.StringVar(value="")
        ttk.Entry(bm3d_params_frame, textvariable=self.bm3d_sigma_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(bm3d_params_frame, text="(estimate if empty)").grid(row=0, column=2, sticky="w")

        # Block size input
        ttk.Label(bm3d_params_frame, text="Block size:").grid(row=1, column=0, sticky="w")
        self.bm3d_block_size_var = tk.StringVar(value="8")
        ttk.Entry(bm3d_params_frame, textvariable=self.bm3d_block_size_var, width=8).grid(row=1, column=1, padx=5)

        # Max blocks input
        ttk.Label(bm3d_params_frame, text="Max blocks:").grid(row=2, column=0, sticky="w")
        self.bm3d_max_blocks_var = tk.StringVar(value="16")
        ttk.Entry(bm3d_params_frame, textvariable=self.bm3d_max_blocks_var, width=8).grid(row=2, column=1, padx=5)

        # Threshold multiplier
        ttk.Label(bm3d_params_frame, text="Threshold:").grid(row=3, column=0, sticky="w")
        self.bm3d_threshold_var = tk.StringVar(value="2.7")
        ttk.Entry(bm3d_params_frame, textvariable=self.bm3d_threshold_var, width=8).grid(row=3, column=1, padx=5)

        self.unet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.control_frame,
            text="UNET-Self2Self",
            variable=self.unet_var
        ).pack(anchor="w", padx=10, pady=5)

        # NLM Denoising
        self.nlm_var = tk.BooleanVar(value=False)
        nlm_frame = ttk.Frame(self.control_frame)
        nlm_frame.pack(fill='x', padx=10, pady=5)
        ttk.Checkbutton(
            nlm_frame,
            text="Non-Local Means (NLM)",
            variable=self.nlm_var
        ).pack(anchor="w")

        nlm_params_frame = ttk.Frame(nlm_frame)
        nlm_params_frame.pack(fill='x', padx=20, pady=2)

        ttk.Label(nlm_params_frame, text="Patch Size:").grid(row=0, column=0, sticky="w")
        self.nlm_patch_size_var = tk.StringVar(value="7")
        ttk.Entry(nlm_params_frame, textvariable=self.nlm_patch_size_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(nlm_params_frame, text="Patch Distance:").grid(row=1, column=0, sticky="w")
        self.nlm_patch_distance_var = tk.StringVar(value="10")
        ttk.Entry(nlm_params_frame, textvariable=self.nlm_patch_distance_var, width=8).grid(row=1, column=1, padx=5)

        ttk.Label(nlm_params_frame, text="h (filtering):").grid(row=2, column=0, sticky="w")
        self.nlm_h_var = tk.StringVar(value="0.1")
        ttk.Entry(nlm_params_frame, textvariable=self.nlm_h_var, width=8).grid(row=2, column=1, padx=5)

        # Rescaling options
        rescale_frame = ttk.LabelFrame(self.control_frame, text="Image Rescaling")
        rescale_frame.pack(fill='x', padx=10, pady=10)

        self.rescale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rescale_frame,
            text="Rescale Images Before Denoising",
            variable=self.rescale_var
        ).pack(anchor="w", padx=5, pady=5)

        # Apply button at the bottom of control frame
        ttk.Button(
            self.control_frame,
            text="Apply Denoise",
            command=self.apply_denoise
        ).pack(anchor="w", padx=10, pady=20)

    def toggle_controls(self):
        """Toggle the visibility of the control panel."""
        if self.controls_visible:
            self.control_frame.pack_forget()
            self.toggle_button.config(text="Show Controls")
        else:
            # Ensure it packs correctly back to the left
            self.control_frame.pack(side='left', fill='y', padx=(0, 10), before=self.preview_frame)
            self.toggle_button.config(text="Hide Controls")
        self.controls_visible = not self.controls_visible

    def create_histogram(self, container, image_data, title="Value Distribution"):
        """Create a small histogram subplot with percentile-based range"""
        fig = Figure(figsize=(self.preview_size/100, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Use the same percentile range as image display
        p1, p99 = np.percentile(image_data, (1, 99))
        if p99 - p1 < 1e-8:  # Handle near-constant images
            p1 = np.min(image_data)
            p99 = np.max(image_data)
            if p99 - p1 < 1e-8:  # If still too small, create a small range
                p1 = np.mean(image_data) - 1e-8
                p99 = np.mean(image_data) + 1e-8

        # Plot histogram with range limits
        ax.hist(image_data.ravel(), bins=50, density=True, alpha=0.7, color='blue',
                range=(p1, p99))
        ax.set_title(title, fontsize=8, pad=2)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Add percentile lines
        ax.axvline(p1, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axvline(p99, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        
        # Add mean and std
        mean = np.mean(image_data)
        std = np.std(image_data)
        if p1 <= mean <= p99:  # Only show if mean is in range
            ax.axvline(mean, color='g', linestyle='-', alpha=0.5, linewidth=0.5)
        
        # Add text with statistics
        stats_text = f'μ={mean:.3e}\nσ={std:.3e}'
        ax.text(0.95, 0.95, stats_text, 
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=6,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Remove margins
        fig.tight_layout()
        
        # Create canvas and display
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        return canvas.get_tk_widget()
    
    def percentile_rescale_image(self, image_data):
        """Rescale image to [0,1] range based on percentiles"""
        p1, p99 = np.percentile(image_data, (1, 99))
        scale_factor = p99 - p1
        if scale_factor < 1e-8:
            # Return a constant image (e.g., all zeros) if range is too small
            # Also return original p1 and p99 for consistency, scale_factor will be ~0
            return np.zeros_like(image_data), p1, p99, scale_factor
        
        # Scale to [0,1] range
        scaled = (image_data - p1) / scale_factor
        return np.clip(scaled, 0, 1), p1, p99, scale_factor

    def save_image(self, title, image_data):
        """
        Save the image to a user-specified location in a lossless format.
        
        Parameters:
            title (str): Title/name of the denoising method (used for default filename)
            image_data (ndarray): The image data to save
        """
        from tkinter import filedialog
        import datetime
        
        # Generate a default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{title.replace(' ', '_')}_{timestamp}.png"
        
        # Prompt user for save location
        filepath = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"), 
                ("TIFF files", "*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:  # User cancelled
            return
            
        try:
            # Normalize the image data for saving
            # We want to preserve as much information as possible for lossless saving
            
            # Approach 1: Save the raw data directly
            # This works if the image is already in a good range
            # But can lead to very dark or very bright images if not properly scaled
            
            # Approach 2: Use contrast stretching as in display
            p1, p99 = np.percentile(image_data, (1, 99))
            if p99 - p1 < 1e-8:
                save_data = np.clip(image_data, 0, 255).astype("uint8")
            else:
                scaled = (image_data - p1) / (p99 - p1)
                save_data = (np.clip(scaled, 0, 1) * 255).astype("uint8")
            
            # Create PIL image and save
            pil_image = Image.fromarray(save_data, mode='L')
            pil_image.save(filepath)
            
            print(f"Image saved to: {filepath}")
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Save Error", f"Error saving image: {str(e)}")
            print(f"Error saving image: {e}")

    def show_result(self, title, image_data, original_data=None):
        """Helper to display one result with title, PSD plot, and histogram"""
        # Create a container frame for this result
        result_container = ttk.Frame(self.results_frame)
        result_container.pack(side='left', padx=10)

        # Add title
        display_title = title
        if image_data is not None:
            h, w = image_data.shape[:2] # Get height and width
            display_title = f"{title} ({h}x{w} px)"

        ttk.Label(
            result_container,
            text=display_title,
            font=('Helvetica', 12, 'bold')
        ).pack(pady=(0, 5))

        # Create canvas for the image
        canvas = tk.Canvas(
            result_container,
            width=self.preview_size,
            height=self.preview_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray"
        )
        canvas.pack(pady=(0, 5))

        # Add metrics/plots and histogram frame
        info_frame = ttk.Frame(result_container)
        info_frame.pack(fill='x', padx=5)

        # Create PSD plot area instead of metrics label
        psd_fig = Figure(figsize=(self.preview_size/80, 1.5), dpi=100) # Adjusted size
        psd_ax = psd_fig.add_subplot(111)
        psd_canvas = FigureCanvasTkAgg(psd_fig, master=info_frame)
        psd_canvas_widget = psd_canvas.get_tk_widget()
        psd_canvas_widget.pack(fill='x', expand=True, pady=(0, 5))
        # Initialize PSD plot
        psd_ax.set_title("PSD (Calculating...)", fontsize=8)
        psd_ax.set_xlabel("Spatial Frequency", fontsize=6)
        psd_ax.set_ylabel("Avg. Power (log)", fontsize=6)
        psd_ax.tick_params(axis='both', which='major', labelsize=6)
        psd_fig.tight_layout()
        psd_canvas.draw()

        # Add labels for single-value metrics
        metrics_frame = ttk.Frame(info_frame)
        metrics_frame.pack(fill='x', pady=(5, 0))
        snr_psd_label = ttk.Label(metrics_frame, text="SNR (PSD): Calc...", font=('Helvetica', 9))
        snr_psd_label.pack(side='left', padx=(0, 10))
        
        # Create a separate frame for the ground truth metrics if available
        gt_metrics_frame = None
        gt_snr_label = None
        gt_rmse_label = None
        gt_psnr_label = None
        
        # Show ground truth metrics for any image panel if clean_image_data is available,
        # except for the "Clean Reference" panel itself.
        if self.clean_image_data is not None and title != "Clean Reference":
            gt_metrics_frame = ttk.LabelFrame(info_frame, text="Ground Truth Metrics")
            gt_metrics_frame.pack(fill='x', pady=(5, 5))
            
            # Add labels for ground truth metrics
            gt_snr_label = ttk.Label(gt_metrics_frame, text="SNR: Calc...", font=('Helvetica', 9))
            gt_snr_label.pack(side='left', padx=(5, 10))
            
            gt_psnr_label = ttk.Label(gt_metrics_frame, text="PSNR: Calc...", font=('Helvetica', 9)) 
            gt_psnr_label.pack(side='left', padx=(0, 10))
            
            gt_rmse_label = ttk.Label(gt_metrics_frame, text="RMSE: Calc...", font=('Helvetica', 9))
            gt_rmse_label.pack(side='left', padx=(0, 10))
        else:
            # Create dummy labels if GT metrics are not applicable, to maintain widget_info structure
            # These will be parented to metrics_frame and likely have empty text from update_psd_display
            gt_snr_label = ttk.Label(metrics_frame, text="") 
            gt_psnr_label = ttk.Label(metrics_frame, text="")
            gt_rmse_label = ttk.Label(metrics_frame, text="")
            # gt_metrics_frame remains None

        hist_widget = self.create_histogram(info_frame, image_data if image_data is not None else np.zeros((1,1)), title="Value Distribution")
        hist_widget.pack(fill='x', expand=True)

        # Create a frame for buttons
        button_frame = ttk.Frame(result_container)
        button_frame.pack(pady=(5, 10))

        # Add "Save Image" button
        save_button = ttk.Button(
            button_frame,
            text="Save Image",
            command=lambda t=title, d=image_data: self.save_image(t, d)
        )
        save_button.pack(side='left', padx=3)

        # Add "Analyze Residuals" button for denoised images
        if title != self.noisy_image_display_title and title != "Clean Reference" and image_data is not None:
            analyze_residuals_button = ttk.Button(
                button_frame,
                text="Analyze Residuals",
                command=lambda current_title=title, current_data=image_data: self.open_residual_analysis_window(current_title, current_data)
            )
            analyze_residuals_button.pack(side='left', padx=3)

        if image_data is None:
            # If no algorithm result, keep canvas black and update PSD plot
            psd_ax.clear()
            psd_ax.set_title("PSD (No Data)", fontsize=8)
            psd_ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=psd_ax.transAxes)
            # Store info even if data is None to maintain structure
            widget_info = {
                'canvas': canvas, 'data': None, 'title': title,
                'original_data': original_data, 'rect_id': None,
                'hist_widget': hist_widget,
                'psd_fig': psd_fig, 'psd_ax': psd_ax, 'psd_canvas': psd_canvas,
                'snr_psd_label': snr_psd_label,
                'gt_snr_label': gt_snr_label, 'gt_rmse_label': gt_rmse_label, 'gt_psnr_label': gt_psnr_label,
                'gt_metrics_frame': gt_metrics_frame
            }
            self.result_widgets.append(widget_info)
            return

        # Apply contrast stretching using the utility function
        disp_uint8 = create_display_image(image_data, method='percentile')

        # Convert to PIL and display
        pil_image = Image.fromarray(disp_uint8, mode='L') # Already uint8
        pil_image = pil_image.resize((self.preview_size, self.preview_size), Image.BILINEAR)
        photo = ImageTk.PhotoImage(pil_image)
        canvas.create_image(
            self.preview_size // 2,
            self.preview_size // 2,
            image=photo
        )
        canvas.image = photo # Keep reference

        # Store widget info (including PSD plot elements)
        widget_info = {
            'canvas': canvas, 'data': image_data, 'title': title,
            'original_data': original_data, 'rect_id': None,
            'hist_widget': hist_widget,
            'psd_fig': psd_fig, 'psd_ax': psd_ax, 'psd_canvas': psd_canvas,
            'snr_psd_label': snr_psd_label,
            'gt_snr_label': gt_snr_label, 'gt_rmse_label': gt_rmse_label, 'gt_psnr_label': gt_psnr_label,
            'gt_metrics_frame': gt_metrics_frame
        }
        self.result_widgets.append(widget_info)

        # Bind mouse events for region selection
        canvas.bind("<Button-1>", self.on_mouse_down)
        canvas.bind("<B1-Motion>", self.on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def apply_denoise(self):
        """Apply selected denoising algorithms and display results horizontally"""
        # Clear previous result widgets' internal Tkinter items from the list
        for widget_dict in self.result_widgets:
            # The actual Tkinter widgets are children of result_container, which is widget_dict['canvas'].master.master
            if widget_dict.get('canvas') and widget_dict['canvas'].master.master.winfo_exists():
                widget_dict['canvas'].master.master.destroy()
        self.result_widgets.clear()

        self.selection_coords = None
        self.select_start_pos = None

        # Determine the title for the noisy image
        self.noisy_image_display_title = "Noisy Image" if self.clean_image_data is not None else "Original Image"

        # Prepare results_frame inside the scrollable_canvas
        if self.results_frame and self.results_frame.winfo_exists():
            # Clear children of existing results_frame
            for widget in self.results_frame.winfo_children():
                widget.destroy()
        else:
            # Create results_frame as a child of scrollable_canvas
            self.results_frame = ttk.Frame(self.scrollable_canvas)
            # If results_frame_id exists (from a previous run), delete the old canvas window item
            if self.results_frame_id:
                self.scrollable_canvas.delete(self.results_frame_id)
            # Add the new results_frame to the canvas
            self.results_frame_id = self.scrollable_canvas.create_window(
                (0, 0), window=self.results_frame, anchor='nw'
            )

        # Get original image array
        original_arr = self.image_data
        
        # Rescale the image if requested using Percentile for processing
        if self.rescale_var.get():
            # Use percentile_rescale_image for denoising scaling
            arr, self.p1, self.p99, self.scale_factor = self.percentile_rescale_image(original_arr)
            # Store the rescaling factors for adjusting noise STD and reversing the scale
            if self.scale_factor < 1e-8: # Avoid division by zero if range is tiny
                self.scale_factor = 1.0 # Effectively disable scaling effect
                arr = original_arr # Don't actually scale if range is zero
                print("Warning: Image percentile range (p99-p1) too small for scaling, using original data.")
        else:
            arr = original_arr
            self.scale_factor = 1.0
            # Still need p1 if not scaled, for potential reverse scaling logic consistency, 
            # although scale_factor=1.0 means reverse scaling won't change anything.
            # Let's just use 0 as a placeholder.
            self.p1 = 0.0 
        
        # Calculate the original signal trend for consistent SNR calculations
        # Use the potentially rescaled 'arr' for this calculation
        try:
            self.orig_metrics = noise_analysis.estimate_snr_lapshenkov(arr)
            self.orig_signal_trend = self.orig_metrics['trend']
            
            # Save the 'arr' value to be used by show_result for SNR calculations
            # This is either the rescaled data or original data depending on rescale_var
            self.snr_calculation_data = arr
        except Exception as e:
            print(f"Failed to estimate original signal trend: {e}")
            self.orig_signal_trend = None
            self.orig_metrics = None
            self.snr_calculation_data = None
        
        # If rescaling enabled, we should display 'arr' (the rescaled data) as "Original Image"
        # but if rescaling disabled, display original_arr
        display_data_original = arr if self.rescale_var.get() else original_arr
        
        if self.rescale_var.get():
            print(f"Using rescaled {self.noisy_image_display_title}: [{np.min(display_data_original):.4f}, {np.max(display_data_original):.4f}]")
        else:
            print(f"Using original {self.noisy_image_display_title}: [{np.min(display_data_original):.4f}, {np.max(display_data_original):.4f}]")
        
        # If clean reference image is available, display it first
        if self.clean_image_data is not None:
            # If rescaling is enabled, also rescale the clean reference
            if self.rescale_var.get():
                # Use same scaling parameters as for the noisy image
                clean_arr = (self.clean_image_data - self.p1) / self.scale_factor
                print(f"Rescaled clean reference to [0,1] range: [{np.min(clean_arr):.4f}, {np.max(clean_arr):.4f}]")
                self.show_result("Clean Reference", clean_arr, original_data=display_data_original)
            else:
                print(f"Using clean reference directly: [{np.min(self.clean_image_data):.4f}, {np.max(self.clean_image_data):.4f}]")
                self.show_result("Clean Reference", self.clean_image_data, original_data=display_data_original)

        # Show original next (either unscaled or rescaled depending on checkbox)
        # Pass the appropriate original data reference (needed for comparisons later)
        self.show_result(self.noisy_image_display_title, display_data_original, original_data=display_data_original)

        # Apply selected algorithms and show results
        if self.wiener_var.get():
            try:
                # Parse wiener filter parameters
                try:
                    noise_std = float(self.wiener_noise_std_var.get())
                    kernel_size = int(self.wiener_kernel_size_var.get())
                except ValueError:
                    noise_std = 0.01
                    kernel_size = 3
                    print("Using default Wiener parameters due to invalid input")
                
                # Adjust noise STD based on the percentile scaling factor used (p99-p1 or 1.0)
                adjusted_noise_std = noise_std / self.scale_factor if self.rescale_var.get() else noise_std
                
                # Use our custom implementation from wiener module
                den_scaled = custom_wiener.wiener_filter(arr, kernel_size, adjusted_noise_std)
                
                # If rescaling is enabled, maintain the scaling for SNR calculation
                # The calculate_ground_truth_metrics method will handle scaling both images consistently
                if self.rescale_var.get():
                    # If we rescaled the input to [0,1], keep the output in [0,1]
                    den_display = den_scaled
                    print(f"Keeping Wiener output in [0,1] range for consistent metrics: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                else:
                    # No rescaling was applied, use as is
                    den_display = den_scaled
                    print(f"Using Wiener output directly: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                
                # Pass appropriate values for SNR calculation and display
                self.show_result("Wiener Filter", 
                                 # For SNR calculation and display
                                 den_display, 
                                 # For comparison and regional SNR
                                 original_data=display_data_original)
            except Exception as e:
                print(f"Wiener filter failed: {e}")
                den = None
                self.show_result("Wiener Filter", den, display_data_original)
            
        if self.starlet_var.get():
            try:
                # Parse starlet parameters
                try:
                    n_scales = int(self.starlet_n_scales_var.get())
                    k = float(self.starlet_k_var.get())
                except ValueError:
                    n_scales = 4
                    k = 3.0
                    print("Using default Starlet parameters due to invalid input")

                # Apply Starlet to the *potentially scaled* data 'arr'
                # Sigma is estimated internally by default if not provided
                den_scaled = starlet.apply_starlet_denoising(arr, n_scales=n_scales, k=k)
                
                # Maintain consistent scaling approach
                if self.rescale_var.get():
                    # Keep in [0,1] range for consistent metrics
                    den_display = den_scaled
                    print(f"Keeping Starlet output in [0,1] range: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                else:
                    # No rescaling applied, use as is
                    den_display = den_scaled
                    print(f"Using Starlet output directly: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                    
                self.show_result("Starlet Transform", den_display, display_data_original)
            except Exception as e:
                print(f"Starlet transform failed: {e}")
                den = None
                self.show_result("Starlet Transform", den, display_data_original)
            
        if self.bm3d_var.get():
            try:
                # Parse BM3D parameters
                custom_params = {}
                
                # Parse sigma
                sigma_str = self.bm3d_sigma_var.get().strip()
                if sigma_str:
                    try:
                        sigma_val = float(sigma_str)
                        # Adjust sigma based on the percentile scaling factor used
                        sigma_adjusted = sigma_val / self.scale_factor if self.rescale_var.get() else sigma_val
                    except ValueError:
                        sigma_adjusted = None
                        print("Invalid sigma value, using automatic estimation")
                else:
                    sigma_adjusted = None
                
                # Parse block size
                try:
                    block_size = int(self.bm3d_block_size_var.get())
                    if block_size > 0:
                        custom_params['block_size'] = block_size
                except ValueError:
                    pass
                
                # Parse max blocks
                try:
                    max_blocks = int(self.bm3d_max_blocks_var.get())
                    if max_blocks > 0:
                        custom_params['max_blocks'] = max_blocks
                except ValueError:
                    pass
                
                # Parse threshold multiplier
                try:
                    threshold = float(self.bm3d_threshold_var.get())
                    if threshold > 0:
                        custom_params['hard_threshold'] = threshold
                except ValueError:
                    pass
                
                # Create progress dialog
                progress_win = tk.Toplevel(self)
                progress_win.title("BM3D Processing")
                progress_win.geometry("300x100")
                progress_win.transient(self)
                progress_win.grab_set()
                
                progress_label = ttk.Label(progress_win, text="Processing BM3D... This may take a while")
                progress_label.pack(pady=(10, 5))
                
                progress_var = tk.DoubleVar(value=0.0)
                progress_bar = ttk.Progressbar(progress_win, variable=progress_var, length=250, mode='determinate')
                progress_bar.pack(pady=5, padx=20)
                
                status_label = ttk.Label(progress_win, text="Initializing...")
                status_label.pack(pady=5)
                
                # Define progress callback
                def update_progress(progress_value):
                    progress_var.set(progress_value * 100)
                    if progress_value < 0.5:
                        status = f"Step 1: Hard thresholding ({progress_value*200:.0f}%)"
                    else:
                        status = f"Step 2: Wiener filtering ({(progress_value-0.5)*200:.0f}%)"
                    status_label.config(text=status)
                    progress_win.update()
                
                # Apply BM3D to the *potentially scaled* data 'arr'
                print("Starting BM3D processing...")
                den_scaled = bm3d.bm3d_denoise(
                    arr, 
                    sigma=sigma_adjusted,
                    stage='all',
                    debug=True,
                    callback=update_progress
                )
                
                # Close progress window
                progress_win.grab_release()
                progress_win.destroy()
                
                # Maintain consistent scaling approach
                if self.rescale_var.get():
                    # Keep in [0,1] range for consistent metrics
                    den_display = den_scaled
                    print(f"Keeping BM3D output in [0,1] range: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                else:
                    # No rescaling applied, use as is
                    den_display = den_scaled
                    print(f"Using BM3D output directly: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                    
                self.show_result("BM3D", den_display, display_data_original)
            except Exception as e:
                print(f"BM3D failed: {e}")
                import traceback
                traceback.print_exc()
                den = None
                self.show_result("BM3D", den, display_data_original)
                
                # Close progress window if it's still open
                if 'progress_win' in locals() and progress_win.winfo_exists():
                    progress_win.grab_release()
                    progress_win.destroy()
                    
                # Show error message
                tkinter.messagebox.showerror("BM3D Error", f"BM3D processing failed: {str(e)}")
            
        if self.unet_var.get():
            try:
                # Apply UNET to the *potentially scaled* data 'arr'
                den_scaled = unet_self2self.unet_self2self_denoise(arr) 
                
                # Maintain consistent scaling approach
                if self.rescale_var.get():
                    # Keep in [0,1] range for consistent metrics
                    den_display = den_scaled
                    print(f"Keeping UNET output in [0,1] range: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                else:
                    # No rescaling applied, use as is
                    den_display = den_scaled
                    print(f"Using UNET output directly: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                    
                self.show_result("UNET-Self2Self", den_display, display_data_original)
            except Exception as e:
                print(f"UNET-Self2Self failed: {e}")
                den = None
                self.show_result("UNET-Self2Self", den, display_data_original)

        if self.nlm_var.get():
            # Parse NLM parameters (synchronous part)
            try:
                patch_size = int(self.nlm_patch_size_var.get())
                patch_distance = int(self.nlm_patch_distance_var.get())
                h_param = float(self.nlm_h_var.get())
                if patch_size <= 0 or patch_distance <= 0 or h_param <= 0:
                    raise ValueError("NLM parameters must be positive.")
            except ValueError as e_param_parse:
                print(f"Invalid NLM parameters: {e_param_parse}. Using defaults.")
                patch_size = 7
                patch_distance = 10
                h_param = 0.1
                self.nlm_patch_size_var.set(str(patch_size))
                self.nlm_patch_distance_var.set(str(patch_distance))
                self.nlm_h_var.set(str(h_param))
                # Proceed with defaults if parsing failed

            # --- Start of threaded NLM execution ---
            print(f"Starting NLM Denoise with patch_size={patch_size}, patch_distance={patch_distance}, h={h_param}...")
            self.show_nlm_processing_dialog()

            self.nlm_result_queue = queue.Queue()
            # Pass a copy of arr to be safe, especially if arr might be a view or shared.
            # NLM itself pads, creating internal copies for that, but this is an extra precaution.
            arr_copy_for_nlm = arr.copy()

            # Define the progress callback for NLM
            def nlm_gui_progress_callback(percentage):
                if hasattr(self, 'nlm_progress_win') and self.nlm_progress_win.winfo_exists():
                    # Schedule the GUI update in the main Tkinter thread
                    self.nlm_progress_win.after(0, self.update_nlm_progress, percentage)

            nlm_args = (arr_copy_for_nlm, patch_size, patch_distance, h_param)
            nlm_kwargs = {'progress_callback': nlm_gui_progress_callback, 'debug': True} # Pass callback and debug

            def nlm_worker_thread():
                try:
                    den_scaled_result = nlm_denoising.nlm_denoise(*nlm_args, **nlm_kwargs)
                    self.nlm_result_queue.put(den_scaled_result)
                except Exception as e_thread:
                    self.nlm_result_queue.put(e_thread) # Put the exception in the queue
            
            self.nlm_thread = threading.Thread(target=nlm_worker_thread)
            self.nlm_thread.daemon = True # Ensure thread exits when main program exits
            self.nlm_thread.start()

            # Start checking the queue for the result
            self.check_nlm_result(display_data_original)
            # --- End of threaded NLM execution ---

        # After all results are shown and widgets created, update their plots
        self.update_all_metrics() # This will now calculate and plot PSDs

        # Update scrollregion after populating results_frame
        self.results_frame.update_idletasks() # Ensure results_frame's size is calculated
        required_width = self.results_frame.winfo_reqwidth()
        required_height = self.results_frame.winfo_reqheight()
        self.scrollable_canvas.config(scrollregion=(
            0, 0, 
            required_width, 
            required_height
        ))

        # Manage scrollbar visibility
        canvas_width = self.scrollable_canvas.winfo_width()
        if required_width > canvas_width:
            if not self.h_scrollbar.winfo_ismapped(): # Avoid repacking if already visible
                self.h_scrollbar.pack(side='bottom', fill='x') # Pack it after the canvas
        else:
            if self.h_scrollbar.winfo_ismapped():
                self.h_scrollbar.pack_forget()

    # ==================== Region Selection Handlers ====================

    def on_mouse_down(self, event):
        """Start selecting a region."""
        # Clear previous selection visuals and state
        self.clear_all_selections()
        self.selection_coords = None

        # Store start position relative to the canvas clicked
        self.select_start_pos = (event.widget, event.x, event.y)

        # Reset metrics to global values
        self.update_all_metrics() # Use None selection_coords

    def on_mouse_drag(self, event):
        """Update the temporary selection rectangle on the source canvas."""
        if not self.select_start_pos:
            return

        source_canvas, start_x, start_y = self.select_start_pos

        # Only draw on the canvas where the drag started
        if event.widget != source_canvas:
            return

        # Find the widget info for the source canvas
        source_widget_info = None
        for widget_info in self.result_widgets:
            if widget_info['canvas'] == source_canvas:
                source_widget_info = widget_info
                break
        if not source_widget_info: return # Should not happen

        # Delete previous temporary rectangle on this canvas
        if source_widget_info['rect_id']:
            source_canvas.delete(source_widget_info['rect_id'])

        # Calculate current coordinates, clamping to preview bounds
        x0 = min(start_x, event.x)
        y0 = min(start_y, event.y)
        x1 = max(start_x, event.x)
        y1 = max(start_y, event.y)

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.preview_size, x1)
        y1 = min(self.preview_size, y1)

        # Draw new temporary rectangle
        rect_id = source_canvas.create_rectangle(
            x0, y0, x1, y1, outline="red", width=2
        )
        source_widget_info['rect_id'] = rect_id

    def on_mouse_up(self, event):
        """Finalize selection, draw on all canvases, update metrics."""
        if not self.select_start_pos:
            return

        source_canvas, start_x, start_y = self.select_start_pos

        # Check if mouse up happened on the same canvas
        if event.widget != source_canvas:
            # If released outside, cancel selection
            self.clear_all_selections()
            self.select_start_pos = None
            # Metrics should already be global from on_mouse_down
            return

        # Calculate final preview coordinates
        x0 = min(start_x, event.x)
        y0 = min(start_y, event.y)
        x1 = max(start_x, event.x)
        y1 = max(start_y, event.y)

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.preview_size, x1)
        y1 = min(self.preview_size, y1)

        # Reset start position tracking
        self.select_start_pos = None

        # Store final selection if area is valid (e.g., > 1 pixel)
        if x1 > x0 and y1 > y0:
            self.selection_coords = (x0, y0, x1, y1)
        else:
            self.selection_coords = None # Treat clicks or tiny drags as reset

        # Clear the temporary rectangle drawn during drag
        self.clear_all_selections() # Clear temporary rect

        # Draw persistent rectangles on all canvases if selection is valid
        self.draw_all_selections()

        # Update all metrics based on the new selection_coords (or None)
        self.update_all_metrics()

    def clear_all_selections(self):
        """Remove selection rectangles from all canvases."""
        for widget_info in self.result_widgets:
            if widget_info['rect_id']:
                try:
                    widget_info['canvas'].delete(widget_info['rect_id'])
                except tk.TclError: # Handle cases where canvas might be gone
                    pass
                widget_info['rect_id'] = None

    def draw_all_selections(self):
        """Draw the current selection_coords on all canvases."""
        if not self.selection_coords:
            self.clear_all_selections()
            return

        self.clear_all_selections() # Clear any old ones first

        x0, y0, x1, y1 = self.selection_coords
        for widget_info in self.result_widgets:
            rect_id = widget_info['canvas'].create_rectangle(
                x0, y0, x1, y1, outline="lime", width=1 # Use lime for persistent
            )
            widget_info['rect_id'] = rect_id

    # ==================== PSD Calculation and Display ====================

    def calculate_radial_psd(self, image_patch):
        """
        Calculates the radially averaged 1D Power Spectral Density (PSD)
        of a 2D image patch.

        Args:
            image_patch (np.ndarray): A 2D numpy array representing the image patch.

        Returns:
            tuple: (frequencies, psd_1d)
                   frequencies (np.ndarray): Array of spatial frequencies.
                   psd_1d (np.ndarray): Radially averaged power spectral density.
                   Returns (None, None) if input is invalid.
        """
        if image_patch is None or image_patch.ndim != 2 or min(image_patch.shape) < 2:
            print("Warning: Invalid input for PSD calculation.")
            return None, None

        # Ensure patch is float type for calculations
        patch = image_patch.astype(float)

        # Get patch dimensions
        h, w = patch.shape

        # Apply a 2D Hanning window to reduce edge artifacts
        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window = np.outer(win_y, win_x)
        patch_windowed = patch * window

        # Compute the 2D FFT and shift the zero frequency to the center
        f_transform = np.fft.fft2(patch_windowed)
        f_shift = np.fft.fftshift(f_transform)

        # Calculate the 2D power spectrum (magnitude squared)
        power_spectrum_2d = np.abs(f_shift)**2

        # Calculate frequency coordinates
        freq_y = np.fft.fftshift(np.fft.fftfreq(h))
        freq_x = np.fft.fftshift(np.fft.fftfreq(w))
        kx, ky = np.meshgrid(freq_x, freq_y)

        # Calculate radial distance (spatial frequency magnitude) for each pixel
        k_radial = np.sqrt(kx**2 + ky**2)

        # Define number of bins for radial averaging
        # Use half the smaller dimension as a reasonable upper limit for frequency
        max_freq_index = min(h, w) // 2
        num_bins = max_freq_index # Use one bin per frequency index up to Nyquist

        if num_bins < 1:
             print("Warning: Patch too small for meaningful PSD binning.")
             return None, None

        bin_edges = np.linspace(0, np.max(k_radial), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bin the radial frequencies
        which_bin = np.digitize(k_radial.ravel(), bin_edges)

        # Calculate the sum of power in each bin
        # Ignore bin 0 (DC component may dominate) and bin > num_bins (outside range)
        psd_sum = np.zeros(num_bins)
        counts = np.zeros(num_bins)

        valid_indices = (which_bin > 0) & (which_bin <= num_bins)
        bin_indices = which_bin[valid_indices] - 1 # 0-based index
        powers = power_spectrum_2d.ravel()[valid_indices]

        np.add.at(psd_sum, bin_indices, powers)
        np.add.at(counts, bin_indices, 1)

        # Calculate the average power per bin, avoiding division by zero
        psd_1d = np.zeros_like(psd_sum)
        valid_counts = counts > 0
        psd_1d[valid_counts] = psd_sum[valid_counts] / counts[valid_counts]

        # Return frequencies (bin centers) and the averaged 1D PSD
        # Exclude the DC component bin (index 0) usually
        return bin_centers, psd_1d

    def map_preview_coords_to_data_coords(self, preview_coords, data_shape):
        """Convert preview (x0,y0,x1,y1) to data indices (ix0,iy0,ix1,iy1)."""
        if not preview_coords or not data_shape:
            return None

        px0, py0, px1, py1 = preview_coords
        data_h, data_w = data_shape

        # Prevent division by zero if preview size is 0
        if self.preview_size == 0:
            return 0, 0, data_w, data_h # Return full image coords as fallback

        scale_x = data_w / self.preview_size
        scale_y = data_h / self.preview_size

        dx0 = max(0, int(round(px0 * scale_x)))
        dy0 = max(0, int(round(py0 * scale_y)))
        dx1 = min(data_w, int(round(px1 * scale_x)))
        dy1 = min(data_h, int(round(py1 * scale_y)))

        # Ensure indices define a valid, non-empty region
        if dx1 <= dx0: dx1 = dx0 + 1
        if dy1 <= dy0: dy1 = dy0 + 1
        dx1 = min(data_w, dx1)
        dy1 = min(data_h, dy1)

        return dx0, dy0, dx1, dy1

    def update_all_metrics(self):
        """Recalculate and update metrics display (PSD plots) for all result widgets."""
        # Determine the title of the image to use as the PSD reference
        if self.clean_image_data is not None:
            psd_reference_title = "Clean Reference"
        else:
            psd_reference_title = self.noisy_image_display_title # This is "Original Image" or "Noisy Image"

        psd_reference_widget = next((w for w in self.result_widgets if w['title'] == psd_reference_title), None)

        reference_freqs = None
        reference_psd = None
        region_label_psd = " (Global)"

        if not psd_reference_widget or psd_reference_widget['data'] is None:
            print(f"{psd_reference_title} data not found for PSD reference calculation.")
        else:
            reference_full_data = psd_reference_widget['data']
            data_coords = self.map_preview_coords_to_data_coords(self.selection_coords, reference_full_data.shape)

            if data_coords:
                dx0, dy0, dx1, dy1 = data_coords
                reference_data_slice = reference_full_data[dy0:dy1, dx0:dx1]
                region_label_psd = f" (Region {dx0}:{dx1},{dy0}:{dy1})"
            else: # No selection, use global
                reference_data_slice = reference_full_data

            # Calculate PSD for the reference slice (regional or global)
            try:
                reference_freqs, reference_psd = self.calculate_radial_psd(reference_data_slice)
            except Exception as e:
                print(f"Error calculating PSD for {psd_reference_title} slice: {e}")
                reference_freqs, reference_psd = None, None

        # Now update each widget's plot
        for widget_info in self.result_widgets:
            self.update_psd_display(widget_info, reference_freqs, reference_psd, psd_reference_title, region_label_psd)

    def update_psd_display(self, widget_info, reference_freqs, reference_psd, psd_reference_title, region_label_psd):
        """Calculate and display PSD plot for a specific widget, using the current selection."""
        image_data = widget_info['data']
        psd_ax = widget_info['psd_ax']
        psd_canvas = widget_info['psd_canvas']
        title = widget_info['title']
        hist_widget = widget_info['hist_widget']
        hist_container = hist_widget.master # Assuming hist is always present
        snr_psd_label = widget_info['snr_psd_label']
        gt_snr_label = widget_info['gt_snr_label']
        gt_rmse_label = widget_info['gt_rmse_label']
        gt_psnr_label = widget_info['gt_psnr_label']
        gt_metrics_frame = widget_info.get('gt_metrics_frame')

        if image_data is None:
            psd_ax.clear()
            psd_ax.set_title(f"PSD {region_label_psd} (No Data)", fontsize=8)
            psd_ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=psd_ax.transAxes)
            try:
                psd_canvas.draw_idle() # Use draw_idle for potentially better performance
            except tk.TclError: pass # Handle if canvas is destroyed

            # Update metric labels for no data
            snr_psd_label.config(text="SNR (PSD): N/A")
            if hasattr(gt_snr_label, 'winfo_ismapped') and gt_snr_label.winfo_ismapped():
                gt_snr_label.config(text="SNR: N/A")
            if hasattr(gt_rmse_label, 'winfo_ismapped') and gt_rmse_label.winfo_ismapped():
                gt_rmse_label.config(text="RMSE: N/A")
            if hasattr(gt_psnr_label, 'winfo_ismapped') and gt_psnr_label.winfo_ismapped():
                gt_psnr_label.config(text="PSNR: N/A")
            return

        # Determine the data slice based on current selection
        data_coords = self.map_preview_coords_to_data_coords(self.selection_coords, image_data.shape)

        if data_coords:
            dx0, dy0, dx1, dy1 = data_coords
            data_slice = image_data[dy0:dy1, dx0:dx1]
        else:
            data_slice = image_data

        # Calculate PSD for the current data slice
        current_freqs, current_psd = None, None
        try:
            current_freqs, current_psd = self.calculate_radial_psd(data_slice)
        except Exception as e:
            print(f"Error calculating PSD for {title}{region_label_psd}: {e}")

        # --- Plotting ---
        psd_ax.clear()

        plot_success = False
        if current_freqs is not None and current_psd is not None:
            # Plot current PSD (log scale for power)
            psd_ax.plot(current_freqs, current_psd, label=f'{title} PSD', color='blue', linewidth=1)
            plot_success = True

        if reference_freqs is not None and reference_psd is not None:
             # Plot reference PSD for comparison (log scale for power)
             psd_ax.plot(reference_freqs, reference_psd, label=f'{psd_reference_title} PSD', color='black', linestyle='--', linewidth=0.8, alpha=0.7)
             plot_success = True


        if plot_success:
             # Use log scale for Y axis (power)
            psd_ax.set_yscale('log')
            psd_ax.set_title(f"Radially Averaged PSD {region_label_psd}", fontsize=8, pad=2)
            psd_ax.set_xlabel("Spatial Frequency", fontsize=6)
            psd_ax.set_ylabel("Avg. Power (log)", fontsize=6)
            psd_ax.tick_params(axis='both', which='major', labelsize=6)
            # Add grid for better readability
            psd_ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            psd_ax.legend(fontsize=6)
        else:
            # Display error message if PSD calculation failed for both
            psd_ax.set_title(f"PSD {region_label_psd} (Error)", fontsize=8)
            psd_ax.text(0.5, 0.5, "Calculation Error", ha='center', va='center', transform=psd_ax.transAxes)


        try:
            psd_ax.figure.tight_layout() # Adjust layout
            psd_canvas.draw_idle()
        except tk.TclError: pass # Handle if canvas is destroyed

        # --- Calculate and Display Single Value Metrics ---
        try:
            snr_metrics = noise_analysis.estimate_snr_psd(data_slice)
            if snr_metrics and snr_metrics['snr_db'] is not None and not np.isnan(snr_metrics['snr_db']):
                snr_psd_label.config(text=f"SNR (PSD): {snr_metrics['snr_db']:.2f} dB")
            else:
                snr_psd_label.config(text="SNR (PSD): N/A")
        except Exception as e:
            print(f"Error calculating PSD SNR for {title}{region_label_psd}: {e}")
            snr_psd_label.config(text="SNR (PSD): Error")
            
        # Calculate ground truth metrics if clean image is available and this is a denoised result
        if self.clean_image_data is not None and gt_metrics_frame:
            try:
                # Get the clean image slice using the same coordinates
                if data_coords:
                    dx0, dy0, dx1, dy1 = data_coords
                    # Ensure the clean image has the same shape as the current image data
                    if self.clean_image_data.shape == image_data.shape:
                        clean_slice = self.clean_image_data[dy0:dy1, dx0:dx1]
                    else:
                        # If shapes don't match, can't do region-specific comparison
                        clean_slice = self.clean_image_data
                        print(f"Warning: Clean image shape ({self.clean_image_data.shape}) doesn't match current image shape ({image_data.shape})")
                else:
                    clean_slice = self.clean_image_data
                
                # DEBUG: Print statistics about the slices before SNR calculation
                print(f"\n=== DEBUG FOR {title} ===")
                print(f"Data slice shape: {data_slice.shape}, min: {np.min(data_slice):.6f}, max: {np.max(data_slice):.6f}")
                print(f"Clean slice shape: {clean_slice.shape}, min: {np.min(clean_slice):.6f}, max: {np.max(clean_slice):.6f}")
                
                # Use our custom calculator instead of noise_analysis.calculate_snr_with_ground_truth
                gt_metrics = self.calculate_ground_truth_metrics(data_slice, clean_slice)
                
                # DEBUG: Print the calculated metrics
                print(f"Signal power: {gt_metrics.get('signal_power', 'N/A'):.6e}")
                print(f"Noise power: {gt_metrics.get('noise_power', 'N/A'):.6e}")
                print(f"Calculated metrics - SNR: {gt_metrics['snr_db']}, PSNR: {gt_metrics['psnr']}, RMSE: {gt_metrics['rmse']}")
                print("=== END DEBUG ===\n")
                
                # Update SNR label
                if gt_metrics['snr_db'] is not None and not np.isnan(gt_metrics['snr_db']):
                    if np.isinf(gt_metrics['snr_db']):
                        gt_snr_label.config(text=f"SNR: \u221E dB")  # Unicode infinity symbol
                    else:
                        gt_snr_label.config(text=f"SNR: {gt_metrics['snr_db']:.2f} dB")
                else:
                    gt_snr_label.config(text="SNR: N/A")
                
                # Update PSNR label
                if gt_metrics['psnr'] is not None and not np.isnan(gt_metrics['psnr']):
                    if np.isinf(gt_metrics['psnr']):
                        gt_psnr_label.config(text=f"PSNR: \u221E dB")  # Unicode infinity symbol
                    else:
                        gt_psnr_label.config(text=f"PSNR: {gt_metrics['psnr']:.2f} dB")
                else:
                    gt_psnr_label.config(text="PSNR: N/A")
                
                # Update RMSE label
                if gt_metrics['rmse'] is not None and not np.isnan(gt_metrics['rmse']):
                    # Format RMSE based on magnitude
                    if gt_metrics['rmse'] < 0.01:
                        gt_rmse_label.config(text=f"RMSE: {gt_metrics['rmse']:.2e}")
                    else:
                        gt_rmse_label.config(text=f"RMSE: {gt_metrics['rmse']:.4f}")
                else:
                    gt_rmse_label.config(text="RMSE: N/A")
                
            except Exception as e:
                print(f"Error calculating ground truth metrics for {title}{region_label_psd}: {e}")
                gt_snr_label.config(text="SNR: Error")
                gt_psnr_label.config(text="PSNR: Error")
                gt_rmse_label.config(text="RMSE: Error")
            
        # Update histogram for the selected region if selection exists
        hist_title = f"Value Distribution{region_label_psd}"
        # Remove old histogram and create new one for the slice
        if hist_widget and hist_widget.winfo_exists():
            hist_widget.destroy()
        new_hist_widget = self.create_histogram(hist_container, data_slice, title=hist_title)
        new_hist_widget.pack(fill='x', expand=True) # Pack below PSD plot
        widget_info['hist_widget'] = new_hist_widget # Store reference to the new one

    # ==================== Comparison Window ====================

    # def show_comparison(self, title, denoised_image, original_image):
    #     """
    #     Open a new window with larger views of both original and denoised images,
    #     along with detailed metrics and comparison information.
    #     
    #     Parameters:
    #         title (str): Title of the denoising method
    #         denoised_image (ndarray): The denoised image data
    #         original_image (ndarray): The original image data
    #     """
    #     # Create new toplevel window
    #     comp_window = tk.Toplevel(self)
    #     comp_window.title(f"Comparison: {title} vs Original")
    #     comp_window.geometry("1000x700")  # Larger window for better visibility
    #     
    #     # Create main container frame
    #     main_frame = ttk.Frame(comp_window, padding="10")
    #     main_frame.pack(fill='both', expand=True)
    #     
    #     # Create image frames side by side
    #     image_frame = ttk.Frame(main_frame)
    #     image_frame.pack(fill='both', expand=True, pady=10)
    #     
    #     # Left side: Original image
    #     original_frame = ttk.LabelFrame(image_frame, text="Original Image", padding="5")
    #     original_frame.pack(side='left', fill='both', expand=True, padx=5)
    #     
    #     # Right side: Denoised image
    #     denoised_frame = ttk.LabelFrame(image_frame, text=title, padding="5")
    #     denoised_frame.pack(side='left', fill='both', expand=True, padx=5)
    #     
    #     # Larger preview size for better visibility
    #     large_preview_size = 400
    #     
    #     # Create canvases for both images
    #     original_canvas = tk.Canvas(
    #         original_frame,
    #         width=large_preview_size,
    #         height=large_preview_size,
    #         bg="black",
    #         highlightthickness=1,
    #         highlightbackground="gray"
    #     )
    #     original_canvas.pack(padx=5, pady=5)
    #     
    #     denoised_canvas = tk.Canvas(
    #         denoised_frame,
    #         width=large_preview_size,
    #         height=large_preview_size,
    #         bg="black",
    #         highlightthickness=1,
    #         highlightbackground="gray"
    #     )
    #     denoised_canvas.pack(padx=5, pady=5)
    #     
    #     # Display original image using the utility function
    #     orig_disp_uint8 = create_display_image(original_image, method='percentile')
    #     pil_orig = Image.fromarray(orig_disp_uint8, mode='L')
    #     pil_orig = pil_orig.resize((large_preview_size, large_preview_size), Image.BILINEAR)
    #     photo_orig = ImageTk.PhotoImage(pil_orig)
    #     original_canvas.create_image(
    #         large_preview_size // 2,
    #         large_preview_size // 2,
    #         image=photo_orig
    #     )
    #     original_canvas.image = photo_orig  # Keep reference
    #     
    #     # Display denoised image using the utility function
    #     den_disp_uint8 = create_display_image(denoised_image, method='percentile')
    #     pil_den = Image.fromarray(den_disp_uint8, mode='L')
    #     pil_den = pil_den.resize((large_preview_size, large_preview_size), Image.BILINEAR)
    #     photo_den = ImageTk.PhotoImage(pil_den)
    #     denoised_canvas.create_image(
    #         large_preview_size // 2,
    #         large_preview_size // 2,
    #         image=photo_den
    #     )
    #     denoised_canvas.image = photo_den  # Keep reference
    #     
    #     # Calculate SNR metrics with consistent signal trend
    #     metrics_frame = ttk.Frame(main_frame)
    #     metrics_frame.pack(fill='x', expand=False, pady=10)
    #     
    #     try:
    #         # Use consistent signal trend for both calculations
    #         orig_metrics = self.orig_metrics if hasattr(self, 'orig_metrics') else noise_analysis.estimate_snr_lapshenkov(original_image)
    #         signal_trend = orig_metrics['trend']
    #         
    #         # Calculate denoised metrics using the same signal trend
    #         denoised_metrics = noise_analysis.estimate_snr_lapshenkov(denoised_image, reference_trend=signal_trend)
    #         
    #         # Calculate improvements
    #         snr_improvement = denoised_metrics['snr_db'] - orig_metrics['snr_db']
    #         noise_reduction = (orig_metrics['noise_rms'] - denoised_metrics['noise_rms']) / orig_metrics['noise_rms'] * 100
    #         
    #         # Display metrics in a larger, more readable format
    #         metrics_text = (
    #             f"Original SNR: {orig_metrics['snr_db']:.2f} dB\n"
    #             f"Denoised SNR: {denoised_metrics['snr_db']:.2f} dB\n"
    #             f"SNR Improvement: {snr_improvement:+.2f} dB ({snr_improvement/orig_metrics['snr_db']*100:.1f}%)\n\n"
    #             f"Original Noise RMS: {orig_metrics['noise_rms']:.4f}\n"
    #             f"Denoised Noise RMS: {denoised_metrics['noise_rms']:.4f}\n"
    #             f"Noise Reduction: {noise_reduction:.1f}%\n\n"
    #             f"Signal RMS: {orig_metrics['signal_rms']:.4f}"
    #         )
    #         
    #         metrics_label = ttk.Label(
    #             metrics_frame,
    #             text=metrics_text,
    #             font=('Helvetica', 12),
    #             justify='left'
    #         )
    #         metrics_label.pack(anchor='w', padx=10)
    #         
    #         # Add histograms
    #         hist_frame = ttk.Frame(main_frame)
    #         hist_frame.pack(fill='x', expand=False, pady=10)
    #         
    #         orig_hist = self.create_histogram(hist_frame, original_image, "Original Histogram")
    #         orig_hist.pack(side='left', fill='x', expand=True, padx=5)
    #         
    #         denoised_hist = self.create_histogram(hist_frame, denoised_image, f"{title} Histogram")
    #         denoised_hist.pack(side='left', fill='x', expand=True, padx=5)
    #         
    #         # Add difference image
    #         diff_frame = ttk.LabelFrame(main_frame, text="Difference Image (Original - Denoised)", padding="5")
    #         diff_frame.pack(fill='x', expand=False, pady=10)
    #         
    #         # Calculate difference
    #         diff_image = original_image - denoised_image
    #         
    #         # Display difference histogram
    #         diff_hist = self.create_histogram(diff_frame, diff_image, "Difference Histogram")
    #         diff_hist.pack(fill='x', expand=True, padx=5)
    #         
    #     except Exception as e:
    #         print(f"Error calculating detailed metrics: {e}")
    #         ttk.Label(
    #             metrics_frame,
    #             text=f"Error calculating metrics: {str(e)}",
    #             font=('Helvetica', 12)
    #         ).pack(anchor='w', padx=10)

    # =================== Single Value Metric Calculations ==================

    def calculate_ground_truth_metrics(self, input_image_arg, clean_image_raw_arg):
        """
        Calculate SNR, PSNR, RMSE, and other statistical metrics between an input image 
        and a clean reference image. Handles scaling consistently based on self.rescale_var.
        
        Parameters:
            input_image_arg (ndarray): Input image data (e.g., noisy or denoised).
                                     If self.rescale_var is ON, this is expected to be
                                     the output of a denoiser that processed an already scaled image.
            clean_image_raw_arg (ndarray): Raw clean reference image data.
            
        Returns:
            dict: Containing various metrics including std, skewness, kurtosis of the residual.
        """
        
        processed_input = input_image_arg.copy()
        processed_clean = clean_image_raw_arg.copy()
        
        # Determine scaling parameters if rescaling is enabled
        p1_to_use = 0.0
        scale_factor_to_use = 1.0
        rescaling_possible = False

        if self.rescale_var.get(): # Checkbox is ON
            if hasattr(self, 'p1') and hasattr(self, 'scale_factor') and self.scale_factor > 1e-8:
                p1_to_use = self.p1
                scale_factor_to_use = self.scale_factor
                rescaling_possible = True
                
                # input_image_arg (e.g., den_display) is already conceptually scaled.
                # Do not modify processed_input based on p1/scale_factor here.
                
                # Scale the raw clean image using the noisy image's p1 and scale_factor
                processed_clean = (clean_image_raw_arg - p1_to_use) / scale_factor_to_use
                
                print(f"DEBUG calculate_gt_metrics (Rescale ON):")
                print(f"  Input (e.g., den_display) original range: [{np.min(input_image_arg):.4f}, {np.max(input_image_arg):.4f}] -> Used as is.")
                print(f"  Clean (raw) range: [{np.min(clean_image_raw_arg):.4f}, {np.max(clean_image_raw_arg):.4f}]")
                print(f"  Clean scaled by noisy's p1={p1_to_use:.4f}, sf={scale_factor_to_use:.4f}: [{np.min(processed_clean):.4f}, {np.max(processed_clean):.4f}]")
            else:
                # Rescale ON but p1/scale_factor not found. Metrics will be on raw images.
                print("WARNING calculate_gt_metrics (Rescale ON): p1/scale_factor not found. Using raw images for metrics.")
        # Else (Checkbox is OFF), processed_input and processed_clean remain copies of raw args.

        # Calculate the residual (noise)
        noise = processed_input - processed_clean
        noise_flat = noise.ravel()
        
        # --- Basic residual statistics ---
        noise_mean = np.mean(noise_flat)
        noise_std = np.std(noise_flat)
        noise_min = np.min(noise_flat)
        noise_max = np.max(noise_flat)
        
        # Skewness (Pearson's moment coefficient of skewness)
        # bias=False for sample skewness
        try:
            noise_skew = skew(noise_flat, bias=False)
        except ValueError: # Can happen for constant arrays
            noise_skew = 0.0 
            
        # Kurtosis (Pearson's definition, where 3 is normal)
        # fisher=False for Pearson's kurtosis. fisher=True for excess kurtosis.
        try:
            noise_kurt = kurtosis(noise_flat, fisher=False) 
        except ValueError:
            noise_kurt = 3.0 # Kurtosis of a constant is undefined, use normal's as placeholder

        # Kolmogorov-Smirnov test for normality
        # Test against a normal distribution with mean=0 and std=1 after standardizing
        # Standardize the residual: (value - mean) / std
        # However, kstest is often used to test if data comes from a specific distribution (e.g. N(0, sigma_hat))
        # For simplicity here, we test if (noise_flat - noise_mean) / noise_std is N(0,1)
        # If noise_std is very small, avoid division by zero.
        ks_stat = 0.0
        ks_pval = 0.0 # Default to 0 (reject H0) if test can't run
        if noise_std > 1e-12:
            try:
                # Test standardized residuals against a standard normal distribution N(0,1)
                standardized_noise = (noise_flat - noise_mean) / noise_std
                ks_stat, ks_pval = kstest(standardized_noise, 'norm')
            except Exception as e_kstest:
                print(f"Warning: KStest failed: {e_kstest}")
                ks_pval = 0.0 # Indicate failure / strong non-normality
        else: # If std is zero (constant residual), it's not normally distributed in a typical sense
            ks_pval = 0.0 
            if np.all(noise_flat == noise_flat[0]): # if truly constant
                 # A single point distribution is not normal. KS test might give pval=1 if N=1.
                 # For N > 1 and constant, pval should be low.
                 if len(noise_flat) > 1: ks_pval = 0.0
                 else: ks_pval = 1.0 # Or handle as undefined for N=1. For now, 1.0.


        # --- SNR, PSNR, RMSE ---
        # Calculate signal power (from the processed clean image)
        signal_power = np.mean(processed_clean ** 2)
        
        # Calculate noise power (MSE of the residual)
        # This is mean of squared errors, where error = residual = noise
        mse = np.mean(noise ** 2) # This is E[residual^2]
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # SNR
        if mse < 1e-12:  # Avoid division by zero if noise is effectively zero
            snr_linear = float('inf')
            snr_db = float('inf')
        else:
            # SNR = P_signal / P_noise = P_signal / MSE
            snr_linear = signal_power / mse 
            if snr_linear < 1e-9 : # Avoid log(0) or log(very small number)
                 snr_db = -float('inf') # Or a very large negative number
            else:
                 snr_db = 10 * np.log10(snr_linear)
        
        # PSNR
        # Data range is from the processed clean image
        data_range = np.max(processed_clean) - np.min(processed_clean)
        if rmse < 1e-12 or data_range < 1e-10: # If RMSE is zero or data_range is zero
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(data_range / rmse)
            
        return {
            'mse': mse,
            'rmse': rmse,
            'psnr': psnr,
            'snr_db': snr_db,
            'snr_linear': snr_linear,
            'signal_power': signal_power,
            'noise_power': mse, # Noise power is MSE
            'noise_min': noise_min,
            'noise_max': noise_max,
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'noise_skew': noise_skew,
            'noise_kurt': noise_kurt,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'scaled_clean_used_for_metrics': processed_clean, # The version of clean image used for metrics
            'scaled_residual': noise  # The computed residual used for metrics
        }

    # ==================== Residual Analysis =====================
    def open_residual_analysis_window(self, denoised_title_str, denoised_image_data_full):
        """
        Open a window to analyze the residuals between images.
        
        Cases:
        1. Clean reference exists (NPZ files): show [denoised - clean] or [noisy - clean]
        2. No clean reference (LBL/IMG pairs): show [original - denoised]
        
        Parameters:
        -----------
        denoised_title_str : str
            Title of the denoised image method
        denoised_image_data_full : ndarray
            Full denoised image data array
        """
        try:
            if denoised_image_data_full is None:
                tkinter.messagebox.showwarning("Missing Data", "No denoised image data available for analysis.")
                return
                
            # Get current selection (if any)
            selected_region = None
            if self.selection_coords:
                data_coords = self.map_preview_coords_to_data_coords(self.selection_coords, denoised_image_data_full.shape)
                if data_coords:
                    dx0, dy0, dx1, dy1 = data_coords
                    min_dim = 2  # Minimum dimension for analysis
                    if (dx1 - dx0) >= min_dim and (dy1 - dy0) >= min_dim:
                        selected_region = (dx0, dy0, dx1, dy1)
            
            # Get appropriate image data based on selection
            if selected_region:
                dx0, dy0, dx1, dy1 = selected_region
                denoised_data = denoised_image_data_full[dy0:dy1, dx0:dx1]
                region_label = f" (Region {dx0}-{dx1}, {dy0}-{dy1})"
            else:
                denoised_data = denoised_image_data_full
                region_label = " (Full Image)"
            
            # === CASE 1: Clean reference image exists (NPZ files) ===
            if self.clean_image_data is not None:
                # --- Prepare Slices --- 
                # denoised_data and clean_data_for_denoised_comparison should correspond to the same region.
                # original_noisy_data_slice and clean_data_for_noisy_comparison should correspond to the same region.

                # Denoised data slice (already scaled if rescale_var is ON)
                denoised_data_slice = denoised_data # This is already sliced based on selected_region or full

                # Raw clean data slice corresponding to denoised_data_slice region
                raw_clean_slice_for_denoised_comp = None
                if selected_region:
                    dx0, dy0, dx1, dy1 = selected_region
                    if dx1 <= self.clean_image_data.shape[1] and dy1 <= self.clean_image_data.shape[0]:
                        raw_clean_slice_for_denoised_comp = self.clean_image_data[dy0:dy1, dx0:dx1]
                    else: # Fallback if region is out of bounds for clean image
                        raw_clean_slice_for_denoised_comp = self.clean_image_data
                        denoised_data_slice = denoised_image_data_full # Match scope
                        print("Warning: Region out of bounds for clean image. Using full images for denoised vs. clean.")
                else:
                    raw_clean_slice_for_denoised_comp = self.clean_image_data
                
                # Ensure denoised_data_slice and raw_clean_slice_for_denoised_comp have matching shapes
                if denoised_data_slice.shape != raw_clean_slice_for_denoised_comp.shape:
                    print(f"Shape mismatch: Denoised slice {denoised_data_slice.shape}, Clean slice for denoised {raw_clean_slice_for_denoised_comp.shape}. Attempting full images.")
                    denoised_data_slice = denoised_image_data_full
                    raw_clean_slice_for_denoised_comp = self.clean_image_data
                    if denoised_data_slice.shape != raw_clean_slice_for_denoised_comp.shape:
                        tkinter.messagebox.showerror("Shape Error", "Cannot align denoised and clean image shapes for residual analysis.")
                        return
                    region_label += " (Full Fallback)" # Append to existing region_label

                # Original noisy data slice (scaled if rescale_var is ON) and its corresponding raw clean slice
                original_noisy_data_slice = None # This will be SCALED if rescale_var is ON
                raw_clean_slice_for_noisy_comp = None
                
                if denoised_title_str != self.noisy_image_display_title:
                    original_widget_info = next((w for w in self.result_widgets if w['title'] == self.noisy_image_display_title), None)
                    if original_widget_info and original_widget_info['data'] is not None:
                        # original_widget_info['data'] is already scaled if rescale_var is ON
                        full_original_noisy_data = original_widget_info['data'] 

                        if selected_region:
                            dx0, dy0, dx1, dy1 = selected_region
                            if dx1 <= full_original_noisy_data.shape[1] and dy1 <= full_original_noisy_data.shape[0] and \
                               dx1 <= self.clean_image_data.shape[1] and dy1 <= self.clean_image_data.shape[0]:
                                original_noisy_data_slice = full_original_noisy_data[dy0:dy1, dx0:dx1]
                                raw_clean_slice_for_noisy_comp = self.clean_image_data[dy0:dy1, dx0:dx1]
                            else: # Fallback for region mismatch
                                original_noisy_data_slice = full_original_noisy_data
                                raw_clean_slice_for_noisy_comp = self.clean_image_data
                                print("Warning: Region mismatch for noisy vs. clean. Using full images.")
                        else: # No region selected, use full images
                            original_noisy_data_slice = full_original_noisy_data
                            raw_clean_slice_for_noisy_comp = self.clean_image_data
                        
                        if original_noisy_data_slice.shape != raw_clean_slice_for_noisy_comp.shape:
                            print(f"Shape mismatch: Noisy slice {original_noisy_data_slice.shape}, Clean slice for noisy {raw_clean_slice_for_noisy_comp.shape}. Using full images if possible.")
                            original_noisy_data_slice = full_original_noisy_data
                            raw_clean_slice_for_noisy_comp = self.clean_image_data
                            if original_noisy_data_slice.shape != raw_clean_slice_for_noisy_comp.shape:
                                print("Error: Cannot align noisy and clean image shapes for comparison. Skipping noisy comparison.")
                                original_noisy_data_slice = None # Prevent comparison
                                raw_clean_slice_for_noisy_comp = None

                # --- Calculate Metrics using the single source of truth --- 
                metrics_denoised_vs_clean = self.calculate_ground_truth_metrics(
                    denoised_data_slice, 
                    raw_clean_slice_for_denoised_comp
                )

                metrics_noisy_vs_clean = None
                if original_noisy_data_slice is not None and raw_clean_slice_for_noisy_comp is not None:
                    metrics_noisy_vs_clean = self.calculate_ground_truth_metrics(
                        original_noisy_data_slice, # Already scaled if rescale_var is ON
                        raw_clean_slice_for_noisy_comp # Always raw
                    )
                else:
                    metrics_noisy_vs_clean = {} # Ensure it's a dict for the popup

                # --- Calculate Actual Residual IMAGES for Display --- 
                actual_denoised_residual = None
                actual_noisy_residual = None
                p1_val = getattr(self, 'p1', 0.0)
                sf_val = getattr(self, 'scale_factor', 1.0)

                if self.rescale_var.get() and sf_val > 1e-8:
                    scaled_clean_for_den_display = (raw_clean_slice_for_denoised_comp - p1_val) / sf_val
                    actual_denoised_residual = denoised_data_slice - scaled_clean_for_den_display
                    
                    if original_noisy_data_slice is not None and raw_clean_slice_for_noisy_comp is not None:
                        if raw_clean_slice_for_noisy_comp.shape == original_noisy_data_slice.shape:
                             scaled_clean_for_noisy_display = (raw_clean_slice_for_noisy_comp - p1_val) / sf_val
                             actual_noisy_residual = original_noisy_data_slice - scaled_clean_for_noisy_display
                        else:
                            print("WARNING: Noisy residual image for display cannot be computed due to shape mismatch after scaling attempt.")
                            actual_noisy_residual = original_noisy_data_slice # Show original scaled noisy as fallback display
                else: # Rescale OFF or bad scale_factor
                    actual_denoised_residual = denoised_data_slice - raw_clean_slice_for_denoised_comp
                    if original_noisy_data_slice is not None and raw_clean_slice_for_noisy_comp is not None:
                         if raw_clean_slice_for_noisy_comp.shape == original_noisy_data_slice.shape:
                            actual_noisy_residual = original_noisy_data_slice - raw_clean_slice_for_noisy_comp
                         else:
                            actual_noisy_residual = original_noisy_data_slice

                if actual_denoised_residual is None:
                    tkinter.messagebox.showerror("Calculation Error", "Failed to calculate main residual image for analysis.")
                    return
                
                # Prepare titles for popup
                snr_text_main = "N/A"
                if 'snr_db' in metrics_denoised_vs_clean and metrics_denoised_vs_clean['snr_db'] is not None:
                    snr_val_main = "∞" if np.isinf(metrics_denoised_vs_clean['snr_db']) else f"{metrics_denoised_vs_clean['snr_db']:.2f}"
                    snr_text_main = f"SNR: {snr_val_main}dB"

                psnr_text_main = "N/A"
                if 'psnr' in metrics_denoised_vs_clean and metrics_denoised_vs_clean['psnr'] is not None:
                    psnr_val_main = "∞" if np.isinf(metrics_denoised_vs_clean['psnr']) else f"{metrics_denoised_vs_clean['psnr']:.2f}"
                    psnr_text_main = f"PSNR: {psnr_val_main}dB"

                window_title_base = f"{denoised_title_str} Residuals (vs Clean){region_label}"
                window_title = f"{window_title_base} - {snr_text_main}, {psnr_text_main}"
                
                popup_main_title = f"{denoised_title_str} - Clean"
                popup_comparison_title = None
                if actual_noisy_residual is not None and metrics_noisy_vs_clean:
                    popup_comparison_title = f"{self.noisy_image_display_title} - Clean"
                
                ResidualAnalysisPopup(self, 
                                    main_residual=actual_denoised_residual, 
                                    metrics=metrics_denoised_vs_clean, # This now comes from calculate_ground_truth_metrics
                                    comparison_residual=actual_noisy_residual,
                                    comparison_metrics=metrics_noisy_vs_clean, # This also from calculate_ground_truth_metrics
                                    main_title=popup_main_title,
                                    comparison_title=popup_comparison_title,
                                    window_title=window_title,
                                    region_label=region_label)
                
            # === CASE 2: No clean reference (LBL/IMG pairs) ===
            else:
                # Get the original image data
                original_widget_info = next((w for w in self.result_widgets if w['title'] == self.noisy_image_display_title), None)
                if not original_widget_info or original_widget_info['data'] is None:
                    tkinter.messagebox.showwarning("Missing Data", f"Original image data not available.")
                    return
                
                original_full = original_widget_info['data']
                
                # Get original data for the same region
                if selected_region:
                    dx0, dy0, dx1, dy1 = selected_region
                    if dx1 <= original_full.shape[1] and dy1 <= original_full.shape[0]:
                        original_data = original_full[dy0:dy1, dx0:dx1]
                    else:
                        original_data = original_full
                else:
                    original_data = original_full
                
                # Ensure shapes match
                if denoised_data.shape != original_data.shape:
                    tkinter.messagebox.showwarning("Shape Mismatch", 
                                                 f"Denoised shape {denoised_data.shape} and original shape {original_data.shape} do not match. Using full images.")
                    denoised_data = denoised_image_data_full
                    original_data = original_full
                    region_label = " (Full Image - Fallback)"
                
                try:
                    # Calculate [original - denoised]
                    residual_data = original_data - denoised_data
                    
                    # Calculate statistics on the residual
                    try:
                        residual_flat = residual_data.ravel()
                        current_mean = np.mean(residual_flat) # Calculate the mean
                        current_std = np.std(residual_flat)
                        metrics = {
                            # Removed mean to avoid errors
                            'std': current_std,
                            'skew': skew(residual_flat, bias=False),
                            'kurt': kurtosis(residual_flat, fisher=False),
                            'min': np.min(residual_flat),
                            'max': np.max(residual_flat),
                            'mean': current_mean # Store the mean as well
                        }
                        
                        # Kolmogorov-Smirnov test against normal distribution
                        # Standardize the residual before testing
                        std_for_kstest = current_std if current_std > 1e-12 else 1e-12  # Avoid division by zero
                        standardized_residual = (residual_flat - current_mean) / std_for_kstest
                        _, metrics['pval'] = kstest(standardized_residual, 'norm')
                        
                        # Add empty placeholders for GT metrics to avoid key errors
                        metrics['snr_db'] = None
                        metrics['psnr'] = None
                        metrics['rmse'] = None
                        
                        print(f"DEBUG: Successfully calculated metrics: {', '.join(metrics.keys())}")
                    except Exception as metric_e:
                        print(f"DEBUG: Error calculating metrics: {metric_e}")
                        print(f"DEBUG: residual_data shape: {residual_data.shape}, type: {type(residual_data)}")
                        print(f"DEBUG: residual_data min: {np.min(residual_data)}, max: {np.max(residual_data)}")
                        print(f"DEBUG: Original data shape: {original_data.shape}, Denoised data shape: {denoised_data.shape}")
                        # Create a basic metrics dict to prevent further errors
                        residual_flat = residual_data.ravel()
                        metrics = {
                            'std': np.std(residual_flat),
                            'skew': 0.0,
                            'kurt': 0.0,
                            'min': np.min(residual_flat),
                            'max': np.max(residual_flat),
                            'pval': 0.5,
                            'snr_db': None,
                            'psnr': None,
                            'rmse': None
                        }
                    
                    window_title = f"{self.noisy_image_display_title} - {denoised_title_str} Residuals{region_label}"
                    
                    # Launch residual analysis popup
                    ResidualAnalysisPopup(self, 
                                        main_residual=residual_data, 
                                        metrics=metrics,
                                        comparison_residual=None,
                                        comparison_metrics=None,
                                        main_title=f"{self.noisy_image_display_title} - {denoised_title_str}",
                                        comparison_title=None,
                                        window_title=window_title,
                                        region_label=region_label)
                except Exception as e:
                    tkinter.messagebox.showerror("Calculation Error", f"Error calculating residuals: {str(e)}")
                    print(f"Error in standard residual calculation: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"An error occurred during residual analysis: {str(e)}")
            print(f"Residual analysis error: {e}")
            import traceback
            traceback.print_exc()

    def show_nlm_processing_dialog(self):
        self.nlm_progress_win = tk.Toplevel(self)
        self.nlm_progress_win.title("NLM Processing")
        # Adjust geometry for progress bar
        self.nlm_progress_win.geometry("350x150") 
        self.nlm_progress_win.transient(self) 
        self.nlm_progress_win.grab_set() 
        self.nlm_progress_win.protocol("WM_DELETE_WINDOW", lambda: None) 

        progress_label = ttk.Label(self.nlm_progress_win, text="Processing Non-Local Means...\nThis may take some time.", justify=tk.CENTER)
        progress_label.pack(pady=10, padx=20) # Adjusted padding

        # Add Progress Bar
        self.nlm_progress_var = tk.DoubleVar()
        self.nlm_progressbar = ttk.Progressbar(self.nlm_progress_win, variable=self.nlm_progress_var, length=300, mode='determinate')
        self.nlm_progressbar.pack(pady=10, padx=20)
        
        self.nlm_progress_win.update_idletasks()

    def update_nlm_progress(self, percentage):
        if hasattr(self, 'nlm_progress_var') and hasattr(self, 'nlm_progressbar') and self.nlm_progressbar.winfo_exists():
            self.nlm_progress_var.set(percentage)
            # self.nlm_progress_win.update_idletasks() # May not be needed if main loop is responsive

    def check_nlm_result(self, display_data_original):
        try:
            # Non-blocking check of the queue
            result_or_exc = self.nlm_result_queue.get_nowait()

            # Close processing dialog
            if hasattr(self, 'nlm_progress_win') and self.nlm_progress_win.winfo_exists():
                self.nlm_progress_win.grab_release()
                self.nlm_progress_win.destroy()
                delattr(self, 'nlm_progress_win')

            if isinstance(result_or_exc, Exception):
                # Handle exception from the NLM thread
                e = result_or_exc
                print(f"NLM Denoise failed in thread: {e}")
                import traceback
                traceback.print_exc()
                den_display = None 
                self.show_result("NLM Denoise", den_display, display_data_original)
                tkinter.messagebox.showerror("NLM Error", f"NLM processing failed: {str(e)}")
            else:
                # Process successful NLM result
                den_scaled = result_or_exc
                print("NLM Denoising complete from thread.")

                # Maintain consistent scaling approach for display and metrics
                if self.rescale_var.get():
                    den_display = den_scaled 
                    print(f"Keeping NLM output in [0,1] range: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                else:
                    den_display = den_scaled
                    print(f"Using NLM output directly: [{np.min(den_scaled):.4f}, {np.max(den_scaled):.4f}]")
                
                self.show_result("NLM Denoise", den_display, display_data_original)
            
            # Update all metrics and plots now that NLM result (or failure) is processed
            self.update_all_metrics()
            # Update scrollregion as well, as a new panel might have been added
            self.results_frame.update_idletasks()
            required_width = self.results_frame.winfo_reqwidth()
            required_height = self.results_frame.winfo_reqheight()
            self.scrollable_canvas.config(scrollregion=(0, 0, required_width, required_height))
            canvas_width = self.scrollable_canvas.winfo_width()
            if required_width > canvas_width:
                if not self.h_scrollbar.winfo_ismapped():
                    self.h_scrollbar.pack(side='bottom', fill='x')
            else:
                if self.h_scrollbar.winfo_ismapped():
                    self.h_scrollbar.pack_forget()

        except queue.Empty:
            # Queue is empty, NLM still processing, reschedule check
            self.after(100, self.check_nlm_result, display_data_original)
        except Exception as e_check:
            # Catch any other unexpected error in this checker function itself
            print(f"Error in NLM result checker: {e_check}")
            if hasattr(self, 'nlm_progress_win') and self.nlm_progress_win.winfo_exists():
                self.nlm_progress_win.grab_release()
                self.nlm_progress_win.destroy()
            tkinter.messagebox.showerror("NLM Check Error", f"Error processing NLM result: {str(e_check)}")


class ResidualAnalysisPopup(tk.Toplevel):
    """
    Popup window for analyzing residuals between images.
    Shows residual image, histogram, and statistics.
    Optionally shows comparison between two residuals.
    """
    def __init__(self, master, main_residual, metrics, 
                 comparison_residual=None, comparison_metrics=None,
                 main_title="Residual", comparison_title=None,
                 window_title="Residual Analysis", region_label=""):
        super().__init__(master)
        self.title(window_title)
        self.geometry("800x650")
        
        # Ensure metrics dictionaries are properly initialized
        if metrics is None:
            print("WARNING: Metrics dictionary is None, creating empty dict")
            metrics = {}
        
        if comparison_metrics is None:
            comparison_metrics = {}
        
        # Store data
        self.main_residual = main_residual
        self.metrics = metrics
        self.comparison_residual = comparison_residual
        self.comparison_metrics = comparison_metrics
        self.main_title = main_title
        self.comparison_title = comparison_title
        self.region_label = region_label
        self.denoise_window = master # Store reference to DenoiseWindow

        # Determine the display scale factor
        self.display_scale_factor = 1.0
        if self.denoise_window.rescale_var.get(): # Check if the checkbox was on
            if hasattr(self.denoise_window, 'scale_factor') and self.denoise_window.scale_factor > 1e-8:
                # self.denoise_window.scale_factor is set to 1.0 in apply_denoise if rescale_var is false.
                # If rescale_var is true, self.denoise_window.scale_factor should be the actual factor.
                self.display_scale_factor = self.denoise_window.scale_factor
        
        # Ensure basic statistics are included
        if main_residual is not None:
            try:
                residual_flat = main_residual.ravel()
                # Removed mean calculation
                self.metrics['std'] = self.metrics.get('std', np.std(residual_flat))
                self.metrics['min'] = self.metrics.get('min', np.min(residual_flat))
                self.metrics['max'] = self.metrics.get('max', np.max(residual_flat))
                # Compute Kolmogorov-Smirnov test if missing
                if 'pval' not in self.metrics:
                    current_mean_main = np.mean(residual_flat) # Calculate mean
                    std_for_kstest_main = self.metrics['std'] if self.metrics['std'] > 1e-12 else 1e-12
                    standardized_residual_main = (residual_flat - current_mean_main) / std_for_kstest_main # Standardize
                    _, self.metrics['pval'] = kstest(standardized_residual_main, 'norm')
                # Compute skewness and kurtosis if missing
                if 'skew' not in self.metrics:
                    self.metrics['skew'] = skew(residual_flat, bias=False)
                if 'kurt' not in self.metrics:
                    self.metrics['kurt'] = kurtosis(residual_flat, fisher=False)
            except Exception as e:
                print(f"ERROR: Failed to calculate statistics for main residual: {e}")
                # Set default values to avoid further errors
                self.metrics['std'] = 0.01
                self.metrics['min'] = np.min(residual_flat) if 'residual_flat' in locals() else -1
                self.metrics['max'] = np.max(residual_flat) if 'residual_flat' in locals() else 1
                self.metrics['pval'] = 0.5
                self.metrics['skew'] = 0.0
                self.metrics['kurt'] = 0.0
        
        # Ensure comparison metrics are included if available
        if comparison_residual is not None:
            try:
                comp_flat = comparison_residual.ravel()
                # Removed mean calculation
                self.comparison_metrics['std'] = self.comparison_metrics.get('std', np.std(comp_flat))
                self.comparison_metrics['min'] = self.comparison_metrics.get('min', np.min(comp_flat))
                self.comparison_metrics['max'] = self.comparison_metrics.get('max', np.max(comp_flat))
                # Compute Kolmogorov-Smirnov test if missing
                if 'pval' not in self.comparison_metrics:
                    current_mean_comp = np.mean(comp_flat) # Calculate mean
                    std_for_kstest_comp = self.comparison_metrics['std'] if self.comparison_metrics['std'] > 1e-12 else 1e-12
                    standardized_residual_comp = (comp_flat - current_mean_comp) / std_for_kstest_comp # Standardize
                    _, self.comparison_metrics['pval'] = kstest(standardized_residual_comp, 'norm')
                # Compute skewness and kurtosis if missing
                if 'skew' not in self.comparison_metrics:
                    self.comparison_metrics['skew'] = skew(comp_flat, bias=False)
                if 'kurt' not in self.comparison_metrics:
                    self.comparison_metrics['kurt'] = kurtosis(comp_flat, fisher=False)
            except Exception as e:
                print(f"ERROR: Failed to calculate statistics for comparison residual: {e}")
                # Set default values to avoid further errors
                self.comparison_metrics['std'] = 0.01
                self.comparison_metrics['min'] = np.min(comp_flat) if 'comp_flat' in locals() else -1
                self.comparison_metrics['max'] = np.max(comp_flat) if 'comp_flat' in locals() else 1
                self.comparison_metrics['pval'] = 0.5
                self.comparison_metrics['skew'] = 0.0
                self.comparison_metrics['kurt'] = 0.0
        
        # Create main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill='both')
        
        # === Top section: Residual images ===
        images_frame = ttk.LabelFrame(main_frame, text="Residual Images", padding="5")
        images_frame.pack(pady=5, fill="x")
        
        # Main residual image
        main_image_frame = ttk.Frame(images_frame)
        main_image_frame.pack(side='left', padx=5, pady=5)
        
        ttk.Label(main_image_frame, text=main_title, font=('Helvetica', 10, 'bold')).pack(pady=(0, 5))
        
        self.main_canvas = tk.Canvas(main_image_frame, width=256, height=256, bg="black")
        self.main_canvas.pack()
        
        # Comparison residual image (if provided)
        if comparison_residual is not None:
            comp_image_frame = ttk.Frame(images_frame)
            comp_image_frame.pack(side='left', padx=5, pady=5)
            
            ttk.Label(comp_image_frame, text=comparison_title, font=('Helvetica', 10, 'bold')).pack(pady=(0, 5))
            
            self.comp_canvas = tk.Canvas(comp_image_frame, width=256, height=256, bg="black")
            self.comp_canvas.pack()
        
        # === Middle section: Statistics ===
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="5")
        stats_frame.pack(pady=5, fill="x")
        
        # Create main statistics
        main_stats_frame = ttk.Frame(stats_frame)
        main_stats_frame.pack(side='left', expand=True, fill='x', padx=10, pady=5)
        
        ttk.Label(main_stats_frame, text=f"{main_title} Statistics:", 
                font=('Helvetica', 10, 'bold')).pack(anchor='w')
        
        # Basic statistics
        scaled_std_main = metrics.get('std', 0.0)
        unscaled_std_main = scaled_std_main * self.display_scale_factor
        
        basic_stats_str = (
            f"Std Dev: {unscaled_std_main:.4e}\n"
            f"Skewness: {metrics.get('skew', 0.0):.4f}\n"
            f"Kurtosis: {metrics.get('kurt', 0.0):.4f}\n"
            f"KS p-value: {metrics.get('pval', 0.5):.4g}"
        )
        
        # Add SNR/PSNR if available
        if 'snr_db' in metrics and metrics['snr_db'] is not None and 'psnr' in metrics and metrics['psnr'] is not None:
            snr_val = "∞" if np.isinf(metrics['snr_db']) else f"{metrics['snr_db']:.2f}"
            psnr_val = "∞" if np.isinf(metrics['psnr']) else f"{metrics['psnr']:.2f}"
            
            scaled_rmse_main = metrics.get('rmse', 0.0)
            unscaled_rmse_main = scaled_rmse_main * self.display_scale_factor
            rmse_val_str = f"{unscaled_rmse_main:.4e}" if metrics.get('rmse') is not None else "N/A"
            
            basic_stats_str += f"\nSNR: {snr_val} dB\nPSNR: {psnr_val} dB\nRMSE: {rmse_val_str}"
        
        ttk.Label(main_stats_frame, text=basic_stats_str, justify='left').pack(anchor='w', pady=5)
        
        # Comparison statistics (if provided)
        if comparison_metrics is not None:
            comp_stats_frame = ttk.Frame(stats_frame)
            comp_stats_frame.pack(side='left', expand=True, fill='x', padx=10, pady=5)
            
            ttk.Label(comp_stats_frame, text=f"{comparison_title} Statistics:", 
                    font=('Helvetica', 10, 'bold')).pack(anchor='w')
            
            scaled_std_comp = comparison_metrics.get('std', 0.0)
            unscaled_std_comp = scaled_std_comp * self.display_scale_factor

            comp_stats_str = (
                f"Std Dev: {unscaled_std_comp:.4e}\n"
                f"Skewness: {comparison_metrics.get('skew', 0.0):.4f}\n"
                f"Kurtosis: {comparison_metrics.get('kurt', 0.0):.4f}\n"
                f"KS p-value: {comparison_metrics.get('pval', 0.5):.4g}"
            )
            
            # Add SNR/PSNR if available
            if ('snr_db' in comparison_metrics and comparison_metrics['snr_db'] is not None and
                'psnr' in comparison_metrics and comparison_metrics['psnr'] is not None):
                snr_val = "∞" if np.isinf(comparison_metrics['snr_db']) else f"{comparison_metrics['snr_db']:.2f}"
                psnr_val = "∞" if np.isinf(comparison_metrics['psnr']) else f"{comparison_metrics['psnr']:.2f}"

                scaled_rmse_comp = comparison_metrics.get('rmse', 0.0)
                unscaled_rmse_comp = scaled_rmse_comp * self.display_scale_factor
                rmse_val_str_comp = f"{unscaled_rmse_comp:.4e}" if comparison_metrics.get('rmse') is not None else "N/A"
                
                comp_stats_str += f"\nSNR: {snr_val} dB\nPSNR: {psnr_val} dB\nRMSE: {rmse_val_str_comp}"
            
            ttk.Label(comp_stats_frame, text=comp_stats_str, justify='left').pack(anchor='w', pady=5)
        
        # === Bottom section: Histograms and plots ===
        plots_frame = ttk.LabelFrame(main_frame, text="Distribution Analysis", padding="5")
        plots_frame.pack(pady=5, fill='both', expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(plots_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Histogram
        hist_frame = ttk.Frame(self.notebook)
        self.notebook.add(hist_frame, text="Histogram")
        
        # Tab 2: Q-Q Plot
        qq_frame = ttk.Frame(self.notebook)
        self.notebook.add(qq_frame, text="Q-Q Plot")
        
        # Initialize plots
        self.init_histogram(hist_frame)
        self.init_qq_plot(qq_frame)
        
        # Display residual images
        self.display_residual_images()
    
    def init_histogram(self, parent):
        """Initialize histogram plot"""
        self.hist_fig = Figure(figsize=(7, 4), dpi=100)
        
        # Create one or two subplots depending on whether we have comparison data
        if self.comparison_residual is not None:
            self.hist_ax1 = self.hist_fig.add_subplot(121)
            self.hist_ax2 = self.hist_fig.add_subplot(122)
        else:
            self.hist_ax1 = self.hist_fig.add_subplot(111)
            self.hist_ax2 = None
        
        # Plot main residual histogram
        self.plot_residual_histogram(self.hist_ax1, self.main_residual, self.metrics, self.main_title)
        
        # Plot comparison residual histogram if available
        if self.comparison_residual is not None and self.hist_ax2 is not None:
            self.plot_residual_histogram(self.hist_ax2, self.comparison_residual, 
                                        self.comparison_metrics, self.comparison_title)
        
        self.hist_fig.tight_layout()
        
        # Create canvas
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=parent)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def init_qq_plot(self, parent):
        """Initialize Q-Q plot"""
        self.qq_fig = Figure(figsize=(7, 4), dpi=100)
        
        # Create one or two subplots depending on whether we have comparison data
        if self.comparison_residual is not None:
            self.qq_ax1 = self.qq_fig.add_subplot(121)
            self.qq_ax2 = self.qq_fig.add_subplot(122)
        else:
            self.qq_ax1 = self.qq_fig.add_subplot(111)
            self.qq_ax2 = None
        
        # Plot main residual Q-Q plot
        self.plot_qq(self.qq_ax1, self.main_residual, self.main_title)
        
        # Plot comparison residual Q-Q plot if available
        if self.comparison_residual is not None and self.qq_ax2 is not None:
            self.plot_qq(self.qq_ax2, self.comparison_residual, self.comparison_title)
        
        self.qq_fig.tight_layout()
        
        # Create canvas
        self.qq_canvas = FigureCanvasTkAgg(self.qq_fig, master=parent)
        self.qq_canvas.draw()
        self.qq_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_residual_histogram(self, ax, residual, metrics, title):
        """Plot histogram with Gaussian fit overlay"""
        try:
            residual_flat = residual.ravel()
            
            # Compute histogram bins
            bins = min(50, max(10, int(np.sqrt(len(residual_flat)))))
            
            # Plot histogram
            ax.hist(residual_flat, bins=bins, density=True, alpha=0.7, color='skyblue')
            
            # Check if metrics contains the required keys
            if metrics is None or 'std' not in metrics:
                # Calculate std on the fly if necessary
                std = np.std(residual_flat)
                min_val = np.min(residual_flat)
                max_val = np.max(residual_flat)
                print(f"WARNING: Missing 'std' in metrics for {title}, calculated on the fly")
            else:
                std = metrics.get('std', np.std(residual_flat))
                min_val = metrics.get('min', np.min(residual_flat))
                max_val = metrics.get('max', np.max(residual_flat))
            
            # Ensure std is not too small to avoid issues
            std = max(std, 1e-12)
            
            # Overlay Gaussian fit - use 0 as mean
            x_range = np.linspace(min_val, max_val, 200)
            pdf = norm.pdf(x_range, loc=0, scale=std)
            ax.plot(x_range, pdf, 'r-', linewidth=2, label='Gaussian Fit (zero mean)')
            
            # Add labels
            ax.set_title(f"{title} Distribution")
            ax.set_xlabel("Residual Value")
            ax.set_ylabel("Density")
            ax.legend()
        except Exception as e:
            print(f"ERROR: Failed to create histogram for {title}: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f"Error creating histogram: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} - Error")
    
    def plot_qq(self, ax, residual, title):
        """Plot Q-Q plot against normal distribution"""
        try:
            residual_flat = residual.ravel()
            # Use loc=0 to compare against a zero-mean normal distribution
            probplot(residual_flat, dist="norm", plot=ax, fit=False)
            ax.set_title(f"{title} Q-Q Plot")
            ax.grid(True, linestyle='--', alpha=0.6)
        except Exception as e:
            print(f"ERROR: Failed to create Q-Q plot for {title}: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f"Error creating Q-Q plot: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} Q-Q Plot - Error")
    
    def display_residual_images(self):
        """Display the residual images on canvases"""
        try:
            # Display main residual
            if self.main_residual is not None:
                img_8bit = create_display_image(self.main_residual, method='minmax')  # Use minmax for residuals
                pil_img = Image.fromarray(img_8bit, mode='L').resize((256, 256), Image.BILINEAR)
                self.main_photo = ImageTk.PhotoImage(pil_img)
                self.main_canvas.create_image(0, 0, anchor=tk.NW, image=self.main_photo)
            
            # Display comparison residual if available
            if self.comparison_residual is not None and hasattr(self, 'comp_canvas'):
                comp_8bit = create_display_image(self.comparison_residual, method='minmax')
                comp_pil_img = Image.fromarray(comp_8bit, mode='L').resize((256, 256), Image.BILINEAR)
                self.comp_photo = ImageTk.PhotoImage(comp_pil_img)
                self.comp_canvas.create_image(0, 0, anchor=tk.NW, image=self.comp_photo)
        except Exception as e:
            print(f"Error displaying residual images: {e}")
            # Display error message on canvas
            if hasattr(self, 'main_canvas'):
                self.main_canvas.create_text(128, 128, text=f"Error displaying image: {str(e)}", 
                                          fill="white", anchor=tk.CENTER)

# Main application setup (if any, usually at the end of the file)
# For example:
# if __name__ == '__main__':
#     app = DenoiseWindow(None, image_data=np.random.rand(100,100)) # Example
#     app.mainloop()


