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
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates # For radial profile
from scipy.stats import norm, skew, kurtosis, kstest, probplot # Added skew, kurtosis, kstest, probplot
import tkinter.messagebox # Added for popups

# Need laplace for new metrics
from scipy.ndimage import laplace

# Import the new utility function
from image_utils import create_display_image


class DenoiseWindow(tk.Toplevel):
    def __init__(self, master, image_data=None):
        super().__init__(master)
        self.title("Denoise & Compare")
        # image_data is a numpy array: float64 for calibrated (32-bit) images,
        # or uint8/uint16 promoted to float64 for uncalibrated images
        self.image_data = image_data
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

        self.unet_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.control_frame,
            text="UNET-Self2Self",
            variable=self.unet_var
        ).pack(anchor="w", padx=10, pady=5)

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
        ttk.Label(
            result_container,
            text=title,
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
        vol_label = ttk.Label(metrics_frame, text="VoL: Calc...", font=('Helvetica', 9))
        vol_label.pack(side='left', padx=(0, 10))

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
        if title != "Original Image" and image_data is not None:
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
                'vol_label': vol_label, 'snr_psd_label': snr_psd_label
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
            'vol_label': vol_label, 'snr_psd_label': snr_psd_label
        }
        self.result_widgets.append(widget_info)

        # Bind mouse events for region selection
        canvas.bind("<Button-1>", self.on_mouse_down)
        canvas.bind("<B1-Motion>", self.on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Calculate and display initial (global) PSD - this will be done by update_all_metrics call in apply_denoise
        # self.update_psd_display(widget_info) # Removed direct call here

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
        
        # Show original first (either unscaled or rescaled depending on checkbox)
        # Pass the appropriate original data reference (needed for comparisons later)
        self.show_result("Original Image", display_data_original, original_data=display_data_original)

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
                
                # Apply Wiener filter with specified parameters to the *potentially scaled* data 'arr'
                den_scaled = wiener(arr, (kernel_size, kernel_size), adjusted_noise_std**2)
                
                # IMPORTANT: If rescaling is enabled, we DO NOT rescale back for SNR calculation
                # Instead, we keep the den_scaled value to compare SNR in the [0,1] domain
                # We'll only rescale back for display if needed
                if self.rescale_var.get() and self.scale_factor > 1e-8:
                    # Keep den_scaled for SNR calculation
                    # Only rescale for final display
                    den_display = den_scaled * self.scale_factor + self.p1
                else:
                    den_display = den_scaled
                    
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
                
                # Similar approach as Wiener filter - don't rescale for SNR calculation if rescaling enabled
                if self.rescale_var.get() and self.scale_factor > 1e-8:
                    den_display = den_scaled * self.scale_factor + self.p1
                else:
                    den_display = den_scaled
                    
                self.show_result("Starlet Transform", den_display, display_data_original)
            except Exception as e:
                print(f"Starlet transform failed: {e}")
                den = None
                self.show_result("Starlet Transform", den, display_data_original)
            
        if self.bm3d_var.get():
            try:
                # Apply BM3D to the *potentially scaled* data 'arr'
                den_scaled = bm3d.bm3d_denoise(arr) 
                
                # Similar approach
                if self.rescale_var.get() and self.scale_factor > 1e-8:
                    den_display = den_scaled * self.scale_factor + self.p1
                else:
                    den_display = den_scaled
                    
                self.show_result("BM3D", den_display, display_data_original)
            except Exception as e:
                print(f"BM3D failed: {e}")
                den = None
                self.show_result("BM3D", den, display_data_original)
            
        if self.unet_var.get():
            try:
                # Apply UNET to the *potentially scaled* data 'arr'
                den_scaled = unet_self2self.unet_self2self_denoise(arr) 
                
                # Similar approach
                if self.rescale_var.get() and self.scale_factor > 1e-8:
                    den_display = den_scaled * self.scale_factor + self.p1
                else:
                    den_display = den_scaled
                    
                self.show_result("UNET-Self2Self", den_display, display_data_original)
            except Exception as e:
                print(f"UNET-Self2Self failed: {e}")
                den = None
                self.show_result("UNET-Self2Self", den, display_data_original)

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
        # Get the original data's slice based on current selection (if any)
        original_widget = next((w for w in self.result_widgets if w['title'] == "Original Image"), None)

        original_psd = None
        original_freqs = None
        region_label_psd = " (Global)"

        if not original_widget or original_widget['data'] is None:
            print("Original image data not found for PSD calculation.")
        else:
            original_full_data = original_widget['data']
            data_coords = self.map_preview_coords_to_data_coords(self.selection_coords, original_full_data.shape)

            if data_coords:
                dx0, dy0, dx1, dy1 = data_coords
                original_data_slice = original_full_data[dy0:dy1, dx0:dx1]
                region_label_psd = f" (Region {dx0}:{dx1},{dy0}:{dy1})"
            else: # No selection, use global
                original_data_slice = original_full_data

            # Calculate PSD for the original slice (regional or global)
            try:
                original_freqs, original_psd = self.calculate_radial_psd(original_data_slice)
            except Exception as e:
                print(f"Error calculating PSD for original slice: {e}")
                original_freqs, original_psd = None, None

        # Now update each widget's plot
        for widget_info in self.result_widgets:
            self.update_psd_display(widget_info, original_freqs, original_psd, region_label_psd)

    def update_psd_display(self, widget_info, original_freqs, original_psd, region_label_psd):
        """Calculate and display PSD plot for a specific widget, using the current selection."""
        image_data = widget_info['data']
        psd_ax = widget_info['psd_ax']
        psd_canvas = widget_info['psd_canvas']
        title = widget_info['title']
        hist_widget = widget_info['hist_widget']
        hist_container = hist_widget.master # Assuming hist is always present
        vol_label = widget_info['vol_label']
        snr_psd_label = widget_info['snr_psd_label']

        if image_data is None:
            psd_ax.clear()
            psd_ax.set_title(f"PSD {region_label_psd} (No Data)", fontsize=8)
            psd_ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=psd_ax.transAxes)
            try:
                psd_canvas.draw_idle() # Use draw_idle for potentially better performance
            except tk.TclError: pass # Handle if canvas is destroyed

            # Update metric labels for no data
            vol_label.config(text="VoL: N/A")
            snr_psd_label.config(text="SNR (PSD): N/A")
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

        if original_freqs is not None and original_psd is not None:
             # Plot original PSD for comparison (log scale for power)
             psd_ax.plot(original_freqs, original_psd, label='Original PSD', color='black', linestyle='--', linewidth=0.8, alpha=0.7)
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
            vol_value = self.calculate_vol(data_slice)
            vol_label.config(text=f"VoL: {vol_value:.3g}")
        except Exception as e:
            print(f"Error calculating VoL for {title}{region_label_psd}: {e}")
            vol_label.config(text="VoL: Error")

        try:
            snr_metrics = noise_analysis.estimate_snr_psd(data_slice)
            if snr_metrics and snr_metrics['snr_db'] is not None and not np.isnan(snr_metrics['snr_db']):
                snr_psd_label.config(text=f"SNR (PSD): {snr_metrics['snr_db']:.2f} dB")
            else:
                snr_psd_label.config(text="SNR (PSD): N/A")
        except Exception as e:
            print(f"Error calculating PSD SNR for {title}{region_label_psd}: {e}")
            snr_psd_label.config(text="SNR (PSD): Error")

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

    def calculate_vol(self, image_patch):
        """Calculate Variance of Laplacian for sharpness estimation."""
        if image_patch is None or image_patch.ndim != 2 or image_patch.size == 0:
            return np.nan
        # Calculate Laplacian (scipy.ndimage.laplace uses a default kernel)
        lap = laplace(image_patch)
        return np.var(lap)

    # ==================== Residual Analysis =====================
    def open_residual_analysis_window(self, denoised_title_str, denoised_image_data_full):
        try:
            original_widget_info = next((w for w in self.result_widgets if w['title'] == "Original Image"), None)
            if not original_widget_info or original_widget_info['data'] is None:
                tkinter.messagebox.showerror("Error", "Original image data not available for residual analysis.")
                return
            original_image_data_full = original_widget_info['data']

            # Determine data slice based on current selection
            if self.selection_coords:
                data_coords_orig = self.map_preview_coords_to_data_coords(self.selection_coords, original_image_data_full.shape)
                # Denoised image should have same dimensions as original *before* preview scaling
                # The denoised_image_data_full is the full denoised image.
                data_coords_denoised = self.map_preview_coords_to_data_coords(self.selection_coords, denoised_image_data_full.shape)


                if not data_coords_orig or not data_coords_denoised:
                    tkinter.messagebox.showerror("Error", "Invalid selection coordinates for residual analysis.")
                    return
                
                dx0_o, dy0_o, dx1_o, dy1_o = data_coords_orig
                dx0_d, dy0_d, dx1_d, dy1_d = data_coords_denoised

                # Check if slice is too small
                min_dim = 2 # Minimum dimension for estimate_noise
                if (dx1_o - dx0_o) < min_dim or (dy1_o - dy0_o) < min_dim or \
                   (dx1_d - dx0_d) < min_dim or (dy1_d - dy0_d) < min_dim:
                    tkinter.messagebox.showwarning("Warning", f"Selected region is too small (min {min_dim}x{min_dim} required). Using full image for residual analysis.")
                    current_orig_data = original_image_data_full
                    current_denoised_data = denoised_image_data_full
                    region_label = " (Full Image)"
                else:
                    current_orig_data = original_image_data_full[dy0_o:dy1_o, dx0_o:dx1_o]
                    current_denoised_data = denoised_image_data_full[dy0_d:dy1_d, dx0_d:dx1_d]
                    # Use original coords for label, assuming they are representative
                    region_label = f" (Region {dx0_o}-{dx1_o}, {dy0_o}-{dy1_o})"
            else:
                current_orig_data = original_image_data_full
                current_denoised_data = denoised_image_data_full
                region_label = " (Full Image)"

            # Get residuals
            try:
                _, _, orig_residuals, _, _, _, _, _ = noise_analysis.estimate_noise(current_orig_data)
            except Exception as e:
                tkinter.messagebox.showerror("Error", f"Failed to estimate noise for original image slice: {e}")
                return
            
            try:
                _, _, denoised_residuals, _, _, _, _, _ = noise_analysis.estimate_noise(current_denoised_data)
            except Exception as e:
                tkinter.messagebox.showerror("Error", f"Failed to estimate noise for denoised image slice: {e}")
                return

            if orig_residuals is None or denoised_residuals is None:
                tkinter.messagebox.showerror("Error", "Failed to obtain residuals for analysis (None returned).")
                return

            eliminated_residual_data = orig_residuals - denoised_residuals
            
            elim_flat = eliminated_residual_data.ravel()
            if elim_flat.size == 0:
                tkinter.messagebox.showerror("Error", "Eliminated residuals data is empty.")
                return

            mean = np.mean(elim_flat)
            std = np.std(elim_flat)
            std_for_kstest = std if std > 1e-12 else 1e-12 # Avoid division by zero
            
            skewness = skew(elim_flat, bias=False)
            kurt = kurtosis(elim_flat, fisher=False) 
            _, pval = kstest((elim_flat - mean) / std_for_kstest, 'norm')
            
            stats = {
                'mean': mean, 'std': std, 
                'skew': skewness, 'kurt': kurt, 'pval': pval,
                'min': np.min(elim_flat), 'max': np.max(elim_flat) # For histogram range
            }

            ResidualAnalysisPopup(self, eliminated_residual_data, stats, 
                                  title_prefix=f"{denoised_title_str}{region_label}")

        except Exception as e:
            tkinter.messagebox.showerror("Residual Analysis Error", f"An unexpected error occurred: {str(e)}")
            print(f"Unexpected error in open_residual_analysis_window: {e}")


    # ==================== PSD Calculation and Display ====================

class ResidualAnalysisPopup(tk.Toplevel):
    def __init__(self, master, residual_data, stats, title_prefix):
        super().__init__(master)
        self.title(f"{title_prefix} - Eliminated Residuals")
        self.geometry("700x550")

        self.residual_data = residual_data
        self.stats = stats

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill='both')

        # Top: Image display
        image_frame = ttk.LabelFrame(main_frame, text="Eliminated Residuals Map", padding="5")
        image_frame.pack(pady=5, fill="x")
        
        self.image_canvas = tk.Canvas(image_frame, width=256, height=256, bg="black")
        self.image_canvas.pack(pady=5, anchor='center')
        self.display_residual_image()

        # Bottom: Histogram and Stats
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding="5")
        analysis_frame.pack(pady=5, fill="both", expand=True)

        # Stats display on the left of histogram
        stats_text_frame = ttk.Frame(analysis_frame)
        stats_text_frame.pack(side='left', fill='y', padx=10, anchor='n')
        
        ttk.Label(stats_text_frame, text="Statistics:", font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=(0,10))
        
        stats_str = (f"Mean: {self.stats['mean']:.4f}\n"
                     f"Std Dev: {self.stats['std']:.4f}\n"
                     f"Skewness: {self.stats['skew']:.4f}\n"
                     f"Kurtosis: {self.stats['kurt']:.4f}\n"
                     f"KS p-value: {self.stats['pval']:.4g}")
        ttk.Label(stats_text_frame, text=stats_str, justify='left').pack(anchor='w')

        # Q-Q Plot button below stats
        qq_button = ttk.Button(stats_text_frame, text="Show Q-Q Plot", command=self.show_qq_plot_popup)
        qq_button.pack(anchor='w', pady=10)

        # Histogram on the right
        self.hist_fig = Figure(figsize=(5, 4), dpi=100)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=analysis_frame)
        self.hist_canvas.get_tk_widget().pack(side='right', fill='both', expand=True, padx=5)
        
        self.plot_residual_histogram()

    def display_residual_image(self):
        if self.residual_data is None: return
        # Use 'minmax' for residuals as they can be negative/positive around zero
        img_8bit = create_display_image(self.residual_data, method='minmax') 
        pil_img = Image.fromarray(img_8bit, mode='L').resize((256,256), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.image_canvas.create_image(0,0, anchor=tk.NW, image=self.tk_img)

    def plot_residual_histogram(self):
        data_flat = self.residual_data.ravel()
        if data_flat.size == 0: return

        self.hist_ax.clear()
        
        # Determine plot range for histogram, can use stats min/max or percentiles
        # Using actual min/max of the residual data for full view
        plot_min = self.stats.get('min', np.min(data_flat))
        plot_max = self.stats.get('max', np.max(data_flat))
        if plot_max - plot_min < 1e-9: # Handle constant data
            plot_min -= 0.5
            plot_max += 0.5

        self.hist_ax.hist(data_flat, bins=50, density=True, alpha=0.7, color='skyblue', range=(plot_min, plot_max), label='Residuals')

        # Overlay Gaussian fit
        x_range = np.linspace(plot_min, plot_max, 200)
        pdf = norm.pdf(x_range, loc=self.stats['mean'], scale=self.stats['std'] if self.stats['std'] > 1e-12 else 1e-12)
        self.hist_ax.plot(x_range, pdf, 'r-', linewidth=2, label='Gaussian Fit')

        self.hist_ax.set_title("Distribution of Eliminated Residuals")
        self.hist_ax.set_xlabel("Value")
        self.hist_ax.set_ylabel("Density")
        self.hist_ax.legend()
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

    def show_qq_plot_popup(self):
        data_flat = self.residual_data.ravel()
        if data_flat.size == 0:
            tkinter.messagebox.showinfo("Q-Q Plot", "No data to plot.")
            return

        qq_window = tk.Toplevel(self) # Parent is the ResidualAnalysisPopup instance
        qq_window.title("Q-Q Plot (Eliminated Residuals vs. Normal)")
        qq_window.geometry("500x450")
  
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
  
        probplot(data_flat, dist="norm", plot=ax) # Uses SciPy's probplot
        ax.set_title("Q-Q Plot against Normal Distribution")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        fig.tight_layout()
  
        canvas = FigureCanvasTkAgg(fig, master=qq_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Main application setup (if any, usually at the end of the file)
# For example:
# if __name__ == '__main__':
#     app = DenoiseWindow(None, image_data=np.random.rand(100,100)) # Example
#     app.mainloop()


