import os
import glob
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from scipy.stats import norm, probplot
import csv
import uuid
import matplotlib.pyplot as plt
import pandas as pd

# Import the new utility function
from image_utils import create_display_image 

# Path Definitions
LOGS_CSV_PATH = "/Volumes/PortableSSD/uni/TFG/cassini-denoiser/data/patch_logs.csv"

# External modules
import pds
import noise_analysis  # Adjust to match your actual module / function names
# For the denoising window
from denoise_window import DenoiseWindow

# For histogram + Gaussian fitting
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MAX_PATCH_SIZE = 30  # maximum region width/height in ORIGINAL image coords

class PDSImageViewer(tk.Tk):
    def __init__(self, start_directory=None):
        super().__init__()
        self.title("PDS Image Viewer with Region Noise Analysis")

        # Create main container that will hold both current layout and patches list
        main_container = ttk.Frame(self)
        main_container.pack(expand=True, fill='both', padx=5, pady=5)

        # ============== TOP FRAME (existing content) ==============
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill='both', expand=True)

        # ============== LAYOUT: LEFT FRAME (noise/histogram), RIGHT FRAME (image/nav) ==============
        self.left_frame = ttk.Frame(top_frame, padding="5")
        self.left_frame.grid(row=0, column=0, sticky=tk.N+tk.S)

        self.right_frame = ttk.Frame(top_frame, padding="5")
        self.right_frame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)

        # ============== LEFT FRAME CONTENT (noise histogram, etc.) ==============
        # 1) Directory label
        self.directory_label = ttk.Label(self.left_frame, text="No directory selected")
        self.directory_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        # 2) Buttons
        self.browse_btn = ttk.Button(self.left_frame, text="Browse Directory", command=self.browse_directory)
        self.browse_btn.grid(row=1, column=0, pady=5, sticky=tk.W)

        # 3) File label
        self.file_label_var = tk.StringVar(value="No file loaded")
        self.file_label = ttk.Label(self.left_frame, textvariable=self.file_label_var, font=("Arial", 14))
        self.file_label.grid(row=2, column=0, pady=10, sticky=tk.W)
        
        # 4) nested frame for the three small plots
        self.small_plots_frame = ttk.Frame(self.left_frame, padding="5")
        self.small_plots_frame.grid(row=3, column=0, sticky=tk.NSEW, pady=5)
        
        for i in range(3):
            self.small_plots_frame.columnconfigure(i, weight=1)
            
        self.small_figures = []
        self.small_canvases = []
        
        for i in range(3):
            fig = Figure(figsize=(2, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Image {i+1}", ha='center', va='center')
            self.small_figures.append(fig)
            canvas = FigureCanvasTkAgg(fig, master=self.small_plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=0, column=i, padx=2, pady=2, sticky=tk.NSEW)
            self.small_canvases.append(canvas)
        
        # 5) Region noise histogram area
        self.noise_fig = Figure(figsize=(4, 3), dpi=100)
        self.noise_ax = self.noise_fig.add_subplot(111)
        self.noise_canvas = FigureCanvasTkAgg(self.noise_fig, master=self.left_frame)
        self.noise_canvas.get_tk_widget().grid(row=4, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.left_frame.rowconfigure(4, weight=1)

        # 6) Show global histogram button (for entire image)
        self.global_hist_btn = ttk.Button(self.left_frame, text="Show Global Histogram", command=self.show_histogram)
        self.global_hist_btn.grid(row=5, column=0, pady=5, sticky=tk.W)

        # ============== RIGHT FRAME CONTENT (the image + nav buttons) ==============
        self.image_canvas = tk.Canvas(
            self.right_frame, width=512, height=512,
            bg="black", highlightthickness=1, highlightbackground="gray"
        )
        self.image_canvas.grid(row=0, column=0, columnspan=3, pady=10)

        # Navigation
        self.prev_btn = ttk.Button(self.right_frame, text="<< Prev", command=self.show_previous_image)
        self.prev_btn.grid(row=1, column=0, padx=5, sticky=tk.E)

        self.next_btn = ttk.Button(self.right_frame, text="Next >>", command=self.show_next_image)
        self.next_btn.grid(row=1, column=2, padx=5, sticky=tk.W)
        
        # Zoom buttons for 4 quadrants
        self.zoom_state = None  # None = full image, else 'Q1', 'Q2', 'Q3', 'Q4'
        
        self.zoom_frame = ttk.Frame(self.right_frame)
        self.zoom_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(self.zoom_frame, text="Zoom Q1", command=lambda: self.set_zoom("Q1")).grid(row=0, column=0, padx=2)
        ttk.Button(self.zoom_frame, text="Zoom Q2", command=lambda: self.set_zoom("Q2")).grid(row=0, column=1, padx=2)
        ttk.Button(self.zoom_frame, text="Zoom Q3", command=lambda: self.set_zoom("Q3")).grid(row=0, column=2, padx=2)
        ttk.Button(self.zoom_frame, text="Zoom Q4", command=lambda: self.set_zoom("Q4")).grid(row=0, column=3, padx=2)
        ttk.Button(self.zoom_frame, text="Reset Zoom", command=lambda: self.set_zoom(None)).grid(row=0, column=4, padx=2)

        # Denoise Image button
        self.denoise_btn = ttk.Button(self.right_frame, text="Denoise Image", command=self.open_denoise_window)
        self.denoise_btn.grid(row=3, column=0, columnspan=3, pady=5)

        # Keep a list of (lbl_path, img_path)
        self.image_pairs = []
        self.current_index = 0

        # Store the raw (unscaled) image data for noise analysis
        self.current_image_data = None  
        self.original_width = None
        self.original_height = None

        # Variables for region selection
        self.select_start_x = None
        self.select_start_y = None
        self.region_rect_id = None

        # Bind mouse events for region selection
        self.image_canvas.bind("<Button-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # ============== BOTTOM FRAME (patches list) ==============
        bottom_frame = ttk.LabelFrame(main_container, text="Logged Noise Patches", padding="5")
        bottom_frame.pack(fill='x', pady=(10, 0))

        # Add statistics summary frame
        self.stats_frame = ttk.Frame(bottom_frame)
        self.stats_frame.pack(fill='x', pady=(0, 5))
        
        # Create labels for each statistic
        self.stats_labels = {}
        stats_names = ["Skewness", "Kurtosis", "Noise Std", "P-Value"]
        for i, name in enumerate(stats_names):
            container = ttk.Frame(self.stats_frame)
            container.pack(side='left', padx=10)
            ttk.Label(container, text=f"{name}:", font=('Helvetica', 9, 'bold')).pack(side='left')
            self.stats_labels[name] = ttk.Label(container, text="No data", font=('Helvetica', 9))
            self.stats_labels[name].pack(side='left', padx=(5, 0))

        # Create Treeview for patches
        self.patches_tree = ttk.Treeview(
            bottom_frame,
            columns=("filename", "location", "mean", "std", "skew", "kurt", "pval"),
            show="headings",
            height=5  # Show 5 rows by default
        )

        # Configure columns
        self.patches_tree.heading("filename", text="Filename")
        self.patches_tree.heading("location", text="Location (x0,y0,x1,y1)")
        self.patches_tree.heading("mean", text="Mean")
        self.patches_tree.heading("std", text="Std")
        self.patches_tree.heading("skew", text="Skewness")
        self.patches_tree.heading("kurt", text="Kurtosis")
        self.patches_tree.heading("pval", text="p-value")

        # Set column widths
        self.patches_tree.column("filename", width=150)
        self.patches_tree.column("location", width=150)
        self.patches_tree.column("mean", width=80)
        self.patches_tree.column("std", width=80)
        self.patches_tree.column("skew", width=80)
        self.patches_tree.column("kurt", width=80)
        self.patches_tree.column("pval", width=80)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.patches_tree.yview)
        self.patches_tree.configure(yscrollcommand=scrollbar.set)

        # Pack the Treeview and scrollbar
        self.patches_tree.pack(side='left', fill='x', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Load existing patches
        self.load_patches()

        if start_directory:
            self.load_directory(start_directory)
            
        self.last_patch_info = None  # Will store info about the last selected patch
        self.log_patch_btn = ttk.Button(self.left_frame, text="Log This Patch", command=self.log_current_patch)
        self.log_patch_btn.grid(row=6, column=0, pady=5, sticky=tk.W)
    # ========================== DIRECTORY AND IMAGE LIST ==========================
    def browse_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.load_directory(selected_dir)

    def load_directory(self, directory):
        """Scan for all .LBL files in the directory and find matching .IMG files."""
        self.directory_label.configure(text=f"Directory: {directory}")
        self.image_pairs.clear()
        self.current_index = 0

        lbl_files = glob.glob(os.path.join(directory, "*.LBL")) + glob.glob(os.path.join(directory, "*.lbl"))
        for lbl_path in lbl_files:
            base_name = os.path.splitext(lbl_path)[0]
            possible_img = base_name + ".IMG"
            if not os.path.exists(possible_img):
                possible_img = base_name + ".img"
            if os.path.exists(possible_img):
                self.image_pairs.append((lbl_path, possible_img))

        if self.image_pairs:
            self.show_image(0)
        else:
            self.file_label_var.set("No .lbl/.img pairs found.")
            self.current_image_data = None
            self.clear_noise_histogram()

    # ========================== IMAGE DISPLAY ==========================
    def show_next_image(self):
        if not self.image_pairs:
            return
        self.current_index = (self.current_index + 1) % len(self.image_pairs)
        self.show_image(self.current_index)

    def show_previous_image(self):
        if not self.image_pairs:
            return
        self.current_index = (self.current_index - 1) % len(self.image_pairs)
        self.show_image(self.current_index)

    def set_zoom(self, quadrant):
        self.zoom_state = quadrant
        self.show_image(self.current_index)

    def show_image(self, index):
        """Load the image pair, apply contrast stretch, display in the canvas."""
        lbl_path, img_path = self.image_pairs[index]
        filename = os.path.basename(lbl_path)
        self.file_label_var.set(f"File: {filename}")

        # 1) Read image with pds
        image_data = pds.read_image(header_file_path=lbl_path, image_file_path=img_path)
        self.current_image_data = image_data
        self.original_height, self.original_width = image_data.shape

        # 2) Contrast stretch using the utility function for display
        disp_uint8 = create_display_image(image_data, method='percentile')

        # Crop only for display (preserve full original image in memory)
        display_data = disp_uint8 # Already uint8
        if self.zoom_state:
            h, w = display_data.shape
            half_h, half_w = h // 2, w // 2
            if self.zoom_state == "Q1":
                display_data = display_data[0:half_h, 0:half_w]
            elif self.zoom_state == "Q2":
                display_data = display_data[0:half_h, half_w:]
            elif self.zoom_state == "Q3":
                display_data = display_data[half_h:, 0:half_w]
            elif self.zoom_state == "Q4":
                display_data = display_data[half_h:, half_w:]

        # 3) Make a 512×512 PIL image
        pil_image = Image.fromarray(display_data, mode='L')
        pil_image = pil_image.resize((512, 512), Image.BILINEAR)

        # 4) Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # 5) Clear any previous rectangle
        if self.region_rect_id:
            self.image_canvas.delete(self.region_rect_id)
            self.region_rect_id = None

        # 6) Display the new image on canvas
        self.image_canvas.config(width=512, height=512)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Clear any old noise histogram
        self.clear_noise_histogram()

    # ========================== GLOBAL HISTOGRAM ==========================
    def show_histogram(self):
        """Show a histogram of the entire raw current_image_data in a Toplevel."""
        if self.current_image_data is None:
            return

        hist_window = tk.Toplevel(self)
        hist_window.title("Global Image Histogram")

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Convert to 1D array and filter out NaNs/Infs
        data = self.current_image_data.ravel()
        data = data[np.isfinite(data)]  # Remove NaN and Inf values

        if len(data) == 0:
            print("Warning: No valid data to display in histogram.")
            return

        # Compute percentile-based min/max for better visualization
        p1, p99 = np.percentile(data, (1, 99))  # Ignore outliers beyond the 1st and 99th percentiles

        ax.hist(data, bins=256, range=(p1, p99), color='blue', alpha=0.6)
        ax.set_xlim(p1, p99)
        ax.set_title("Global Histogram (Filtered Data)")

        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ========================== REGION SELECTION + NOISE ==========================
    def on_mouse_down(self, event):
        if self.current_image_data is None:
            return
        self.select_start_x = event.x
        self.select_start_y = event.y
        # Remove any old rectangle
        if self.region_rect_id:
            self.image_canvas.delete(self.region_rect_id)
            self.region_rect_id = None

    def on_mouse_drag(self, event):
        """Drag a red rectangle to show region selection."""
        if self.select_start_x is None or self.select_start_y is None:
            return

        if self.current_image_data is None:
            return

        # Remove old rect
        if self.region_rect_id:
            self.image_canvas.delete(self.region_rect_id)

        # Compute how many canvas pixels correspond to MAX_PATCH_SIZE in original image pixels
        disp_w, disp_h = 512, 512

        # Get visible region size (depending on zoom)
        zoom_scale_x = 1.0
        zoom_scale_y = 1.0
        if self.zoom_state:
            zoom_scale_x = 0.5
            zoom_scale_y = 0.5

        full_height, full_width = self.current_image_data.shape
        visible_width = full_width * zoom_scale_x
        visible_height = full_height * zoom_scale_y

        max_disp_width = MAX_PATCH_SIZE * (disp_w / visible_width)
        max_disp_height = MAX_PATCH_SIZE * (disp_h / visible_height)

        # Limit the selection box
        width = min(abs(event.x - self.select_start_x), max_disp_width)
        height = min(abs(event.y - self.select_start_y), max_disp_height)

        end_x = self.select_start_x + width * (1 if event.x > self.select_start_x else -1)
        end_y = self.select_start_y + height * (1 if event.y > self.select_start_y else -1)

        self.region_rect_id = self.image_canvas.create_rectangle(
            self.select_start_x, self.select_start_y, end_x, end_y,
            outline="red", width=2
        )

    def on_mouse_up(self, event):
        """Once the mouse is released, finalize region, estimate noise, and show histogram."""
        if self.current_image_data is None:
            return
        if self.select_start_x is None or self.select_start_y is None:
            return

        # 1) Determine bounding box in display coords
        x0 = min(self.select_start_x, event.x)
        x1 = max(self.select_start_x, event.x)
        y0 = min(self.select_start_y, event.y)
        y1 = max(self.select_start_y, event.y)

        # Reset for next selection
        self.select_start_x = None
        self.select_start_y = None

        # 2) Convert display coords -> original coords
        disp_w = 512
        disp_h = 512
        if (self.original_width is None) or (self.original_height is None):
            return

        if self.zoom_state:
            scale_x = (self.original_width / 2) / disp_w
            scale_y = (self.original_height / 2) / disp_h
        else:
            scale_x = self.original_width / disp_w
            scale_y = self.original_height / disp_h

        orig_x0 = int(round(x0 * scale_x))
        orig_x1 = int(round(x1 * scale_x))
        orig_y0 = int(round(y0 * scale_y))
        orig_y1 = int(round(y1 * scale_y))
        
        # Apply offset based on zoom
        if self.zoom_state:
            if self.zoom_state in ("Q2", "Q4"):
                orig_x0 += self.original_width // 2
                orig_x1 += self.original_width // 2
            if self.zoom_state in ("Q3", "Q4"):
                orig_y0 += self.original_height // 2
                orig_y1 += self.original_height // 2

        # Clamp to image bounds
        orig_x0 = max(0, min(orig_x0, self.original_width - 1))
        orig_x1 = max(0, min(orig_x1, self.original_width))
        orig_y0 = max(0, min(orig_y0, self.original_height - 1))
        orig_y1 = max(0, min(orig_y1, self.original_height))

        # Must have a valid region
        if orig_x1 <= orig_x0 or orig_y1 <= orig_y0:
            return

        # 3) Enforce region max
        region_width = orig_x1 - orig_x0
        region_height = orig_y1 - orig_y0
        if region_width > MAX_PATCH_SIZE:
            orig_x1 = orig_x0 + MAX_PATCH_SIZE
        if region_height > MAX_PATCH_SIZE:
            orig_y1 = orig_y0 + MAX_PATCH_SIZE

        # 4) Extract patch from the *original* data
        patch = self.current_image_data[orig_y0:orig_y1, orig_x0:orig_x1]

        noise_mean, noise_std, noise_map, noise_skewness, noise_kurtosis, noise_ks_stat, noise_ks_pval, plane = noise_analysis.estimate_noise(patch)
        self.plot_noise_histogram(noise_mean, noise_std, noise_map, noise_skewness, noise_kurtosis, noise_ks_stat, noise_ks_pval)
        self.update_small_images(patch, plane, noise_map)
        self.last_patch_info = {
            "filename": os.path.basename(self.image_pairs[self.current_index][0]),
            "orig_x0": orig_x0,
            "orig_x1": orig_x1,
            "orig_y0": orig_y0,
            "orig_y1": orig_y1,
            "mean": noise_mean,
            "std": noise_std,
            "skew": noise_skewness,
            "kurt": noise_kurtosis,
            "pval": noise_ks_pval,
            "residual": noise_map.ravel()  # flattened residual for Q-Q plot
        }

    def plot_noise_histogram(self, noise_mean, noise_std, noise_map, noise_skewness, noise_kurtosis, noise_ks_stat, noise_ks_pval):
        """
        Plot the noise_map in the left figure (flattened to 1D),
        then overlay a Gaussian fit with the computed mean & std.
        Rescale X-axis to the min/max of the data.
        """

        arr = noise_map.ravel()  # Ensure 1D

        self.noise_ax.clear()


        # (B) Plot histogram with density=True so we can overlay PDF
        self.noise_ax.hist(arr, bins=50, color='blue', alpha=0.6, density=True, label='Noise data')

        # (C) Fit a Gaussian
        x_min, x_max = arr.min(), arr.max()
        x_range = np.linspace(x_min, x_max, 200)
        pdf = norm.pdf(x_range, loc=noise_mean, scale=noise_std)

        # (D) Overlay the fitted Gaussian
        self.noise_ax.plot(x_range, pdf, color='red', linewidth=2, label='Gaussian Fit')

        # (E) Rescale x-axis
        self.noise_ax.set_xlim(x_min, x_max)

        # (F) Display the mean & std text in upper right
        text_str = (f"Mean: {noise_mean:.4f}\n"
                    f"Std:  {noise_std:.4f}\n"
                    f"Skew: {noise_skewness:.4f}\n"
                    f"Kurt: {noise_kurtosis:.4f}\n"
                    f"p-val: {noise_ks_pval:.4g}")
        self.noise_ax.text(
            0.95, 0.95, text_str,
            transform=self.noise_ax.transAxes,
            fontsize=10, color='black',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

        self.noise_ax.set_title("Noise Histogram (Residual)")
        self.noise_ax.set_xlabel("Noise value")
        self.noise_ax.set_ylabel("Density")
        self.noise_ax.legend(loc='upper left')

        self.noise_fig.tight_layout()
        self.noise_canvas.draw()

        btn = ttk.Button(self.left_frame, text="Q-Q Plot", command=lambda: self.show_qq_plot(arr))
        btn.grid(row=5, column=0, pady=5, sticky=tk.W)

    def clear_noise_histogram(self):
        """Clear the noise histogram on the left panel."""
        self.noise_ax.clear()
        self.noise_ax.set_title("Noise Histogram (No region selected)")
        self.noise_canvas.draw()

    def update_small_images(self, patch, plane, residual):
        """
        Show three images in the small figures:
        1) Original patch
        2) Fitted plane
        3) Residual
        All using min-max scaling for clarity.
        """
        # Use create_display_image with 'minmax' for these previews
        patch_8 = create_display_image(patch, method='minmax')
        plane_8 = create_display_image(plane, method='minmax')
        residual_8 = create_display_image(residual, method='minmax')

        images = [patch_8, plane_8, residual_8]
        titles = ["Patch", "Fitted Plane", "Residual"]

        for i in range(3):
            fig = self.small_figures[i]
            fig.clear()
            ax = fig.add_subplot(111)
            ax.imshow(images[i], cmap='gray', aspect='auto')
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')
            self.small_canvases[i].draw()

    def show_qq_plot(self, data_array):
        """Open a new Toplevel window showing a Q-Q plot vs. Normal."""
        qq_window = tk.Toplevel(self)
        qq_window.title("Q-Q Plot (vs. Normal)")
  
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
  
        probplot(data_array, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot")
  
        canvas = FigureCanvasTkAgg(fig, master=qq_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_stats_summary(self):
        """Update the statistics summary from the CSV data"""
        try:
            if os.path.exists(LOGS_CSV_PATH):
                df = pd.read_csv(LOGS_CSV_PATH)
                
                # Convert relevant columns to float
                columns_to_convert = ["skew", "kurt", "std", "pval"]
                for col in columns_to_convert:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Drop rows with NaN values
                df = df.dropna(subset=columns_to_convert)
                
                if not df.empty:
                    # Compute statistics
                    summary = {
                        "Skewness": (df["skew"].mean(), df["skew"].std()),
                        "Kurtosis": (df["kurt"].mean(), df["kurt"].std()),
                        "Noise Std": (df["std"].mean(), df["std"].std()),
                        "P-Value": (df["pval"].mean(), df["pval"].std())
                    }
                    
                    # Update labels
                    for metric, (mean, std) in summary.items():
                        self.stats_labels[metric].configure(
                            text=f"μ={mean:.2e} σ={std:.2e}"
                        )
                else:
                    for label in self.stats_labels.values():
                        label.configure(text="No data")
            else:
                for label in self.stats_labels.values():
                    label.configure(text="No data")
        except Exception as e:
            print(f"Error updating stats summary: {e}")
            for label in self.stats_labels.values():
                label.configure(text="Error")

    def load_patches(self):
        """Load and display patches from the CSV file"""
        try:
            # Clear existing items
            for item in self.patches_tree.get_children():
                self.patches_tree.delete(item)

            # Read CSV if it exists
            if os.path.exists(LOGS_CSV_PATH):
                with open(LOGS_CSV_PATH, mode="r", newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Format location string
                        location = f"({row['orig_x0']},{row['orig_y0']},{row['orig_x1']},{row['orig_y1']})"
                        
                        # Format values with appropriate precision
                        self.patches_tree.insert("", "end", values=(
                            row['filename'],
                            location,
                            f"{float(row['mean']):.2e}",
                            f"{float(row['std']):.2e}",
                            f"{float(row['skew']):.2f}",
                            f"{float(row['kurt']):.2f}",
                            f"{float(row['pval']):.2e}"
                        ))
                
                # Update statistics summary
                self.update_stats_summary()
        except Exception as e:
            print(f"Error loading patches: {e}")

    def log_current_patch(self):
        """Log the most recent patch data to a CSV and update the display"""
        if not self.last_patch_info:
            print("No patch data to log.")
            return

        patch_data = self.last_patch_info

        # Write to CSV
        file_exists = os.path.exists(LOGS_CSV_PATH)
        with open(LOGS_CSV_PATH, mode="a", newline="") as csvfile:
            fieldnames = [
                "filename", "orig_x0", "orig_x1", "orig_y0", "orig_y1",
                "mean", "std", "skew", "kurt", "pval"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "filename": patch_data["filename"],
                "orig_x0": patch_data["orig_x0"],
                "orig_x1": patch_data["orig_x1"],
                "orig_y0": patch_data["orig_y0"],
                "orig_y1": patch_data["orig_y1"],
                "mean": patch_data["mean"],
                "std": patch_data["std"],
                "skew": patch_data["skew"],
                "kurt": patch_data["kurt"],
                "pval": patch_data["pval"]
            })

        # Reload the patches display and update statistics
        self.load_patches()
        print(f"Patch logged to {LOGS_CSV_PATH}")
    
    def open_denoise_window(self):
        """Open a separate window for testing denoising algorithms."""
        if self.current_image_data is None:
            return
        DenoiseWindow(self, image_data=self.current_image_data)
        
def main():
    viewer = PDSImageViewer()
    viewer.mainloop()

if __name__ == "__main__":
    main()