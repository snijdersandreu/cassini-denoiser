#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Variable descriptions for Saturn simulation
VARIABLE_DESCRIPTIONS = {
    'ETA': 'Model vertical (η-) coordinate (1=surface, 0=model top)',
    'PERTS': 'Perturbation surface pressure',
    'PVORT': 'Potential vorticity (s⁻¹)',
    'TRACER': 'Mass-mixing-ratio of inert tracer (clouds/atmospheric constituents)',
    'U': 'Zonal (east-west) wind component (m s⁻¹)',
    'V': 'Meridional (north-south) wind component (m s⁻¹)',
    'WINDX': 'Total horizontal wind speed (m s⁻¹)'
}

# Custom colormap for Saturn's surface
def create_saturn_colormap():
    colors = [(0.0, 0.0, 0.0),           # Black for no tracer
              (0.8, 0.7, 0.5),           # Tan/brown for light tracer
              (0.9, 0.8, 0.6),           # Light tan
              (0.95, 0.9, 0.7),          # Very light tan
              (1.0, 1.0, 1.0)]           # White for maximum tracer
    return LinearSegmentedColormap.from_list('saturn_surface', colors)

def load_simulation_as_image(csv_path, variable='WINDX',
                             lon_col='Points:0', lat_col='Points:1',
                             lon_in_radians=True):
    """
    Loads the simulation CSV and reshapes the specified variable into a 2D grid.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Convert longitude to degrees if needed
    if lon_in_radians:
        df['lon_deg'] = np.degrees(df[lon_col])
    else:
        df['lon_deg'] = df[lon_col]

    # Pivot to grid: rows = lat, columns = lon
    pivot = df.pivot(index=lat_col, columns='lon_deg', values=variable)

    # Sort axes for correct image orientation
    pivot = pivot.sort_index(ascending=False)  # lat descending
    pivot = pivot.sort_index(axis=1)          # lon ascending

    grid = pivot.values
    return grid

def get_colormap(variable):
    """
    Returns an appropriate colormap for the given variable.
    """
    if variable == 'TRACER':
        return create_saturn_colormap()
    elif variable in ['U', 'V', 'WINDX', 'PVORT']:
        return 'binary'  # Simple black and white colormap
    elif variable == 'ETA':
        return 'plasma'   # Good for vertical coordinates
    else:
        return 'binary'  # Default to black and white

def show_simulation_image(csv_path, variable='WINDX', save_path=None):
    """
    Visualizes a variable from the Saturn simulation as a 2D image.
    """
    grid, lon_vals, lat_vals = load_simulation_as_image(csv_path, variable)
    
    plt.figure(figsize=(12, 8), dpi=300)  # Higher DPI for better quality
    
    # Create the plot as a pure pixel grid
    im = plt.imshow(
        grid,
        cmap=get_colormap(variable),
        aspect='equal'  # Make pixels square
    )
    
    # Remove axis labels and ticks since we're treating it as pure pixels
    plt.axis('off')
    
    # Add colorbar with appropriate label
    cbar = plt.colorbar(im)
    if variable == 'TRACER':
        cbar.set_label('Cloud/Constituent Density')
    elif variable == 'PVORT':
        cbar.set_label('Potential Vorticity (s⁻¹)')
    else:
        cbar.set_label(f'{variable} ({VARIABLE_DESCRIPTIONS.get(variable, "")})')
    
    # Add title
    if variable == 'TRACER':
        plt.title(f'Saturn Surface Visualization (Cloud/Constituent Distribution)\nValue range: {grid.min():.2e} to {grid.max():.2e}')
    elif variable == 'PVORT':
        plt.title(f'Saturn Potential Vorticity\nValue range: {grid.min():.2e} to {grid.max():.2e} s⁻¹')
    else:
        plt.title(f'Saturn Simulation: {variable}')
    
    if save_path:
        # Save with maximum quality settings
        plt.savefig(
            save_path,
            dpi=300,  # High DPI for better quality
            bbox_inches='tight',
            pad_inches=0.1,
            format='png',
            transparent=False,
            facecolor='white',
            edgecolor='none'
        )
        print(f"\nSaved visualization to: {save_path}")
    else:
        plt.show()

def save_raw_image(csv_path, variable='WINDX', save_path=None):
    """
    Saves the raw simulation data as a black and white image.
    """
    # Load the grid data
    grid = load_simulation_as_image(csv_path, variable)
    
    # Normalize the data to 0-255 range for image
    grid_normalized = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype(np.uint8)
    
    # Create image from array
    img = Image.fromarray(grid_normalized, mode='L')  # 'L' mode is for grayscale
    
    # Save as PNG with maximum quality
    if save_path:
        img.save(save_path, format='PNG', optimize=False)
        print(f"\nSaved raw data image to: {save_path}")
        print(f"Image size: {img.size}")
        print(f"Original value range: {grid.min():.2e} to {grid.max():.2e}")
    else:
        print("No save path provided")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse.py <path_to_csv> [variable] [save_path]")
        print("\nAvailable variables:")
        for var, desc in VARIABLE_DESCRIPTIONS.items():
            print(f"  {var}: {desc}")
        sys.exit(1)

    path = sys.argv[1]
    var = sys.argv[2] if len(sys.argv) > 2 else 'WINDX'
    save_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    save_raw_image(path, variable=var, save_path=save_path)

if __name__ == '__main__':
    main()