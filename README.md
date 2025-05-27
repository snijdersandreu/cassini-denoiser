# CASSINI DENOISER

A collection of tools for noise analysis and image denoising, primarily focused on images from the Cassini-Huygens mission. This project provides a graphical user interface (GUI) for applying various denoising algorithms, viewing PDS (Planetary Data System) files, and several scripts for data analysis and simulation.

## Features

*   **Interactive Denoising GUI**: `denoise_window.py` offers a user-friendly interface to load images, apply denoising algorithms, and visualize results.
*   **PDS File Support**: Includes tools (`pds.py`, `pds_viewer.py`) for parsing and viewing Cassini .IMG and .LBL files.
*   **Multiple Denoising Algorithms**:
    *   Non-Local Means (NLM)
    *   Starlet Transform Denoising
    *   BM3D
    *   Total Variation (Edge-preserving regularization)
    *   Wiener Filter
*   **Image Analysis Tools**: Scripts for noise characterization, Signal-to-Noise Ratio (SNR) analysis, contrast analysis, and histogram analysis.
*   **Image Simulation**: Utilities like `adapt_simulation.py` for creating simulated datasets.

## Project Structure

*   `denoise_window.py`: The main graphical user interface for image denoising.
*   `pds_viewer.py`: A tool for viewing PDS images and their metadata.
*   `denoising_algorithms/`: Contains Python implementations of various denoising algorithms.
*   `analyze_*.py`, `contrast_analysis.py`, etc.: Various scripts for specific analysis tasks.
*   `pds.py`: Core module for PDS file parsing.
*   `image_utils.py`: Utility functions for image manipulation.
*   `LUT/`: Directory for Lookup Tables.
*   `data/`: Intended for storing sample or user-provided image data (create if it doesn't exist).
*   `requirements.txt`: Lists project dependencies.
*   `LICENSE`: Project license information.

## Setup

### Python Virtual Environment

After cloning this repository, it's highly recommended to create and activate a Python virtual environment.

If you are using an IDE like PyCharm, it may handle this for you. Otherwise, create one manually:

```shell
python3 -m venv .venv
```

Activate it (on macOS/Linux):

```shell
source .venv/bin/activate
```

Or on Windows:

```shell
.venv\Scripts\activate
```

### Python Packages

With the virtual environment activated, install the required Python packages:

```shell
pip install -r requirements.txt
```

## Usage

### Main Denoising Application

To run the main GUI for denoising:

```shell
python denoise_window.py
```

### PDS Image Viewer

To run the PDS image viewer:

```shell
python pds_viewer.py
```

Other Python scripts in the repository can be run similarly, e.g., `python analyze_snr.py`. Some scripts may accept command-line arguments; refer to their source code for details.

## Dependencies

All Python dependencies are listed in `requirements.txt`.

## License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for more details.
