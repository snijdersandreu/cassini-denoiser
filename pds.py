# Work with PDS Image file
#
# Example usage (command line):
#   python3 pds.py --header 'N1473172656_1.LBL' --image 'N1473172656_1.IMG'
#
# Example usage (imported):
#   header_file_path = 'N1473172656_1.LBL'
#   image_file_path = 'N1473172656_1.IMG'
#   process_image(header_file_path, image_file_path)

# Import system modules
import argparse

# Import external modules
import numpy as np
import matplotlib.pyplot as plt


# Read the configuration parameters from a header file
def parse_header(header_file_path):
    """
    Parse the header file to extract configuration parameters.
    Supports both uncalibrated (8-bit with line prefixes) and calibrated (32-bit, no prefix) images.
    """
    # Set of required parameters. Defaults: line_prefix_bytes=0 if not provided.
    config_params = {
        "record_length": None,
        "image_start_record": None,
        "lines": None,
        "line_samples": None,
        "sample_bits": None,
        "line_prefix_bytes": 0,  # default to 0 if header has no such entry (as in calibrated images)
        "sample_type": None,
        "data_conversion_type": None,
        "data_conversion_text": None,
        "units": None
    }

    # Open header file and read all the contents
    with open(header_file_path, 'r') as file:
        header_content = file.readlines()

    # Parse the required lines
    for line in header_content:
        line = line.strip()
        if "RECORD_BYTES" in line:
            config_params["record_length"] = int(line.split('=')[1].strip())
        elif "^IMAGE" in line:
            # Format: ^IMAGE = ("FILENAME",record)
            config_params["image_start_record"] = int(line.split('=')[1].split(',')[1].split(')')[0].strip())
        elif line.startswith("LINES") and config_params["lines"] is None:
            config_params["lines"] = int(line.split('=')[1].strip())
        elif "LINE_SAMPLES" in line:
            config_params["line_samples"] = int(line.split('=')[1].strip())
        elif "SAMPLE_BITS" in line:
            config_params["sample_bits"] = int(line.split('=')[1].strip())
        elif "LINE_PREFIX_BYTES" in line:
            config_params["line_prefix_bytes"] = int(line.split('=')[1].strip())
        elif "SAMPLE_TYPE" in line:
            # Remove quotes if present.
            config_params["sample_type"] = line.split('=')[1].strip().strip('"')
        elif "DATA_CONVERSION_TYPE" in line:
            config_params["data_conversion_type"] = line.split('=')[1].strip().strip('"')
        elif "DATA_CONVERSION_TEXT" in line:
            config_params["data_conversion_text"] = line.split('=')[1].strip().strip('"')
        elif "UNITS" in line:
            config_params["units"] = line.split('=')[1].strip().strip('"')

    # Return configuration parameters
    return config_params


# Read a PDS image from a binary file
def read_image(header_file_path, image_file_path, keep_float=True):
    """
    Read a PDS image from a binary file.
    - If keep_float=True and the image is 32-bit float (PC_REAL),
      we keep the full float range (no min–max scaling).
    - Otherwise, for 8-bit or 16-bit data, we just read them in,
      then optionally convert to float64 for subsequent analysis.
    """

    config = parse_header(header_file_path)

    sample_bits = config["sample_bits"]
    sample_type = config.get("sample_type", "").upper()
    lines = config["lines"]
    samples = config["line_samples"]
    prefix_bytes = config.get("line_prefix_bytes", 0)
    record_length = config["record_length"]
    image_start_record = config["image_start_record"]

    # Decide the numpy dtype
    if sample_bits == 8:
        sample_dtype = np.uint8
        sample_byte_size = 1
    elif sample_bits == 16:
        sample_dtype = np.uint16
        sample_byte_size = 2
    elif sample_bits == 32 and sample_type == "PC_REAL":
        sample_dtype = np.float32
        sample_byte_size = 4
    else:
        raise ValueError(f"Unsupported sample_bits={sample_bits}, sample_type={sample_type}")

    image_array = np.zeros((lines, samples), dtype=sample_dtype)

    # The offset in bytes to the start of the image data
    image_start_offset = record_length * (image_start_record - 1)

    with open(image_file_path, 'rb') as file:
        file.seek(image_start_offset)

        # Some Cassini images have an extra row for PC_REAL - 
        # comment out or enable if truly needed:
        # if sample_type == "PC_REAL":
        #     file.read(samples * sample_byte_size)  # skip extra row

        for i in range(lines):
            if prefix_bytes > 0:
                file.read(prefix_bytes)
            line_data = file.read(samples * sample_byte_size)
            if len(line_data) != samples * sample_byte_size:
                raise ValueError(f"Incomplete data for line {i}.")
            image_array[i, :] = np.frombuffer(line_data, dtype=sample_dtype, count=samples)

    # --- If it's float data, optionally keep it unscaled ---
    if sample_bits == 32 and sample_type == "PC_REAL":
        if keep_float:
            # We do NOT min–max scale. Just convert to double if you want full numeric precision.
            image_array = image_array.astype(np.float64)
        else:
            # Possibly min–max scale for quicklook only (NOT recommended for noise analysis!)
            fmin, fmax = image_array.min(), image_array.max()
            denom = fmax - fmin
            if denom < 1e-12:
                scaled = np.zeros_like(image_array, dtype=np.uint8)
            else:
                scaled = 255.0 * (image_array - fmin) / denom
            image_array = scaled.astype(np.uint8)
    else:
        # For integer data (8/16 bit), just promote to float64 for analysis
        image_array = image_array.astype(np.float64)

    return image_array


# Write an updated image stream into a binary file, keeping the header information of the original file
def save_image(image_data, new_image_file_path, original_image_file_path, header_file_path):
    """
    Save the modified image data into a new binary file, preserving the original header and structure.
    This version adapts to both calibrated (no line prefix) and uncalibrated (with line prefix) formats.
    """

    # Parse the header file for configuration parameters
    config = parse_header(header_file_path)

    # Determine sample byte size based on SAMPLE_BITS.
    sample_bits = config["sample_bits"]
    if sample_bits == 8:
        sample_byte_size = 1
    elif sample_bits == 16:
        sample_byte_size = 2
    elif sample_bits == 32:
        sample_byte_size = 4
    else:
        raise ValueError("Unsupported SAMPLE_BITS value: {}".format(sample_bits))

    lines = config["lines"]
    samples = config["line_samples"]
    prefix_bytes = config.get("line_prefix_bytes", 0)

    with open(original_image_file_path, 'rb') as orig_file, open(new_image_file_path, 'wb') as new_file:
        # Copy the non-image (header) portion of the file.
        header_offset = config["record_length"] * (config["image_start_record"] - 1)
        orig_file.seek(0)
        new_file.write(orig_file.read(header_offset))

        for i in range(lines):
            # For uncalibrated images, copy the line prefix.
            if prefix_bytes > 0:
                seek_offset = header_offset + i * (prefix_bytes + samples * sample_byte_size)
                orig_file.seek(seek_offset)
                prefix = orig_file.read(prefix_bytes)
                new_file.write(prefix)
            # Write the image data line. Convert the numpy row to bytes.
            new_file.write(image_data[i, :].tobytes())


# Plot a PDS image
def plot_image(image_data, plot_file_path=None, title_str=None, show=True):
    """
    Plot the image.
    """

    # Plot the image
    plt.imshow(image_data, cmap='gray')

    # Title
    if title_str:
        plt.title(title_str)

    # Save the plot
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')

    # Display the plot
    if show:
        plt.show()


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()

    # Read the image from the PDS file
    image_raw = read_image(header_file_path=args.header, image_file_path=args.image)

    # Display the image
    plot_image(image_raw)
