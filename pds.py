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
        "sample_type": None
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

    # Return configuration parameters
    return config_params


# Read a PDS image from a binary file
def read_image(header_file_path, image_file_path):
    """
    Read a PDS image from a binary file.
    This generalized function supports both uncalibrated (8-bit with a prefix) and calibrated (32-bit PC_REAL) images.
    """

    # Parse the header file for configuration parameters
    config = parse_header(header_file_path)

    # Determine the number of bytes per sample and the numpy data type from SAMPLE_BITS (and SAMPLE_TYPE, if needed)
    sample_bits = config["sample_bits"]
    if sample_bits == 8:
        sample_dtype = np.uint8
        sample_byte_size = 1
    elif sample_bits == 16:
        sample_dtype = np.uint16
        sample_byte_size = 2
    elif sample_bits == 32:
        # Typically calibrated images use PC_REAL which we interpret as 32-bit float
        sample_dtype = np.float32
        sample_byte_size = 4
    else:
        raise ValueError("Unsupported SAMPLE_BITS value: {}".format(sample_bits))

    lines = config["lines"]
    samples = config["line_samples"]
    prefix_bytes = config.get("line_prefix_bytes", 0)

    # Allocate array to hold the image; note that for calibrated images the file includes an extra non-image row.
    image_array = np.zeros((lines, samples), dtype=sample_dtype)

    # Calculate offset where the image payload starts
    image_start_offset = config["record_length"] * (config["image_start_record"] - 1)

    with open(image_file_path, 'rb') as file:
        file.seek(image_start_offset)
        # For calibrated images (SAMPLE_TYPE 'PC_REAL'), skip one extra row that holds non-image information.
        if config.get("sample_type", "").upper() == "PC_REAL":
            file.read(samples * sample_byte_size)
        for i in range(lines):
            # For uncalibrated images, skip the line prefix bytes; for calibrated images prefix_bytes is 0.
            if prefix_bytes > 0:
                file.read(prefix_bytes)
            # Read one full line (note that each sample occupies sample_byte_size bytes)
            line_data = file.read(samples * sample_byte_size)
            if len(line_data) != samples * sample_byte_size:
                raise ValueError("Incomplete data for line {}: expected {} bytes, got {} bytes".format(
                    i, samples * sample_byte_size, len(line_data)))
            image_array[i, :] = np.frombuffer(line_data, dtype=sample_dtype, count=samples)

    # Return the processed image
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
