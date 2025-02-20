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
    """

    # Set of required configuration parameters
    config_params = {
        "record_length": None,
        "image_start_record": None,
        "lines": None,
        "line_samples": None,
        "sample_bits": None,
        "line_prefix_bytes": None
    }

    # Open header file and read all the contents
    with open(header_file_path, 'r') as file:
        header_content = file.readlines()

    # Parse the required lines
    for line in header_content:
        if "RECORD_BYTES" in line:
            config_params["record_length"] = int(line.split('=')[1].strip())
        elif "^IMAGE" in line:
            config_params["image_start_record"] = int(line.split('=')[1].split(',')[1].split(')')[0].strip())
        elif "LINES" in line:
            config_params["lines"] = int(line.split('=')[1].strip())
        elif "LINE_SAMPLES" in line:
            config_params["line_samples"] = int(line.split('=')[1].strip())
        elif "SAMPLE_BITS" in line:
            config_params["sample_bits"] = int(line.split('=')[1].strip())
        elif "LINE_PREFIX_BYTES" in line:
            config_params["line_prefix_bytes"] = int(line.split('=')[1].strip())

    # Return configuration parameters
    return config_params


# Read a PDS image from a binary file
def read_image(header_file_path, image_file_path):
    """
    Process the image using the parameters from the header file.
    """

    # Parse the header file for configuration parameters
    config = parse_header(header_file_path)

    # Define the data array for the image
    image_array = np.zeros((config["lines"], config["line_samples"]), dtype=np.uint8)

    # Open binary file for reading
    with open(image_file_path, 'rb') as file:

        # Skipp non-image data
        file.seek(config["record_length"] * (config["image_start_record"] - 1))

        # Read each line of the image
        for line in range(config["lines"]):

            # Skip the line prefix
            file.read(config["line_prefix_bytes"])

            # Read the line data
            line_data = file.read(config["line_samples"])
            image_array[line, :] = np.frombuffer(line_data, dtype=np.uint8)

    # Return the processed image
    return image_array


# Write an updated image stream into a binary file, keeping the header information of the original file
def save_image(image_data, new_image_file_path, original_image_file_path, header_file_path):
    """
    Save the modified image data into a file, preserving the original format and structure.

    :param image_data: Numpy array containing the modified image data.
    :param original_image_file_path: Path to the original image file.
    :param new_image_file_path: Path where the new image file will be saved.
    :param header_file_path: Path where the header file will be saved.
    """

    # Parse the header file for configuration parameters
    config = parse_header(header_file_path)

    # Open original/new image files for reading/writing
    with open(original_image_file_path, 'rb') as original_file, open(new_image_file_path, 'wb') as new_file:

        # Copy the initial non-image part of the file
        original_file.seek(0)
        non_image_data = original_file.read(config['record_length'] * (config['image_start_record'] - 1))
        new_file.write(non_image_data)

        # Write modified image data with line prefixes
        for line in range(config["lines"]):

            # Copy the line prefix from the original file
            original_file.seek(config['record_length'] * (config['image_start_record'] - 1) +
                               line * (config['line_samples'] + config['line_prefix_bytes']))
            line_prefix = original_file.read(config['line_prefix_bytes'])
            new_file.write(line_prefix)

            # Write the modified line data
            new_file.write(image_data[line, :])


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
