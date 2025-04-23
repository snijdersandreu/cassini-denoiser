import os
import glob
import re
import shutil
import sys
from pds import parse_header
from pathlib import Path
from collections import Counter

ROOT_DIR = "/Volumes/PortableSSD/uni/TFG/cassini-denoiser/data/COISS_2006_images/COISS_2006/data"

def find_labels(root_dir):
    """Recursively find all .LBL files under root_dir."""
    return sorted(Path(root_dir).rglob("*.LBL"))  # Ensures deep search in subdirectories


def duplicate_dataset(source_root, destination):
    """Duplicate the original dataset before applying filtering."""
    if not os.path.exists(destination):
        print(f"Duplicating dataset from {source_root} to {destination}...")
        shutil.copytree(source_root, destination)
        print("Dataset duplication complete.")
    else:
        print(f"Dataset already duplicated at {destination}, proceeding with filtering.")

def filter_and_delete(lbl_files, dataset_root):
    """Filter .LBL/.IMG pairs and delete non-matching files in the duplicated dataset."""
    for lbl_file in lbl_files:
        metadata = parse_header(lbl_file)
        # Check if the file meets the criteria
        if not (metadata["lines"] in [1024, "1024"] and
                metadata["line_samples"] in [1024, "1024"] and
                str(metadata["sample_bits"]) == "32" and
                metadata["sample_type"] == "PC_REAL" and
                metadata["data_conversion_text"] == "'Converted from 8 to 12 bits'" and
                metadata["units"] == "'I/F'"):

            img_file = lbl_file.with_suffix(".IMG")

            print(f"Deleting non-matching files: {lbl_file} and {img_file}")
            os.remove(lbl_file)
            if img_file.exists():
                os.remove(img_file)

def main():
    source_root = ROOT_DIR
    destination = ROOT_DIR + "_filtered_1"

    if not os.path.exists(source_root):
        print(f"Error: The directory {source_root} does not exist!")
        sys.exit(1)

    # Step 1: Duplicate the dataset
    duplicate_dataset(source_root, destination)

    # Step 2: Find .LBL files in the duplicated dataset
    lbl_files = find_labels(destination)
    print(f"Found {len(lbl_files)} .LBL files in duplicated dataset.")

    # Step 3: Apply filtering and delete non-matching files
    filter_and_delete(lbl_files, destination)
    
    print(f"Filtering completed. The filtered dataset is available at: {destination}")

if __name__ == '__main__':
    main()