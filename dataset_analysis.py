import os
import glob
from collections import Counter
import sys

ROOT_DIR = "/Volumes/PortableSSD/uni/TFG/cassini-denoiser/data/COISS_2006_images/COISS_2006_filtered"

def find_labels(root_dir):
    """Recursively find all .LBL files under root_dir."""
    pattern = os.path.join(root_dir, '**', '*.LBL')  # Ensure deep recursive search
    return glob.glob(pattern, recursive=True)

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: The directory {ROOT_DIR} does not exist!")
        sys.exit(1)
        
    lbl_files = find_labels(ROOT_DIR)
    print(f"Found {len(lbl_files)} .LBL files in {ROOT_DIR}")
    if lbl_files:
        print("Sample files:")
        for lbl in lbl_files[:5]:  # Print first 5 files to verify
            print(f"  {lbl}")

    # We want to track these keywords:
    keywords = ["lines", "line_samples", "sample_bits", "sample_type", "data_conversion_type", "data_conversion_text", "units"]

    # Create a dictionary of counters keyed by each keyword
    counters = {k: Counter() for k in keywords}

    for lbl_file in lbl_files:
        print(f"Processing: {lbl_file}")
        from pds import parse_header
        config_params = parse_header(lbl_file)
        for k in keywords:
            val = config_params.get(k, None)
            if val is not None:
                counters[k][str(val)] += 1

    # Print summaries
    for k in keywords:
        print(f"\n{k} frequency:")
        for val, count in counters[k].items():
            print(f"  {val}: {count}")

    print("Final extracted counts:", counters)

if __name__ == '__main__':
    main()