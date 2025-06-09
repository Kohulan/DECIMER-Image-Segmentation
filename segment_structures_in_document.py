"""
* This Software is under the MIT License
* Refer to LICENSE or https://opensource.org/licenses/MIT for more information
* Written by ©Kohulan Rajan 2020
"""

import sys
import os
import argparse
from decimer_segmentation import segment_chemical_structures_from_file
from decimer_segmentation import save_images, get_bnw_image, get_square_image


def main():
    """
    This script segments chemical structures in a document, saves the original
    segmented images as well as a binarized image and a an undistorted square
    image
    """
    parser = argparse.ArgumentParser(
        description="Segment chemical structures from documents"
    )
    parser.add_argument("input_path", help="Path to the input document (PDF or image)")
    parser.add_argument("--output_dir", help="Custom output directory (optional)")

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        sys.exit(1)

    try:
        # Extract chemical structure depictions and save them
        print(f"Processing: {args.input_path}")
        raw_segments = segment_chemical_structures_from_file(args.input_path)

        if not raw_segments:
            print("No chemical structures found in the document.")
            return

        # Determine output directory
        if args.output_dir:
            segment_dir = os.path.join(args.output_dir, "segments")
        else:
            segment_dir = os.path.join(f"{args.input_path}_output", "segments")

        # Create output directory if it doesn't exist
        os.makedirs(segment_dir, exist_ok=True)

        filename_base = os.path.splitext(os.path.basename(args.input_path))[0]

        print(f"Found {len(raw_segments)} chemical structure(s). Saving segments...")

        # Save original segments
        save_images(raw_segments, segment_dir, f"{filename_base}_orig")

        # Get binarized segment images
        binarized_segments = [get_bnw_image(segment) for segment in raw_segments]
        save_images(binarized_segments, segment_dir, f"{filename_base}_bnw")

        # Get segments in size 299*299 and save them
        normalized_segments = [
            get_square_image(segment, 299) for segment in raw_segments
        ]
        save_images(normalized_segments, segment_dir, f"{filename_base}_norm")

        print(f"Segments saved at {segment_dir}")

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
