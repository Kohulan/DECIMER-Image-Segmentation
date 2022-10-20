"""
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
"""
import sys
import os
from decimer_segmentation import segment_chemical_structures_from_file
from decimer_segmentation import save_images, get_bnw_image, get_square_image


def main():
    """
    This script segments chemical structures in a document, saves the original
    segmented images as well as a binarized image and a an undistorted square
    image
    """
    if len(sys.argv) != 2:
        print("Usage of this function: convert.py input_path")
    if len(sys.argv) == 2:
        # Extract chemical structure depictions and save them
        raw_segments = segment_chemical_structures_from_file(sys.argv[1])
        segment_dir = os.path.join(f"{sys.argv[1]}_output", "segments")
        save_images(
            raw_segments, segment_dir, f"{os.path.split(sys.argv[1])[1][:-4]}_orig"
        )
        # Get binarized segment images
        binarized_segments = [get_bnw_image(segment) for segment in raw_segments]
        save_images(
            binarized_segments, segment_dir, f"{os.path.split(sys.argv[1])[1][:-4]}_bnw"
        )
        # Get segments in size 299*299 and save them
        normalized_segments = [
            get_square_image(segment, 299) for segment in raw_segments
        ]
        save_images(
            normalized_segments,
            segment_dir,
            f"{os.path.split(sys.argv[1])[1][:-4]}_norm",
        )
        print(f"Segments saved at {segment_dir}.")


if __name__ == "__main__":
    main()
