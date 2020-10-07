'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
import sys
import glob

import warnings
warnings.filterwarnings("ignore")


from Utils import pdf_2_img_Convert
from Utils import Image_resizer
import Detect_and_save_segmentation

def main():
	if len(sys.argv) != 2:
		print("\"Usage of this function: convert.py input_path")
	if len(sys.argv) == 2:
		output_directory = pdf_2_img_Convert.convert(sys.argv[1])
		converted_images = glob.glob(output_directory+"/*.png") # Get all images converted into JPEGs

		for i in converted_images:
			print(i)
			directory,info = Detect_and_save_segmentation.get_segments(i,output_directory)

		path_to_segments = glob.glob(directory+"*.png")
		print("Cleaning up images")

		for im_pth in path_to_segments:
			Image_resizer.get_clean_segments(im_pth)

	sys.exit(1)

if __name__ == '__main__':
    main()