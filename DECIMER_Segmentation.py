'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
import sys
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


from Utils import pdf_2_img_Convert
from Utils import Image_resizer
import Detect_and_save_segmentation
import multiprocessing as mp
from functools import partial

def main():
	if len(sys.argv) != 2:
		print("\"Usage of this function: convert.py input_path")
	if len(sys.argv) == 2:
		output_directory = pdf_2_img_Convert.convert(sys.argv[1])
		converted_images = glob.glob(output_directory+"/*.png") # Get all images converted into JPEGs
		pool = mp.Pool(4)

		Pool_function = partial(Detect_and_save_segmentation.get_segments,output_directory)

		expanded_masks = pool.map(Pool_function,converted_images)

		path_to_segments = glob.glob(expanded_masks[0]+"*.png")
		print("Cleaning up images")
		pool.map(Image_resizer.get_clean_segments, path_to_segments)
		
		pool.close()
		pool.join()

	sys.exit(1)

if __name__ == '__main__':
	main()