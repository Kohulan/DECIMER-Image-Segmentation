'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

import warnings
warnings.filterwarnings("ignore")

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import moldetect
from Scripts import complete_structure 


# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

def main():

	parser = argparse.ArgumentParser(description="Select the chemical structures from a scanned literature and save them")
    # Input Arguments
	parser.add_argument(
		'--input',
		help='Enter the input filename',
		required=True
	)

	args = parser.parse_args()
	
	IMAGE_PATH = (args.input)
	r = get_masks(IMAGE_PATH)

	#Expand Masks
	image = skimage.io.imread(IMAGE_PATH)
	expanded_masks = complete_structure.complete_structure_mask(image_array = image, mask_array = r['masks'], debug = False)
	final_seg = save_segments(expanded_masks,IMAGE_PATH)

	print(final_seg)



def get_masks(IMAGE_PATH):
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join("model_trained//mask_rcnn_molecule_0045.h5")

	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	# Image Path
	#IMAGE_DIR = os.path.join("/media/data_drive/Kohulan/After-Meeting_20190522/MaskRCNN/Test")
	image = skimage.io.imread(IMAGE_PATH)
	config = moldetect.MolDetectConfig()

	# Override the training configurations with a few
	# changes for inferencing.
	class InferenceConfig(config.__class__):
		# Run detection on one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()

	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	class_names=['BG', 'Molecule']

	# Run detection
	results = model.detect([image], verbose=1)

	r = results[0]

	return r

def save_segments(expanded_masks,IMAGE_PATH):
	mask = expanded_masks

	for i in range(mask.shape[2]):
		image = cv2.imread(os.path.join(IMAGE_PATH), -1)

		for j in range(image.shape[2]):
			image[:,:,j] = image[:,:,j] * mask[:,:,i]

		original = image.copy()

		#Remove unwanted background
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_,thresholded = cv2.threshold(grayscale,0,255,cv2.THRESH_OTSU)
		bbox = cv2.boundingRect(thresholded)
		x, y, w, h = bbox
		foreground = image[y:y+h, x:x+w]

		masked_image = np.zeros(image.shape).astype(np.uint8)
		masked_image = visualize.apply_mask(masked_image, mask[:, :, i],[1,1,1])
		masked_image = Image.fromarray(masked_image)
		masked_image = masked_image.convert('RGB')

		im_gray = cv2.cvtColor(np.asarray(masked_image), cv2.COLOR_RGB2GRAY)
		(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		#Removal of transparent layer - black background
		_,alpha = cv2.threshold(im_bw,0,255,cv2.THRESH_BINARY)
		b, g, r = cv2.split(image)
		rgba = [b,g,r, alpha]
		dst = cv2.merge(rgba,4)
		background = dst[y:y+h, x:x+w]
		trans_mask = background[:,:,3] == 0
		background[trans_mask] = [255, 255, 255, 255]
		new_img = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

		#save segments
		filename = "output/segment_%d.png"%i
		cv2.imwrite(filename,new_img)
		

	return "Completed, Segments saved inside the ouput folder!"


if __name__ == '__main__':
    main()

