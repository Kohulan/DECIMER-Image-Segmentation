import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools

from PIL import Image
from matplotlib.patches import Polygon
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import img_as_float, img_as_int

from imantics import Mask, Polygons
from typing import Dict, List, Tuple

from multiprocessing import Pool


def plot_it_multiple(image_array, bounding_boxes):
	'''This function shows the plot of a given image (np.array([...])) with a given list of bounding boxes np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	fig,ax = plt.subplots(1)
	imgplot = ax.imshow(image_array)
	for bounding_box in bounding_boxes:
		bounding_box = Polygon(bounding_box[0], linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(bounding_box)
	#plt.savefig("./image_"+str(image_counter)+".png")
	plt.imshow(image_array)
	plt.show()
	return

def plot_it(image_array, bounding_box):
	'''This function shows the plot of a given image (np.array([...])) with a given bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	bounding_box = Polygon(bounding_box, linewidth=1,edgecolor='r',facecolor='none')
	fig,ax = plt.subplots(1)
	imgplot = ax.imshow(image_array)
	ax.add_patch(bounding_box)
	#plt.savefig("./image_"+str(image_counter)+".png")
	plt.show()
	return

def define_bounding_box_center(bounding_box):
	'''This function return the center np.array([x,y]) of a given bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	x_values = np.array([])
	y_values = np.array([])
	for node in bounding_box:
		x_values = np.append(x_values, node[0])
		y_values = np.append(y_values, node[1])
	return np.array([np.mean(x_values), np.mean(y_values)])

def define_edge_line(node_1, node_2):
	'''This function returns a linear function between two given points in a 2-dimensional space'''
	#if node_1[0]-node_2[0] != 0:
	slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0])
	#else:
	#	slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0]+0.00000000000001) # avoid dividing by zero, this line should not be necesary anymore.
	#intercept = y - m*x
	intercept = node_1[1] - slope * node_1[0]
	return slope, intercept

def euklidian_distance(node_1, node_2):
	'''This function returns the euklidian distance between two given points in a 2-dimensional space.'''
	return np.sqrt((node_2[0]-node_1[0]) ** 2 + (node_2[1]-node_1[1])**2)
    
def set_x_diff_range(x_diff, euklidian_distance, image_array):
	'''This function takes the amount of steps on an edge and returns the corresponding list of steps in random order'''
	blur_factor = int(image_array.shape[1]/500) if image_array.shape[1]/500 >= 1 else 1
	if x_diff > 0:
		x_diff_range = np.arange(0, x_diff, blur_factor/euklidian_distance)
		#x_diff_range = np.arange(0, x_diff, 00.1) 
		
	else:
		x_diff_range = np.arange(0, x_diff, -blur_factor/euklidian_distance)
		#x_diff_range = np.arange(0, x_diff, -00.1)
		
	np.random.shuffle(x_diff_range)
	return x_diff_range

def define_next_pixel_to_check(bounding_box, node_index, step, image_shape):
	'''This function returns the next pixel to check in the image (along the edge of the bounding box). 
	In the case that it tries to check a pixel outside of the image, this is corrected.'''
	# Define the edges of the bounding box
	slope, intercept = define_edge_line(bounding_box[node_index], bounding_box[node_index-1])
	x = int(bounding_box[node_index-1][0] + step)
	y = int((slope * (bounding_box[node_index-1][0] + step)) + intercept)
	if y >= image_shape[0]:
		y = image_shape[0]-1
	if y < 0:
		y = 0
	if x >= image_shape[1]:
		x = image_shape[1]-1
	if x < 0:
		x = 0
	return x,y

def adapt_x_values(bounding_box, node_index, image_shape):
	'''If two nodes form a vertical edge the function that descripes that edge will in- or decrease infinitely with dx so we need to alter those nodes. 
	This function returns a bounding box where the nodes are altered depending on their relative position to the center of the bounding box.
	If a node is at the border of the image, then it is not changed.'''
	if bounding_box[node_index][0] != image_shape[1]:
		bounding_box_center = define_bounding_box_center(bounding_box)
		if bounding_box[node_index][0] < bounding_box_center[0]:
			bounding_box[node_index][0] = bounding_box[node_index][0] - 1
		else: 
			bounding_box[node_index][0] = bounding_box[node_index][0] + 1
	return bounding_box

def mask_2_polygons(mask_array):
	'''This function takes a mask array and returns a list of polygon bounding boxes (node coordinates)'''
	
	#imantics is built on openCV. The mask needs to be inverted so that the findcontours function works.
	inverted_mask = np.invert(mask_array).astype(int)
	#split the mask array with all masks into n arrays with shape (x,y,1) (one mask_array per molecule)
	polygons = []
	for mask_array in np.dsplit(inverted_mask, mask_array.shape[2]):
		#Transform mask into bounding box polygon, append it to list and return list
		current_mask = Mask(mask_array)
		polygons.append(Polygons.from_mask(current_mask).points)
	return polygons

def binarize_image(image_array, threshold = "otsu"):
	'''This function takes a Numpy array that represents an RGB image and returns the binarized form of that image
	by applying the otsu threshold.'''
	grayscale = rgb2gray(image_array)
	if threshold == "otsu":
		threshold = threshold_otsu(grayscale)
	binarized_image_array = grayscale > threshold
	return binarized_image_array

def define_relevant_polygons(polygons):
	'''Sometimes, the mask R CNN model produces a weird output with one big masks and some small irrelevant islands.
	This function takes the imantics Polygon object and returns the Polygon object that only contains the biggest 
	bounding box (which is defined by the biggest range in y-direction).'''
	modified_polygons = []
	for polygon in polygons:
		if len(polygon) == 2: 
			modified_polygons.append(polygon)
		else:
			y_variance = [] # list<tuple<box_index, y-variance>>
			for box in polygon[:-1]:
				y_values = np.array([value[1] for value in box])
				y_variance.append(max(y_values)-min(y_values))
			modified_polygons.append([polygon[y_variance.index(max(y_variance))], polygon[-1]])
	return modified_polygons


def find_seeds(image_array, bounding_box):
	'''This function an array that represents an image and a bounding box. 
	It returns a list of tuples with indices of objects in the image on the bounding box edges.'''

	# Check edges for pixels that are not white
	seed_pixels = []
	for node_index in range(len(bounding_box)):
		# Define amount of steps we have to go in x-direction to get from node 1 to node 2
		x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0]
		if x_diff == 0:
			bounding_box = adapt_x_values(bounding_box = bounding_box, node_index = node_index, image_shape = image_array.shape)
		x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0] # Define amount of steps we have to go in x-direction to get from node 1 to node 2
		x_diff_range = set_x_diff_range(x_diff, euklidian_distance(bounding_box[node_index], bounding_box[node_index-1]), image_array)
		#Go down the edge and check if there is something that is not white. If something was found, the corresponding coordinates are saved.
		for step in x_diff_range:
			x,y = define_next_pixel_to_check(bounding_box, node_index, step, image_shape = image_array.shape)
			# If there is something that is not white	
			if image_array[y, x] < 0.9: 
				seed_pixels.append((x,y))
				#break
	return seed_pixels

def determine_neighbour_pixels(seed_pixel, image_shape):
	'''This function takes a tuple of x and y coordinates and returns a list of tuples of the coordinates of the eight neighbour pixels.'''
	neighbour_pixels = []
	x,y = seed_pixel
	for new_x in range(x-1, x+2):
		if new_x in range(image_shape[0]):
			for new_y in range(y-1, y+2):
				if new_y in range(image_shape[1]):
					if (x,y) != (new_x, new_y):
						neighbour_pixels.append((new_x, new_y))
	return neighbour_pixels

def expand_masks(image_array, seed_pixels, mask_array):
	'''This function takes...
	image_array - Numpy array that represents an image (float)
	mask_array - Numpy array containing binary matrices that define masks (y, x, mask_index)
	seed_pixels - List of tuples with x and y positions of objects on the mask edges
	mask_index - Integer
	and returns a mask array where the mask has been expanded to surround an object in the image completely.'''
	mask_array = copy.deepcopy(mask_array)
	for seed_pixel in seed_pixels:
		neighbour_pixels = determine_neighbour_pixels(seed_pixel, image_array.shape)
		for neighbour_pixel in neighbour_pixels:
			x,y = neighbour_pixel
			if not mask_array[y, x]:
				if image_array[y, x] < 0.90:
					mask_array[y, x] = True
					seed_pixels.append((x,y))
	return mask_array

def expansion_coordination(mask_bounding_box_tuple: Tuple[np.array, np.array], image_array: np.array) -> np.array:
	'''This function takes a tuple containing a bounding box, a single mask and an image (all three: np.array) and coordinates
	the mask expansion. It returns the expanded mask'''
	mask_array, bounding_box = mask_bounding_box_tuple
	seed_pixels = find_seeds(image_array, bounding_box)
	mask_array = expand_masks(image_array, seed_pixels, mask_array)
	return mask_array

def complete_structure_mask(image_array: np.array, mask_array: np.array, debug = False) -> np.array:
	'''This funtion takes an image (array) and an array containing the masks (shape: x,y,n where n is the amount of masks and x and y are the pixel coordinates).
	It detects objects on the contours of the mask and expands it until it frames the complete object in the image. 
	It returns the expanded mask array'''

	if mask_array.size != 0:
		# Turn masks into list of polygon bounding boxes
		polygons = mask_2_polygons(mask_array)

		polygons = define_relevant_polygons(polygons)

		#Binarization of input image
		binarized_image_array = binarize_image(image_array)

		# Apply gaussian filter to the image and binarize again
		blur_factor = int(image_array.shape[1]/500) if image_array.shape[1]/500 >= 1 else 1
		
		blurred_image_array = gaussian_filter(img_as_float(binarized_image_array).copy(), sigma=blur_factor)

		# Create list tuples of masks and corresponding bounding_box
		mask_bounding_box_tuples = [(mask_array[:,:,index], polygons[index][0]) for index in range(len(polygons))]
		
		# Run expansion using Multiprocessing
		image_repeat = itertools.repeat(blurred_image_array, len(mask_bounding_box_tuples)) # Why does this not work?
		#image_repeat = [copy.deepcopy(blurred_image_array) for _ in range(len(mask_bounding_box_tuples))]

		#expanded_split_mask_arrays = map(expansion_coordination, mask_bounding_box_tuples, image_repeat)
		with Pool(20) as p:
			expanded_split_mask_arrays = p.starmap(expansion_coordination, zip(mask_bounding_box_tuples, image_repeat))

		# Stack mask arrays to give the desired output format
		mask_array = np.stack(expanded_split_mask_arrays, -1)

		#for polygon_index in range(len(polygons)): 

		#	bounding_box = polygons[polygon_index][0]

			# Find seed pixels for expansion
		#	seed_pixels = find_seeds(blurred_image_array, bounding_box)
		#	mask_array = expand_masks(blurred_image_array, seed_pixels, mask_array, mask_index = polygon_index)

	else: 
		print("No masks found.")
	return mask_array