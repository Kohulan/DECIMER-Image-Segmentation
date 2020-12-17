import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools

from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion
from imantics import Mask, Polygons
from typing import Dict, List, Tuple

def plot_it(image_array: np.array) -> None:
	'''This function shows the plot of a given image (np.array([...])) with a given bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	#bounding_box = Polygon(bounding_box, linewidth=1,edgecolor='r',facecolor='none')
	plt.rcParams["figure.figsize"] = (20,15)

	fig,ax = plt.subplots(1)
	imgplot = ax.imshow(image_array)
	#ax.add_patch(bounding_box)
	#plt.savefig("./image_"+str(image_counter)+".png")
	plt.show()
	return

def define_bounding_box_center(bounding_box: np.array) -> np.array:
	'''This function return the center np.array([x,y]) of a given bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	x_values = np.array([])
	y_values = np.array([])
	for node in bounding_box:
		x_values = np.append(x_values, node[0])
		y_values = np.append(y_values, node[1])
	return np.array([np.mean(x_values), np.mean(y_values)])

def define_edge_line(node_1: Tuple[int, int], node_2: Tuple[int, int]) -> Tuple[float, float]:
	'''This function returns slope and intercept of a linear function between two given points in a 2-dimensional space'''
	slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0])
	intercept = node_1[1] - slope * node_1[0]
	return slope, intercept

def euklidian_distance(node_1: Tuple[int, int], node_2: Tuple[int, int]) -> float:
	'''This function returns the euklidian distance between two given points in a 2-dimensional space.'''
	return np.sqrt((node_2[0]-node_1[0]) ** 2 + (node_2[1]-node_1[1])**2)
    
def set_x_diff_range(x_diff: float, euklidian_distance: float, image_array: np.array) -> np.array:
	'''This function takes the amount of steps on an edge and returns the corresponding list of steps in random order'''
	blur_factor = int(image_array.shape[1]/500) if image_array.shape[1]/500 >= 1 else 1
	if x_diff > 0:
		x_diff_range = np.arange(0, x_diff, blur_factor/euklidian_distance)		
	else:
		x_diff_range = np.arange(0, x_diff, -blur_factor/euklidian_distance)
	np.random.shuffle(x_diff_range)
	return x_diff_range

def define_next_pixel_to_check(bounding_box: np.array, node_index: int, step: int, image_shape: Tuple[int, int, int]) -> Tuple[int, int]:
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

def adapt_x_values(bounding_box: np.array, node_index: int, image_shape: Tuple[int, int, int]) -> np.array:
	'''If two nodes form a vertical edge the function that descripes that edge will in- or decrease infinitely with dx so we need to alter those nodes. 
	This function returns a bounding box where the nodes are altered depending on their relative position to the center of the bounding box.
	If a node is at the border of the image, then it is not changed.'''
	bounding_box = copy.deepcopy(bounding_box)
	if bounding_box[node_index][0] != image_shape[1]:
		bounding_box_center = define_bounding_box_center(bounding_box)
		if bounding_box[node_index][0] < bounding_box_center[0]:
			bounding_box[node_index][0] = bounding_box[node_index][0] - 1
		else: 
			bounding_box[node_index][0] = bounding_box[node_index][0] + 1
	return bounding_box

def mask_2_polygons(mask_array: np.array) -> List:
	'''This function takes a mask array and returns a list of polygon bounding boxes (node coordinates)'''
	#imantics is built on openCV. The mask needs to be inverted so that the findcontours function works. Unintuitive but it works this way.
	inverted_mask = np.invert(mask_array).astype(int)
	return Polygons.from_mask(mask_array).points

def binarize_image(image_array: np.array, threshold = "otsu") -> np.array:
	'''This function takes a Numpy array that represents an RGB image and returns the binarized form of that image
	by applying the otsu threshold.'''
	grayscale = rgb2gray(image_array)
	if threshold == "otsu":
		threshold = threshold_otsu(grayscale)
	binarized_image_array = grayscale > threshold
	return binarized_image_array

def define_relevant_polygons(polygon: np.array) -> np.array:
	'''Sometimes, the mask R CNN model produces a weird output with one big masks and some small irrelevant islands.
	This function takes the imantics Polygon object and returns the Polygon object that only contains the biggest 
	bounding box (which is defined by the biggest range in y-direction).'''
	if len(polygon) == 1:
		return polygon
	else:
		y_variance = [] # list<tuple<box_index, y-variance>>
		for box in polygon:
			y_values = np.array([value[1] for value in box])
			y_variance.append(max(y_values)-min(y_values))
		return [polygon[y_variance.index(max(y_variance))]]

def find_seeds_contours(image_array: np.array, bounding_box: np.array) -> List[Tuple[int, int]]:
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
	return seed_pixels

def find_mask_center(mask_array: np.array)->Tuple[int,int]:
	'''This function takes a binary matrix (numpy array containing a mask) and defines the center of the mask.
	It returns a tuple with the center indices.'''
	# First, try to find global mask center. If that point is included in the mask, return it.
	y_coordinates, x_coordinates = np.nonzero(mask_array)
	x_center = int((x_coordinates.max()+x_coordinates.min())/2)
	y_center = int((y_coordinates.max()+y_coordinates.min())/2)
	x_dist = int(x_coordinates.max()-x_coordinates.min())-1
	y_dist = int(y_coordinates.max()-y_coordinates.min())-1
	if mask_array[y_center, x_center]:
		return (x_center, y_center)
	else:
		# If the global mask center is not placed in the mask, take the center on the x-axis and the first-best y-coordinate that lies in the mask
		x_center = np.where(mask_array[y_center]==True)[0][0]
		return (x_center, y_center)

def find_seeds(image_array: np.array, mask_array: np.array) -> List[Tuple[int, int]]:
	'''This function takes an array that represents an image and a mask. 
	It returns a list of tuples with indices of seeds in the structure covered by the mask.'''
	x_center, y_center = find_mask_center(mask_array)
	# Starting at the mask center, check for pixels that are not white
	seed_pixels = []
	up, down, right, left = True, True, True, True
	for n in range(1, 1000):
		# Check for seeds above center
		if up:
			if x_center+n < image_array.shape[1]:
				if not mask_array[y_center, x_center+n]:
					up = False
				if not image_array[y_center, x_center+n]:
					seed_pixels.append((x_center+n, y_center))
					up = False
		# Check for seeds below center
		if down:
			if x_center-n >= 0:
				if not mask_array[y_center, x_center-n]:
					down = False
				if not image_array[y_center, x_center-n]:
					seed_pixels.append((x_center-n, y_center))
					down = False
		# Check for seeds left from center		
		if left:
			if y_center+n < image_array.shape[0]:
				if not mask_array[y_center+n, x_center]:
					left = False
				if not image_array[y_center+n, x_center]:
					seed_pixels.append((x_center, y_center+n))
					left = False
		# Check for seeds right from center
		if right:
			if y_center-n >= 0:
				if not mask_array[y_center-n, x_center]: 
					right = False
				if not image_array[y_center-n, x_center]:
					seed_pixels.append((x_center, y_center-n))
					right = False
	return seed_pixels

def determine_neighbour_pixels(seed_pixel: Tuple[int, int], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
	'''This function takes a tuple of x and y coordinates and returns a list of tuples of the coordinates of the eight neighbour pixels.'''
	neighbour_pixels = []
	x,y = seed_pixel
	for new_x in range(x-1, x+2):
		if new_x in range(image_shape[1]):
			for new_y in range(y-1, y+2):
				if new_y in range(image_shape[0]):
					if (x,y) != (new_x, new_y):
						neighbour_pixels.append((new_x, new_y))
	return neighbour_pixels

def expand_masks(image_array: np.array, seed_pixels: List[Tuple[int, int]], mask_array: np.array, contour_expansion = False) -> np.array:
	'''This function takes...
	image_array - Numpy array that represents an image (float)
	mask_array - Numpy array containing binary matrices that define masks (y, x, mask_index)
	seed_pixels - List of tuples with x and y positions of objects on the mask edges
	mask_index - Integer
	and returns a mask array where the mask has been expanded to surround an object in the image completely.'''
	
	# If the mask is supposed to be reconstructed and not expanded from the contours
	if not contour_expansion:
		mask_array = np.zeros(mask_array.shape)
	for seed_pixel in seed_pixels:
		neighbour_pixels = determine_neighbour_pixels(seed_pixel, image_array.shape)
		for neighbour_pixel in neighbour_pixels:
			x,y = neighbour_pixel
			if not mask_array[y, x]:
				if not image_array[y, x]:
					mask_array[y, x] = True
					seed_pixels.append((x,y))
	return mask_array

def expansion_coordination(mask_array: np.array, image_array: np.array) -> np.array:
	'''This function takes a tuple containing a bounding box, a single mask and an image (all three: np.array) and coordinates
	the mask expansion. It returns the expanded mask. 
	The purpose of this function is wrapping up the expansion procedure in a map function.'''
	seed_pixels = find_seeds(image_array, mask_array)
	if seed_pixels != []:
		mask_array = expand_masks(image_array, seed_pixels, mask_array)
	else:
		# If the seed detection inside of the mask has failed for some reason, look for seeds on the contours of the mask and expand from there on.
		# Turn masks into list of polygon bounding boxes
		polygon = mask_2_polygons(mask_array)
		# Delete unnecessary mask blobs
		polygon = define_relevant_polygons(polygon)
		seed_pixels = find_seeds_contours(image_array = image_array, bounding_box = polygon[0])
		mask_array = expand_masks(image_array, seed_pixels, mask_array, contour_expansion = True)
	return mask_array

def complete_structure_mask(image_array: np.array, mask_array: np.array, debug = False) -> np.array:
	'''This funtion takes an image (array) and an array containing the masks (shape: x,y,n where n is the amount of masks and x and y are the pixel coordinates).
	It detects objects on the contours of the mask and expands it until it frames the complete object in the image. 
	It returns the expanded mask array'''
	
	if mask_array.size != 0:
		#Binarization of input image
		binarized_image_array = binarize_image(image_array, threshold = 0.85)

		# Apply gaussian filter with a resolution-dependent standard deviation to the image 
		blur_factor = int(image_array.shape[1]/185) if image_array.shape[1]/185 >= 2 else 2
		
		if debug:
			plot_it(binarized_image_array)
		
		#Define kernel and apply
		kernel = np.ones((blur_factor, blur_factor))
		blurred_image_array = binary_erosion(binarized_image_array, selem = kernel)
		
		if debug:
			plot_it(blurred_image_array)
		# Slice mask array along third dimension into single masks
		split_mask_arrays = np.array([mask_array[:,:,index] for index in range(mask_array.shape[2])])		
		
		# Run expansion the expansion
		image_repeat = itertools.repeat(blurred_image_array, mask_array.shape[2])
		# Faster with map function 
		expanded_split_mask_arrays = map(expansion_coordination, split_mask_arrays, image_repeat)
		
		# Stack mask arrays to give the desired output format
		mask_array = np.stack(expanded_split_mask_arrays, -1)
		return mask_array

	else: 
		print("No masks found.")
		return mask_array