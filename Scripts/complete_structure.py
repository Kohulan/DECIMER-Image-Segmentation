import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage.filters import gaussian_filter

from skimage.transform import downscale_local_mean, rescale
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, square
from skimage import img_as_float

from imantics import Mask, Polygons


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

def define_local_center(bounding_box, node_index, n=20):
	'''This function return the center np.array([x,y]) of the n previous and the n following nodes 
	of bounding box np.array([[x1, y1], [x2, y2],...,[x_n, y_n]])'''
	x_values = np.array([])
	y_values = np.array([])
	for node in np.roll(bounding_box, -node_index+n)[:2*n]:
		x_values = np.append(x_values, bounding_box[node_index][0])
		y_values = np.append(y_values, bounding_box[node_index][1])
	return np.array([np.mean(x_values), np.mean(y_values)])

def define_edge_line(node_1, node_2):
	'''This function returns a linear function between two given points in a 2-dimensional space'''
	if node_1[0]-node_2[0] != 0:
		slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0])
	else:
		slope = (node_2[1]-node_1[1])/(node_2[0]-node_1[0]+0.00000000000001) # avoid dividing by zero, this line should not be necesary anymore.
	#intercept = y - m*x
	intercept = node_1[1] - slope * node_1[0]
	return slope, intercept

def euklidian_distance(node_1, node_2):
	'''This function returns the euklidian distance between two given points in a 2-dimensional space.'''
	return np.sqrt((node_2[0]-node_1[0]) ** 2 + (node_2[1]-node_1[1])**2)

def define_stepsize(slope, shape, step_factor):
	'''This function takes the slope of the line along which the node is pushed out of the bounding box center
	and the shape of the image. Depending on the resolution and the slope, the step size in x-direction is defined.
    The step_factor can be varied to alter the step_size (The bigger it is, the smaller are the steps.)'''
	#diagonal_slope=shape[0]/shape[1]
	diagonal_slope = 1
	if slope < diagonal_slope and slope > -diagonal_slope:
		return shape[1]/step_factor
	else:
		return (shape[1]/step_factor)/slope
    

def expand_bounding_box(bounding_box, nodes_to_be_changed, step_factor, shape, iterations, local_center_ratio = False):
	'''This function takes a list of nodes to be changed and modifies them by moving them away from the 
	bounding box center or a local center along a line between the node and the center point.'''

	if not local_center_ratio or iterations % local_center_ratio != 0:
		center = define_bounding_box_center(bounding_box) # Define bounding box center

	for node_index in nodes_to_be_changed:
		if local_center_ratio and iterations % local_center_ratio == 0:
			center = define_local_center(bounding_box, node_index, n = 25) # Define local center
            
		#Define the axis along which we want to move the node
		slope, intercept = define_edge_line(center, bounding_box[node_index])
		step_size = define_stepsize(slope, shape, step_factor)
		changed_node_1 = [bounding_box[node_index][0]+step_size, slope * (bounding_box[node_index][0]+step_size) + intercept]
		changed_node_2 = [bounding_box[node_index][0]-step_size, slope * (bounding_box[node_index][0]-step_size) + intercept]
		if euklidian_distance(changed_node_1, center) >= euklidian_distance(changed_node_2, center):
			bounding_box[node_index] = changed_node_1
		else:
			bounding_box[node_index] = changed_node_2
	return bounding_box


def set_x_diff_range(x_diff, euklidian_distance):
	'''This function takes the amount of steps on an edge and returns the corresponding list of steps in random order'''
	blur_factor = int(image.shape[1]/400) if image.shape[1]/400 >= 1 else 1
	if x_diff > 0:
		x_diff_range = np.arange(0, x_diff, blur_factor/euklidian_distance) 
	else:
		x_diff_range = np.arange(0, x_diff, -blur_factor/euklidian_distance)
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

def reduce_polygon_nodes(polygons, number):
	'''This function takes a list of imantics Polygon objects and reduces the amount of nodes to a given number.
	The adapted list is then returned. If the given number is smaller than the amount of nodes, the original polygon
	is kept.'''
	adapted_polygons = polygons.copy()
	for polygon_index in range(len(polygons)):
		factor = int(len(polygons[polygon_index][0])/number)
		if factor >= 1:
			adapted_polygons[polygon_index][0] = polygons[polygon_index][0][1::factor]
	return adapted_polygons

def factor_fold_nodes(bounding_box, factor):	
	'''A bounding box which is defined by n points is turned into a bounding box where each edge between two nodes has 8 more equidistant nodes.'''	
	new_bounding_box = np.zeros((bounding_box.shape[0] * factor, bounding_box.shape[1]))	
	for node_index in range(len(bounding_box)):	
        #These if/else blocks avoid steps of zero in the arange.
		if bounding_box[node_index][0]-bounding_box[node_index-1][0]:
			x_range = np.arange(bounding_box[node_index-1][0], bounding_box[node_index][0], (bounding_box[node_index][0]-bounding_box[node_index-1][0])/factor)	
		else:
			x_range = np.full((1, factor), bounding_box[node_index][0]).flatten()
		if (bounding_box[node_index][1]-bounding_box[node_index-1][1]) != 0:
			y_range = np.arange(bounding_box[node_index-1][1], bounding_box[node_index][1], (bounding_box[node_index][1]-bounding_box[node_index-1][1])/factor)	
		else:
			y_range = np.full((1, factor), bounding_box[node_index][1]).flatten()
		for index in range(len(x_range)):	
			new_bounding_box[node_index*factor+index] = [x_range[index], y_range[index]]	
	return new_bounding_box

def bb_expansion_algorithm(image_array, original_bounding_box, debug = False, parameter_combinations = [(200, 50, False)]):
	'''This function applies the bounding box algorithm to a given bounding box with a given image and returns the expanded bounding box.'''
	for parameter_combination in parameter_combinations:
		bounding_box = original_bounding_box.copy()
		step_factor, step_limit, local_center_ratio = parameter_combination
		nothing_on_the_edges = False
		iterations = 0	#count iteration steps
		# Check edges for pixels that are not white and expand the bounding box until there is nothing on the edges.
		while nothing_on_the_edges == False:
			nodes_to_be_changed = []
			for node_index in range(len(bounding_box)):
				# Define amount of steps we have to go in x-direction to get from node 1 to node 2
				x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0]
				if x_diff == 0:
					bounding_box = adapt_x_values(bounding_box = bounding_box, node_index = node_index, image_shape = image_array.shape)
				x_diff = bounding_box[node_index][0] - bounding_box[node_index-1][0] # Define amount of steps we have to go in x-direction to get from node 1 to node 2
				x_diff_range = set_x_diff_range(x_diff, euklidian_distance(bounding_box[node_index], bounding_box[node_index-1]))
				#Go down the edge and check if there is something that is not white. If something was found, the corresponding nodes are saved.
				for step in x_diff_range:
					x,y = define_next_pixel_to_check(bounding_box, node_index, step, image_shape = image_array.shape)
					# If there is something that is not white	
					if sum(image_array[y, x]) < 2.9: 
						nodes_to_be_changed.append(node_index)
						nodes_to_be_changed.append(node_index-1)
						break
			if iterations >= step_limit:
				break
			if nodes_to_be_changed == []:
				print('Bounding box expansion complete. \n Step factor: ' + str(step_factor) + '\n Local center ratio: ' + str(local_center_ratio))
				nothing_on_the_edges = True
			else: 
				iterations += 1
				if debug:
					print("The bounding box is modified. Iteration No. " + str(iterations))
					plot_it(image_array = image_array, bounding_box = bounding_box)
				nodes_to_be_changed = set(nodes_to_be_changed)
				bounding_box = expand_bounding_box(bounding_box, nodes_to_be_changed, step_factor, shape = image_array.shape, iterations = iterations, local_center_ratio = local_center_ratio)
		if iterations < step_limit:
			return bounding_box
	print("Bounding box expansion was not successful. Return original bounding box.")
	return original_bounding_box


def binarize_image(image_array):
	'''This function takes a Numpy array that represents an RGB image and returns the binarized form of that image
	by applying the otsu threshold.'''
	grayscale = rgb2gray(image_array)
	threshold = threshold_otsu(image)
	binarized_image_array = image > threshold
	return binarized_image_array

def new_polygon_2_mask(new_polygon):
	new_polygon = Polygons(Polygons(new_polygon).segmentation) # I know it looks ridiculous, but otherwise, it does not work.
	width, height = new_polygon.bbox().max_point 
	width += 1
	height += 1
	return new_polygon.mask(height = height, width = width)[:,:,None]

def complete_structure_mask(image_array, mask_array, debug = False):
	'''This funtion takes an image (array) and an array containing the masks (shape: x,y,n where n is the amount of masks and x and y are the pixel coordinates).
	It turns it into a polygon bounding box and expands the bounding box so that it frames the complete object in the image. Then, the polygon bounding box is
	transformed back into the mask array format again (which is then returned).'''

	if mask_array.size != 0:
		# Turn masks into list of polygon bounding boxes
		polygons = mask_2_polygons(mask_array)    
    
		# Reduce number of nodes of polygon
		polygons = reduce_polygon_nodes(polygons, number = 10)
    
		#plot it (in debug-mode)
		if debug:
			print('Original image with bounding boxes')
			plot_it_multiple(image_array = image_array, bounding_boxes = polygons)    

		#Binarization of input image
		binarized_image_array = binarize_image(image_array)
    
		#plot it (in debug-mode)
		if debug:
			print(	'Input image after binarization:')
			plot_it_multiple(image_array = img_as_float(binarized_image_array), bounding_boxes = polygons) 

		# Apply gaussian filter to the image
		blur_factor = int(image.shape[1]/400) if image.shape[1]/400 >= 1 else 1
		blurred_image_array = gaussian_filter(img_as_float(binarized_image_array).copy(), sigma=blur_factor)
		print("BLUR")
		print(blur_factor)
		#blurred_image_array = binary_erosion(binarized_image_array) #selem=square(width = 100))
    
		#plot it (in debug-mode)
		if debug:
			print('Image after the application of a gaussian blurr')
			plot_it_multiple(image_array = blurred_image_array, bounding_boxes = polygons)
    
		#parameter_combinations: list<tuple<step_factor, step_limit, local_center_rat	io
		parameter_combinations = [(100, 25, False), (100, 25, 8), (100, 25, 4), (200, 50, False), (200, 50, 8), (200, 50, 4), (500, 120, False), (500, 120, 8), (500, 120, 4)]    
		#Apply bounding box expansion algorithm to the polygons
		expanded_masks = []
		for polygon in polygons:
			new_polygon = polygon.copy()
			bounding_box = factor_fold_nodes(polygon[0], factor = 10)
			new_polygon[0] = bb_expansion_algorithm(image_array = blurred_image_array, original_bounding_box = bounding_box, debug = debug, parameter_combinations = parameter_combinations)
			#Recreate the mask from polygon bounding box
			new_polygon = new_polygon_2_mask(new_polygon)
			expanded_masks.append(new_polygon)

		#Merge the masks into one array and invert it again.
		expanded_masks = np.invert(np.dstack(expanded_masks[:]))
	else: 
		expanded_masks = mask_array
	return expanded_masks

def complete_structure(image, bounding_box):
	'''This funtion takes an image and a bounding box and returns the expanded bounding box'''
	
	# Open image as Numpy array
	image_array = np.asarray(Image.open(image))

	# Plot it
	plot_it(image_array = image_array, bounding_box = bounding_box)

	# Tenfold nodes
	bounding_box = tenfold_nodes(bounding_box)

	# Apply Gaussian Filter
	blurred_image_array = gaussian_filter(image_array.copy(), sigma=2)

	# Plot it
	plot_it(image_array = blurred_image_array, bounding_box = bounding_box)

	#Apply bounding box expansion algorithm
	bounding_box = bb_expansion_algorithm(blurred_image_array, bounding_box)

	# Plot it
	plot_it(image_array = image_array, bounding_box = bounding_box)

	return bounding_box