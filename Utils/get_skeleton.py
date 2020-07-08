'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
from skimage import img_as_float
from skimage import io, color, morphology
import matplotlib.pyplot as plt

def get_skeleton_and_thin(input_image):
	image = img_as_float(color.rgb2gray(io.imread(input_image)))
	#image = img_as_float(color.rgb2gray(input_image))
	image_binary = image < 0.5
	out_skeletonize = morphology.skeletonize(image_binary)
	out_thin = morphology.thin(image_binary)


	f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))

	plt.imsave('thinned_output.png', 255-out_thin,cmap='gray')
	plt.imsave('skeletonized_output.png', 255-out_skeletonize,cmap='gray')

	return out_skeletonize,out_thin
