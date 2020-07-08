'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import cv2
from get_skeleton import get_skeleton_and_thin

def main():
	
	im_pth = "segment_1.png"

	im = get_bnw_image(im_pth)
	image = Image.open(im)

	#Size parameter
	desired_size = 299

	#Resize image
	resize_image(image,desired_size,get_skeleton=False)

def get_bnw_image(im_pth):
	original_img = cv2.imread(im_pth)
	grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	#_,thresholded = cv2.threshold(grayscale,0,255,cv2.THRESH_OTSU)
	(thresh, im_bw) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	cv2.imwrite("black_and_white.png", im_bw)
	#image_new.save()
	return "black_and_white.png"

def resize_image(im,desired_size,get_skeleton=False):
	old_size = im.size  # old_size[0] is in (width, height) format
	greyscale_image = im.convert('L')
	print(len(greyscale_image.split()))

	enhancer = ImageEnhance.Contrast(greyscale_image)

	greyscale_image = enhancer.enhance(1.0)

	if(old_size[0] or old_size[1] != desired_size):
		ratio = float(desired_size)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
		greyscale_image = greyscale_image.resize(new_size, Image.ANTIALIAS)
	else:
		new_size = old_size
	blank_image = Image.new('L', (299, 299), 'white')

	blank_image.paste(greyscale_image, ((desired_size-new_size[0])//2,
	                    (desired_size-new_size[1])//2))

	blank_image.save('output_image.png')

	if get_skeleton:
		out_skeletonize,out_thin = get_skeleton_and_thin('output_image.png')

	#blank_image.show()

	#image_x = np.array(blank_image)
	#image_enhanced = cv2.equalizeHist(image_x)

	#plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
	#plt.show()

if __name__ == '__main__':
	main()