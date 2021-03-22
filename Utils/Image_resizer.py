'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2

from Utils.get_skeleton import get_skeleton_and_thin

def main():
	
	im_pth = "segment.png"

	get_clean_segments(im_pth)

	print("Process Completed")


	
def get_clean_segments(IMAGE_PATH,desired_size=299):
	
	bnw_img = get_bnw_image(IMAGE_PATH)
	image = Image.open(bnw_img)

	#Resize image
	resize_image(IMAGE_PATH,image,desired_size,get_skeleton=False)

def get_bnw_image(im_pth):
	original_img = cv2.imread(im_pth)
	grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	#_,thresholded = cv2.threshold(grayscale,0,255,cv2.THRESH_OTSU)
	(thresh, im_bw) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	cv2.imwrite(im_pth[:-4]+"_bnw.png", im_bw)
	#image_new.save()
	return im_pth[:-4]+"_bnw.png"

def resize_image(IMAGE_PATH,im,desired_size,get_skeleton=False):
	old_size = im.size  # old_size[0] is in (width, height) format
	greyscale_image = im.convert('L')
	#print(len(greyscale_image.split()))

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

	blank_image.save(IMAGE_PATH[:-4]+'_clean.png')

	if get_skeleton:
		out_skeletonize,out_thin = get_skeleton_and_thin(IMAGE_PATH[:-4]+'_skeleton.png')

	#blank_image.show()

	#image_x = np.array(blank_image)
	#image_enhanced = cv2.equalizeHist(image_x)

	#plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
	#plt.show()

if __name__ == '__main__':
	main()
