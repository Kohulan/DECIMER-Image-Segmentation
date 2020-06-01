# Copyright Kohulan Rajan,2020
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import cv2

desired_size = 299
im_pth = "test_scanned-2.jpg"

im = Image.open(im_pth)

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

blank_image.show()
blank_image.save('someimage.png')

image_x = np.array(blank_image)
image_enhanced = cv2.equalizeHist(image_x)

plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
plt.show()