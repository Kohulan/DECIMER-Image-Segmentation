from django.db import models

from django.core.files.storage import default_storage


from django.core.files.images import ImageFile

from django.conf import settings

from uuid import uuid4


import os

import sys
import glob

import warnings
warnings.filterwarnings("ignore")

from .DECIMER_Image_Segmentation.Utils import pdf_2_img_Convert
from .DECIMER_Image_Segmentation.Utils import Image_resizer
from .DECIMER_Image_Segmentation import Detect_and_save_segmentation

#from .models import SegmentedImage, UploadedArticle


def process_article(articleObj):
    

    # segmentation
    output_directory = pdf_2_img_Convert.convert(articleObj.article.path)
    print(output_directory)
    converted_images = glob.glob(output_directory+"/*.png") # Get all images converted into JPEGs
    print(converted_images)

    if output_directory.endswith('/'):
        output_directory = output_directory[:-1]

    if os.path.exists(output_directory+"/segments"):
        os.system("rm -rf "+output_directory+"/segments")
        os.system("mkdir "+output_directory+"/segments")
    else:
        os.system("mkdir "+output_directory+"/segments")



    for i in converted_images:
        directory,info = Detect_and_save_segmentation.get_segments(i, output_directory)

    path_to_segments = glob.glob(directory+"*.png")


    articleObj.all_segmented_images_names = "$x$x$x$".join(path_to_segments)
    print(path_to_segments)
    print("Cleaning up images")

    for im_pth in path_to_segments:
        Image_resizer.get_clean_segments(im_pth)

    articleObj.path_to_segmented_images = path_to_segments
    print("*******")
    print('segmentation success')
    print("saving images")

    #super().save(*args, **kwargs)

    
    # create image objects
    if len( path_to_segments)>0:
        
        
        t = path_to_segments[0].split("/")[0:-1]
        images_directory = "/".join(t)
        print("images directory: "+ str(images_directory))
        all_images = glob.glob(images_directory+"/*.png")

        print(all_images)

        base_names_list = set()
        list_of_image_triplets = []

        for img in all_images:
            print(img)

            n = img[:-4]
            n = n.replace("_bnw", "")
            n = n.replace("_clean", "")
            base_names_list.add(n)


        for base_name in base_names_list:

            triplet = [x for x in all_images if base_name in x]
            list_of_image_triplets.append(triplet)
        

        for tri_image in list_of_image_triplets:

            image_clean = ''
            image_bnw = ''
            image_ori = ''

            clean_name = ""
            bnw_name = ""
            ori_name = ""


            for cl_img in tri_image:

                if "bnw" in cl_img:
                    fi = open(cl_img, 'rb')
                    image_bnw = ImageFile(fi)
                    bnw_name = cl_img.split("/")[-1]
                    #fi.close()
                elif "clean" in cl_img:
                    # do clean
                    fii = open(cl_img, 'rb')
                    image_clean = ImageFile(fii)
                    clean_name = cl_img.split("/")[-1]
                    #fi.close()
                else:
                    #do original
                    fiii = open(cl_img, 'rb')
                    image_ori = ImageFile(fiii)
                    ori_name = cl_img.split("/")[-1]
                    #fi.close()

            # with all the images, create the object

            nsi = SegmentedImage()
            nsi.ori_article_id = articleObj.id
            nsi.ori_image.save(ori_name, image_ori, save=True)
            nsi.bnw_image.save(bnw_name, image_bnw, save=True)
            nsi.clean_image.save(clean_name, image_clean, save=True)
            #nsi = SegmentedImage(ori_image = image_ori, bnw_image = image_bnw, clean_image = image_clean , ori_article_id = self.id)
            
            nsi.save()
            print("nsi created")
        articleObj.processingState = 2
        articleObj.save()

                    
