from django.db import models

from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from django.core.files.images import ImageFile
from django.core.files.base import ContentFile
from django.utils.deconstruct import deconstructible
from django.conf import settings
from uuid import uuid4
import os
import sys
import glob
import shutil
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import multiprocessing as mp
from functools import partial

from .DECIMER_Image_Segmentation.Utils import pdf_2_img_Convert
from .DECIMER_Image_Segmentation.Utils import Image_resizer
from .DECIMER_Image_Segmentation import Detect_and_save_segmentation





class OverwriteStorage(FileSystemStorage):

    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def process_article(art_id):
    
    articleObj = UploadedArticle.objects.get(id=art_id)
    articleObj.processingState=1
    articleObj.processingStep=1 # uploaded
    articleObj.save(update_fields=['processingState','processingStep'])

    print("saved article")
    # segmentation
    output_directory = pdf_2_img_Convert.convert(articleObj.article.path)
    print("converted PDF")
    articleObj.processingStep=2 # PDF separated in pages and converted to images
    articleObj.save(update_fields=['processingStep'])


    converted_images = glob.glob(output_directory+"/*.png") # Get all images converted into JPEGs
    print("got converted articles")
    

    if output_directory.endswith('/'):
        output_directory = output_directory[:-1]

    if os.path.exists(output_directory+"/segments"):
        os.system("rm -rf "+output_directory+"/segments")
        os.system("mkdir "+output_directory+"/segments")
    else:
        os.system("mkdir "+output_directory+"/segments")

    articleObj.processingStep=3 # starting mask expansion for each PDF page
    articleObj.save(update_fields=['processingStep'])

    pool = mp.Pool(4)
    Pool_function = partial(Detect_and_save_segmentation.get_segments,output_directory)
    print("finished pool function")
    expanded_masks = pool.map(Pool_function,converted_images)
    print("getting expanded masks:")
    print(expanded_masks)
    #for i in converted_images:
    #   directory,info = Detect_and_save_segmentation.get_segments(i, output_directory)

    path_to_segments = glob.glob(expanded_masks[0]+"*.png")
    print("path to segments:")
    print(path_to_segments)


    print("Cleaning up images")
    articleObj.processingStep=4 # image cleaning
    articleObj.save(update_fields=['processingStep'])

    #for im_pth in path_to_segments:
    #    Image_resizer.get_clean_segments(im_pth)

    pool.map(Image_resizer.get_clean_segments,path_to_segments)	
    pool.close()
    pool.join()


    #articleObj.path_to_segmented_images = path_to_segments
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
            nsi.ori_article_name = articleObj.ori_name
            nsi.ori_image.save(ori_name, image_ori, save=True)
            nsi.bnw_image.save(bnw_name, image_bnw, save=True)
            nsi.clean_image.save(clean_name, image_clean, save=True)
            
            nsi.save()
            print("nsi created")

        return(2)
    else:

        return(-1)




@deconstructible
class UploadToPathAndRename(object):

    def __init__(self, path):
        self.sub_path = path

    def __call__(self, instance, filename):
        ext = filename.split('.')[-1]
        # get filename
        if instance.pk:
            filename = '{}.{}'.format(instance.pk, ext)
        else:
            # set filename as random string
            filename = '{}.{}'.format(uuid4().hex, ext)
        # return the whole path to the file
        return os.path.join(self.sub_path, filename)


class SegmentedImage(models.Model):
    ori_image = models.ImageField(upload_to='images', max_length=500, blank=True)
    bnw_image = models.ImageField(upload_to='images', max_length=500, blank=True)
    clean_image = models.ImageField(upload_to='images', max_length=500, blank=True)

    #name = models.CharField(max_length=500, blank=True)
    smiles = models.CharField(max_length=500, blank=True)

    ori_article_name = models.CharField(max_length=500, blank=True)
    ori_article_id = models.IntegerField()

    def __str__(self):
        return "Image generated from article id "+self.ori_article_id

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)


class UploadedArticle(models.Model):
    article = models.FileField(upload_to='articles', storage=OverwriteStorage())
    ori_name = models.CharField(max_length=600, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)
    path_to_segmented_images = models.CharField(max_length=600, blank=True)
    processingState = models.IntegerField(blank=False, default=0)
    processingStep = models.IntegerField(blank=False, default=0)
    all_segmented_images_names = models.TextField(blank=True)
    zippedExtractedImages = models.FileField(upload_to='archives',storage=OverwriteStorage(), blank=True )

    def __str__(self):
        return "Article uploaded at {}".format(self.uploaded.strftime('%Y-%m-%d %H:%M'))


    def save(self, *args, **kwargs):

        

        try:
            self.ori_name = self.article.name
            super().save(*args, **kwargs)
            if(self.processingState==0):
                print("saving the article")

                print("just before processing")

                print("******************")
                self.processingState= process_article(self.id)
                print("******************")
                print("just after processing")

                print("******************")

                self.save(update_fields=['processingState'])

                print("******************")
                print("saving the processing state")



                print("done segmentation")
            else:

                print("entered in the elif loop!")
                print(self.processingState)
                if self.processingState == 2 or self.processingState== -1:
                    print("should do nothing and return")

        except Exception as e:
            print('segmentation failed')
            print(sys.exc_info())
            print(e)

    #def doSegmentation(article):

        

        
