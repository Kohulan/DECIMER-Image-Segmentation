# DECIMER-Image-Segmentation
Image segmentation initial works by Kohulan Rajan and Otto Brinkhaus

Using Mask R-CNN, we are trying to recognize and segment depictions of chemical structures from the published literature. The superiour algorith of the mask R-CNN model can easily segment an Image after being trained with a mixed set of data, including a manually annotated set of pages from the published literature.

Initial work was done training the Mask R-CNN network with our data and deploying it on a completely new set of data to examine the segmentation accuracy. There is a post-processing procedure Additionally, we are working on a post-processing procedure for the segmented images to make them suitable for the image detection algorithm. 

## Usage

-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU enabled PC, so please do make sure you have all the necessary drivers installed.

```
$ git clone ..
$ cd DECIMER-Image-Segmentation
```
- Create a python virual environment. (We recommend conda environment)
```
$conda activate
$conda install pip
$pip install tensorflow-gpu==1.15 pillow opencv-python matplotlib scikit-image keras==2.3.1 imantics IPython #Install tensorflow==1.15 if you do not have a nVidia GPU
$ python3 Detect_and_save_segmentation.py --input path/to/input/Image
```
- Segmented images will be saved inside the output folder

## Authors 
- [Kohulan](github.com/Kohulan)
- [Otto Brinkhaus](github.com/OBrink)

## Project page

![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/DECIMER_logo.png?raw=true)

- [DECIMER](https://kohulan.github.io/Decimer-Official-Site/)

## More information about our research group
- [Website](https://cheminf.uni-jena.de)

![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)
