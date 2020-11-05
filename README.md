# DECIMER-Image-Segmentation

Image segmentation initial works by Kohulan Rajan and Otto Brinkhaus

Using Mask R-CNN, we are trying to recognize and segment depictions of chemical structures from the published literature. The superiour algorith of the mask R-CNN model can easily segment an Image after being trained with a mixed set of data, including a manually annotated set of pages from the published literature.

Initial work was done training the Mask R-CNN network with our data and deploying it on a completely new set of data to examine the segmentation accuracy. There is a post-processing procedure additionally where we incorporated a mask expansion algorithm to segement complete chemical structures. later on the complete segmented structures will under go a cleaning up process and the results will be given back to the user. 

## Usage

-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU enabled PC or simply on CPU, so please do make sure you have all the necessary drivers installed if you are using GPU.
- Install Git LFS.

```
sudo apt-get install -y git-lfs
```
- Now enter the following commands.
```
$ git clone ..
$ cd DECIMER-Image-Segmentation
```
- Download the [trained model](https://storage.googleapis.com/mrcnn-weights/mask_rcnn_molecule.h5)
- Copy the model to DECIMER-Image-Segmentation/model_trained/
- Create a python virtual environment. (We recommend having a conda environment)
```
$ conda create --name DECIMER_IMGSEG python=3.7
$ conda activate DECIMER_IMGSEG
$ conda install pip
$ pip install tensorflow-gpu==2.3.0 pillow opencv-python matplotlib scikit-image imantics IPython pdf2image #Install tensorflow==2.3.0 if you do not have a nVidia GPU
$ python3 DECIMER_segmentation.py pdf_file_name 

$python3 Detect_and_save_segmentation.py --input path/to/input/Image (optional)
```
- Segmented images will be saved inside the output folder generated under the name of the pdf file.

## Authors 
- [Kohulan](github.com/Kohulan)
- [Otto Brinkhaus](github.com/OBrink)

## Project page

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)
## More information about our research group

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
