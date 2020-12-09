# DECIMER-Image-Segmentation

Deep learning-based image segmentation work by Kohulan Rajan and Otto Brinkhaus

We are using Mask R-CNN to recognize and segment depictions of chemical structures from the published literature. Mask R-CNN can easily detect chemical image depiction after being trained on data annotated from previously published literature. After detection, a Mask expansion algorithm will go through the detected structures for its completeness. Finally, the Segmentation algorithm segments out the chemical image depictions into individual image files.

## Usage

-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU enabled PC or simply on CPU, so please do make sure you have all the necessary drivers installed if you are using GPU.

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
$ python3 DECIMER_Segmentation.py pdf_file_name 

$python3 Detect_and_save_segmentation.py --input path/to/input/Image (optional)
```
- Segmented images will be saved inside the output folder generated under the name of the pdf file.

## Authors 
- [Kohulan](github.com/Kohulan)
- [Otto Brinkhaus](github.com/OBrink)

## decimer.ai

- A web application implementation is available at [decimer.ai](https://decimer.naturalproducts.net), implemented by [Dr.Maria Sorokina](https://github.com/mSorok)

## Citation

- to do

## Project page

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)
## More information about our research group

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
