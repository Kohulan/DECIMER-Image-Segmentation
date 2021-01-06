# DECIMER-Image-Segmentation

Chemistry looks back at almost a century of publications on chemical compounds, their structures and properties, in scientific articles. Extracting this knowledge (semi-)automatically and making it available to the world in open-access databases is a current challenge. Apart from mining textual information, Optical Chemical Structure Recognition (OCSR), the translation of an image of a chemical structure into a machine-readable representation, is part of this workflow. As the OCSR process requires an image containing a chemical structure, there is a need for a publicly available tool that automatically recognizes and segments chemical structure depictions from scientific publications. This is especially important for older documents which are only available as scanned pages. Here, we present DECIMER (Deep lEarning for Chemical IMagE Recognition) Segmentation, the first open-source, deep learning-based tool for automated recognition and segmentation of chemical structures from the scientific literature.

The workflow is divided into two main stages. During the detection step, a deep learning model recognizes chemical structure depictions and creates masks which define their positions on the input page. Subsequently, potentially incomplete masks are expanded in a post-processing workflow. The performance of DECIMER Segmentation has been systematically evaluated on three sets of publications from different publishers. 

By making the source code and the trained model publicly available, we hope to contribute to the development of comprehensive chemical data extraction workflows. In order to facilitate access to DECIMER Segmentation, we also developed a web application. The web application, available at https://decimer.ai, lets the user upload a pdf file and retrieve the segmented structure depictions.

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/Validation/Abstract1.png)](https://decimer.ai)

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

$ python3 Detect_and_save_segmentation.py --input path/to/input/Image (optional)
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
