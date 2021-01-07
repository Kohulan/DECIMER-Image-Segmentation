# DECIMER-Image-Segmentation

Chemistry looks back at many decades of publications on chemical compounds, their structures and properties, in scientific articles. Liberating this knowledge (semi-)automatically and making it available to the world in open-access databases is a current challenge. Apart from mining textual information, Optical Chemical Structure Recognition (OCSR), the translation of an image of a chemical structure into a machine-readable representation, is part of this workflow. As the OCSR process requires an image containing a chemical structure, there is a need for a publicly available tool that automatically recognizes and segments chemical structure depictions from scientific publications. This is especially important for older documents which are only available as scanned pages. Here, we present DECIMER (Deep lEarning for Chemical IMagE Recognition) Segmentation, the first open-source, deep learning-based tool for automated recognition and segmentation of chemical structures from the scientific literature.

The workflow is divided into two main stages. During the detection step, a deep learning model recognizes chemical structure depictions and creates masks which define their positions on the input page. Subsequently, potentially incomplete masks are expanded in a post-processing workflow. The performance of DECIMER Segmentation has been manually evaluated on three sets of publications from different publishers. The approach operates on bitmap images of journal pages to be applicable also to older articles before the introduction of vector images in PDFs. 

By making the source code and the trained model publicly available, we hope to contribute to the development of comprehensive chemical data extraction workflows. In order to facilitate access to DECIMER Segmentation, we also developed a web application. The web application, available at www.decimer.ai, lets the user upload a pdf file and retrieve the segmented structure depictions.

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/Validation/Abstract1.png)](www.decimer.ai)

## Usage
-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU-enabled PC or simply on CPU, so please do make sure you have all the necessary drivers installed if you are using the GPU.

### We suggest to use DECIMER-Segmentation inside a Conda environment, which makes the dependencies to install easily.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) plattforms (Python 3.7). We recommend to install miniconda3. Using Linux you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
## How to use Decimer-Segmantation

```
$ git clone https://github.com/Kohulan/DECIMER-Image-Segmentation
$ cd DECIMER-Image-Segmentation
```
- You can Download the [trained model](https://storage.googleapis.com/mrcnn-weights/mask_rcnn_molecule.h5)
```
$ conda create --name DECIMER_IMGSEG python=3.7
$ conda activate DECIMER_IMGSEG
$ conda install pip
$ python -m pip install -U pip #Upgrade pip
$ pip install tensorflow-gpu==2.3.0 pillow opencv-python matplotlib scikit-image imantics IPython pdf2image #Install tensorflow==2.3.0 if you do not have a nVidia GPU
$ python3 DECIMER_Segmentation.py pdf_file_name 

$ python3 Detect_and_save_segmentation.py --input path/to/input/Image (optional)
```
- Segmented images are saved in the output folder (which has the name of the pdf file).

#### Separate usage of model detection and mask expansion

- An example of how to apply the model and the mask expansion separately is given [in this Jupyter Notebook](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/DECIMER_Segmentation_notebook.ipynb).

#### Notes for Windows users:

- Execute DECIMER_Segmentation.py in the Anaconda Powershell Prompt


- If you run into an error with the pdf conversion on Windows, you need to [download poppler](http://blog.alivate.com.au/poppler-windows/) and extract the file.
- Open DECIMER-Image-Segmentation/Utils/pdf_2_img_Convert.py
  
- Look for the following line (line 28):

```
  $   pages = convert_from_path(str(path), 500)
```
- Replace it with the following line (Don't forget to modify the path!)
```
  $   pages = convert_from_path(str(path), 500, poppler_path = 'PATH/TO/POPPLER/bin')
```
- Now, everything should run as described above.




  
  
## Authors 
- [Kohulan](https://github.com/Kohulan)
- [Otto Brinkhaus](https://github.com/OBrink)

## decimer.ai

- A web application implementation is available at [decimer.ai](www.decimer.ai), implemented by [Dr.Maria Sorokina](https://github.com/mSorok)

## Citation

- to do

## Project page

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)
## More information about our research group

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
