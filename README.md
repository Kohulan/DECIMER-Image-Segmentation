# DECIMER-Image-Segmentation
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIt)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image-Segmentation.svg)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image-Segmentation.svg)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/graphs/contributors/)
[![DOI](https://zenodo.org/badge/268631290.svg)](https://zenodo.org/badge/latestdoi/268631290)

Chemistry looks back at many decades of publications on chemical compounds, their structures and properties, in scientific articles. Liberating this knowledge (semi-)automatically and making it available to the world in open-access databases is a current challenge. Apart from mining textual information, Optical Chemical Structure Recognition (OCSR), the translation of an image of a chemical structure into a machine-readable representation, is part of this workflow. As the OCSR process requires an image containing a chemical structure, there is a need for a publicly available tool that automatically recognizes and segments chemical structure depictions from scientific publications. This is especially important for older documents which are only available as scanned pages. Here, we present DECIMER (Deep lEarning for Chemical IMagE Recognition) Segmentation, the first open-source, deep learning-based tool for automated recognition and segmentation of chemical structures from the scientific literature.

The workflow is divided into two main stages. During the detection step, a deep learning model recognizes chemical structure depictions and creates masks which define their positions on the input page. Subsequently, potentially incomplete masks are expanded in a post-processing workflow. The performance of DECIMER Segmentation has been manually evaluated on three sets of publications from different publishers. The approach operates on bitmap images of journal pages to be applicable also to older articles before the introduction of vector images in PDFs. 

By making the source code and the trained model publicly available, we hope to contribute to the development of comprehensive chemical data extraction workflows. In order to facilitate access to DECIMER Segmentation, we also developed a web application. The web application, available at https://decimer.ai, lets the user upload a pdf file and retrieve the segmented structure depictions.

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/Validation/Abstract1.png)](https://decimer.ai)

## Usage
-  To use DECIMER Segmentation, clone the repository to your local disk. Mask-RCNN runs on a GPU-enabled PC or simply on CPU, so please do make sure you have all the necessary drivers installed if you are using the GPU.

##### We recommend to use DECIMER-Segmentation inside a Conda environment to facilitate the installation of the dependencies.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) platforms (Python 3.0). We recommend to install miniconda3. Using Linux you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
## How to install DECIMER-Segmentation

```
$ git clone https://github.com/Kohulan/DECIMER-Image-Segmentation
$ cd DECIMER-Image-Segmentation
$ conda create --name DECIMER_IMGSEG python=3.10
$ conda activate DECIMER_IMGSEG
$ conda install pip
$ python -m pip install -U pip #Upgrade pip
$ pip install .

#From Pypi
$ pip install decimer-segmentation
```

### The Mask-RCNN Model is available at: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7228583.svg)](https://doi.org/10.5281/zenodo.7228583)

## How to use DECIMER-Segmentation
- The repository contains a script that can be used for the segmentation of chemical structures from an image of a scanned page or from a pdf document:
```
$ python3 segment_structures_in_document.py file_name (the file can be an image of a scanned page or a pdf document) 
```
- Segmented images are saved in the output folder (which has the name of the pdf file).

- Alternatively, you can use integrate DECIMER Segmentation in your Python code:
```
from decimer_segmentation import segment_chemical_structures, segment_chemical_structures_from_file
import cv2

# Segment structures in scanned page image (np.array)
page = cv2.imread(scanned_page_file_path)
segments = segment_chemical_structures(page, expand=True)

# Segment structures from file (pdf or image)
# Windows users may need to specify the location of their poppler installation with the poppler_path argument if they want to process pdf files
segments = segment_chemical_structures_from_file(path, expand=True, poppler_path=None)

```

- More examples are given [in this Jupyter Notebook](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/DECIMER_Segmentation_notebook.ipynb).

#### Notes for Windows users:

- Execute DECIMER_Segmentation.py in the Anaconda Powershell Prompt


- If you run into an error with the pdf conversion on Windows, you need to [download poppler](http://blog.alivate.com.au/poppler-windows/) and extract the file.
- The method segment_chemical_structures_from_file() takes a 'poppler_path' argument where the user can specify the path of their poppler installation ('PATH/TO/POPPLER/bin').




  
  
## Authors 
- [Kohulan](https://github.com/Kohulan)
- [Otto Brinkhaus](https://github.com/OBrink)

## decimer.ai

- A web application implementation is available at [decimer.ai](https://decimer.ai), implemented by [Otto Brinkhaus](https://github.com/OBrink)

## Citation
Rajan, K., Brinkhaus, H.O., Sorokina, M. et al. DECIMER-Segmentation: Automated extraction of chemical structure depictions from scientific literature. J Cheminform 13, 20 (2021). https://doi.org/10.1186/s13321-021-00496-1

## Project page

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)
## More information about our research group

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)
