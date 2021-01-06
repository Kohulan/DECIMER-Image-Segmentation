# DECIMER-Image-Segmentation

Deep learning-based image segmentation work by Kohulan Rajan and Otto Brinkhaus

We are using Mask R-CNN to recognize and segment depictions of chemical structures from the published literature. Mask R-CNN can easily detect chemical image depiction after being trained on data annotated from previously published literature. After detection, a mask expansion algorithm completes potentially incomplete masks. Finally, the resulting chemical structure depictions are segmented and saved as individual image files.

## Usage

-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU-enabled PC or simply on CPU, so please do make sure you have all the necessary drivers installed if you are using the GPU.

- Enter the following commands:
```
$ git clone https://github.com/Kohulan/DECIMER-Image-Segmentation
$ cd DECIMER-Image-Segmentation
```
- Download the [trained model](https://storage.googleapis.com/mrcnn-weights/mask_rcnn_molecule.h5)
- Copy the model to DECIMER-Image-Segmentation/model_trained/
- Create a python virtual environment. (We recommend using a conda environment)
```
$ conda create --name DECIMER_IMGSEG python=3.7
$ conda activate DECIMER_IMGSEG
$ conda install pip
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
