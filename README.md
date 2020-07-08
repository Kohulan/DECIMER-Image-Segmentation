# DECIMER-Image-Segmentation
Image segmentation initial works by Kohulan Rajan and Otto Brinkhaus

Using Mask R-CNN we are trying to segment the chemical structures from published literature. The mask R-CNN's superior algorithm can easily segment an Image upon training it widely with a mixed set of Data, Includes the Images of published literature with text and corresponding manually annotated data regarding the position of the images.

Initial work was done using training the Mask R-CNN network with our data and deploying on a completely new set of data to see the segmentation accuracy. Later we are working on a cleaning up procedure on the segmented images to fit the Image detection algorithm.

## Usage

-  To use the scripts clone the repository to your local disk. Mask-RCNN runs on a GPU enabled PC, so please do make sure you have all the necessary drivers installed.

```
$ git clone ..
$ cd DECIMER-Image-Segmentation
$ pip3 install tensorflow-gpu==1.15 pillow opencv-python matplotlib scikit-image
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
