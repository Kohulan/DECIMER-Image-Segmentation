'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
import os
import numpy as np
import skimage.io
import cv2
from PIL import Image
import argparse
import warnings
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import moldetect
from Scripts.complete_structure import complete_structure_mask

warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))


class InferenceConfig(moldetect.MolDetectConfig):
    """
    Inference configuration class for MRCNN
    """
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    
class DecimerSegmentation:
    """
    This class contains the main functionalities of DECIMER Segmentation
    """
    def __init__(self):
        self.model = self.load_model()

    def load_model(self) -> modellib.MaskRCNN:
        """
        This function loads the segmentation model and returns it. The weights
        are downloaded if necessary.

        Returns:
            modellib.MaskRCNN: MRCNN model with trained weights
        """
        # Define directory with trained model weights
        root_dir = os.path.split(__file__)[0]
        model_path = os.path.join(root_dir, "model_trained/mask_rcnn_molecule.h5")
        # Download trained weights if needed
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=".",
                                  config=InferenceConfig())
        # Load weights
        model.load_weights(model_path, by_name=True)
        return model

    def get_segments(self, image):
        # Structure detection
        masks = self.get_masks(image)
        # Mask expansion
        expanded_masks = complete_structure_mask(image_array=image,
                                                 mask_array=masks)
        # Save segments
        zipper = (expanded_masks, IMAGE_PATH, output_directory)
        segmented_img = save_segments(zipper)
        return segmented_img

    def get_masks(self, image: np.array) -> np.array:
        """
        This function runs the segmentation model and returns an
        array with the masks (shape: height, width, num_masks).
        Slicing along the third axis of the output of this function
        yields a binary array of shape (h, w) for a single structure-
        """
        results = self.model.detect([image], verbose=1)
        masks = results[0]['masks']
        return masks


def save_segments(zipper):
    expanded_masks, IMAGE_PATH, output_directory = zipper
    mask = expanded_masks

    for i in range(mask.shape[2]):
        image = cv2.imread(os.path.join(IMAGE_PATH), -1)

        for j in range(image.shape[2]):
            image[:,:,j] = image[:,:,j] * mask[:,:,i]

        #Remove unwanted background
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,thresholded = cv2.threshold(grayscale,0,255,cv2.THRESH_OTSU)
        bbox = cv2.boundingRect(thresholded)
        x, y, w, h = bbox
        foreground = image[y:y+h, x:x+w]

        masked_image = np.zeros(image.shape).astype(np.uint8)
        masked_image = visualize.apply_mask(masked_image, mask[:, :, i],[1,1,1])
        masked_image = Image.fromarray(masked_image)
        masked_image = masked_image.convert('RGB')

        im_gray = cv2.cvtColor(np.asarray(masked_image), cv2.COLOR_RGB2GRAY)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #Removal of transparent layer - black background
        _,alpha = cv2.threshold(im_bw,0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        background = dst[y:y+h, x:x+w]
        trans_mask = background[:,:,3] == 0
        background[trans_mask] = [255, 255, 255, 255]
        new_img = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

        #Save segments
        #Making directory for saving the segments
        if os.path.exists(output_directory+"/segments"):
            pass
        else:
            os.system("mkdir "+str(os.path.normpath(output_directory+"/segments")))

        #Define the correct path to save the segments
        segment_dirname = os.path.normpath(output_directory+"/segments/")
        filename = str(IMAGE_PATH).replace("\\", "/").split("/")[-1][:-4]+"_%d.png"%i
        file_path = os.path.normpath(segment_dirname + "/" +filename)

        print(file_path)
        cv2.imwrite(file_path, new_img)
    return output_directory+"/segments/"


def main():
    # Handle input arguments
    parser = argparse.ArgumentParser(description="Select the chemical structures from a scanned literature and save them")
    parser.add_argument(
        '--input',
        help='Enter the input filename',
        required=True
    )
    args = parser.parse_args()

    # Define image path and output path
    IMAGE_PATH = os.path.normpath(args.input)
    output_directory = str(IMAGE_PATH) + '_output'
    if os.path.exists(output_directory):
        pass
    else:
        os.system("mkdir " + output_directory)

    # Segment chemical structure depictions
    zipper = get_segments(output_directory, IMAGE_PATH)
    print("Segmented Images can be found in: ", str(os.path.normpath(zipper)))

if __name__ == '__main__':
    main()
