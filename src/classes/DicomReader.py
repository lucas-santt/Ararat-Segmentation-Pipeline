from matplotlib import pyplot as plt
import numpy as np
import cv2

from pydicom import dcmread
from imageio import imwrite

class DicomReader:
    """ 
        Class to read and extract information from DICOM files. 
        It can also write image arrays to PNG files.
            For it, it has an index to name the output 
            files sequentially, starting from 1.
    """

    def __init__(self):
        self.name_index = 1

    def _normalize(self, image_array: np.ndarray) -> np.ndarray[np.float64]:
        """ Normalizes the image array to the range 0-255. """
        max_val = float(np.max(image_array))
        norm = image_array.copy().astype(float)

        if max_val == 0:
            return norm
        
        norm *= 255 / max_val
        return norm

    def _preprocess(self, filename: str) -> np.ndarray[np.uint8]:
        """ Preprocesses the DICOM file and returns a normalized image array. """
        ds = dcmread(filename)
        ds_image = self._normalize(ds.pixel_array)
        ds_image = ds_image.astype(np.uint8)

        return ds_image

    def read_image(self, filename, out_path="output/") -> np.ndarray[np.uint8]:
        """ Reads a DICOM file and returns the image array after preprocessing. """
        image_array = self._preprocess(filename)

        return image_array

    def write_image(self, image_array: np.ndarray, out_path: str = "output/"):
        """
            Writes an image array to a png file.
            If the array has multiple slices, saves each slice as a separate image.
        """
        if len(image_array.shape) < 3:
            cv2.imwrite(f"{out_path}/{self.name_index}.png", image_array)
        else:
            # In case of multiple slices
            for i in range(image_array.shape[0]):
                cv2.imwrite(f"{out_path}/{self.name_index}_{i+1}.png", image_array[i, :, :])
        
        self.name_index += 1

    def write_image(self, X_images: np.ndarray, Y_images: np.ndarray, out_path: str ="output/combined/"):
        """ Writes the combined image array (X and Y) to the specified output path. """

        for i, (x_img, y_mask) in enumerate(zip(X_images, Y_images)):
            # The two images must be in BGR format for proper visualization
            if len(x_img.shape) == 2:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            if len(y_mask.shape) == 2:
                y_mask = cv2.cvtColor(y_mask, cv2.COLOR_GRAY2BGR)

            # Ensure they have the same height (necessary for hstack)
            if x_img.shape[0] != y_mask.shape[0]:
                y_mask = cv2.resize(y_mask, (x_img.shape[1], x_img.shape[0]))

            combined = np.hstack((x_img, y_mask))
            cv2.imwrite(f"{out_path}/{i}.png", combined)

    def reset_index(self):
        """ Resets the name index to 1. """
        self.name_index = 1