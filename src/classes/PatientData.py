import yaml
import os

from classes.DicomReader import DicomReader

class PatientData:
    """
        Prepare the dataset for training and testing.
        Uses the config.yaml and params.yaml files to set paths and parameters.

        This would have to be modified to fit other datasets.
    """

    def __init__(self, mr_path: str, seg_path: str, verbose: bool = False):
        """
            Initializes the PatientData object by scanning the provided directories
            for patient data and storing the paths to MR and SEG DICOM files.

            It follows the PROSTATEx dataset structure:
                <patient_folder>/<data_folder>/<dcom_folder>
            Where <data_folder> is the same for both MR and SEG datasets.

            Args:
                mr_path (str): Full path to the directory containing MR DICOM files.
                seg_path (str): Full path to the directory containing SEG DICOM files.
                verbose (bool): If True, prints detailed information during initialization.
        """
        self.X = []
        self.Y = []
        self.verbose = verbose

        for patient_folder in os.scandir(seg_path):
            if not patient_folder.is_dir():
                continue

            # <base_path>/<patient_folder>
            patient_paths = [
                os.path.join(mr_path, patient_folder.name),
                os.path.join(seg_path, patient_folder.name)
            ]

            date_folder = os.listdir(patient_paths[1])[0] # Same date folder for both MR and SEG

            # Get DICOM folder names
            dcom_folders = [
                os.listdir(os.path.join(patient_paths[0], date_folder))[0],
                os.listdir(os.path.join(patient_paths[1], date_folder))[0]
            ]

            # <patient_folder>/<date_folder>/<dcom_folder>
            dcom_paths = [
                os.path.join(patient_paths[0], date_folder, dcom_folders[0]),
                os.path.join(patient_paths[1], date_folder, dcom_folders[1])
            ]
            
            self.X.append(dcom_paths[0])
            self.Y.append(dcom_paths[1])
            
            if self.verbose:
                print("Patient ID:", patient_folder.name)
                print("    |-", dcom_paths[0])
                print("    |-", dcom_paths[1])
                print()
        
        if len(self.X) != len(self.Y):
            raise Exception("Mismatch between number of MR and SEG files.")
        
        if self.verbose:
            print("Total patients found:", len(self.X))

    def get_data(self, split_part: float = 0.8):
        """ 
            Returns the train and test images datasets as numpy arrays.

            Trusting the PROSTATEx dataset structure, we assume that:
                - MR dcom folder: Has one dcom file for each slice
                - SEG dcom folder: Has all slices in one dcom file
            
            And, with that, we know that the prostate segmentation images
            have index 19->38 in the single SEG dcom file.
        """

        dr = DicomReader()

        im_X = []
        im_Y = []

        for i in range(len(self.X)):
            if self.verbose: print("Processing patient:", i+1, "of", len(self.X))
            for dcom_file in os.listdir(self.X[i]):
                if not dcom_file.endswith(".dcm"):
                    continue
                
                dcom_file_path = os.path.join(self.X[i], dcom_file)
                slice_image = dr.read_image(dcom_file_path)
                im_X.append(slice_image)
            
            for dcom_file in os.listdir(self.Y[i]):
                if not dcom_file.endswith(".dcm"):
                    continue
                
                dcom_file_path = os.path.join(self.Y[i], dcom_file)
                seg_image = dr.read_image(dcom_file_path)

                # Extract only slices 19 to 38
                for slice_idx in range(19, 39):
                    im_Y.append(seg_image[slice_idx, :, :])

        return im_X, im_Y

        train_size = int(len(self.X) * split_part)
        return (self.X[:train_size], self.Y[:train_size]), (self.X[train_size:], self.Y[train_size:])
