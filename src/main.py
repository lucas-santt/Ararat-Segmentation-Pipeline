from classes.PatientData import PatientData
from classes.DicomReader import DicomReader

import os
import cv2
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dicom_path = os.path.join(base_dir, "1-1-SEG.dcm")

    mr_path = "[FULL_PATH_TO_DATA]"
    seg_path = "[FULL_PATH_TO_DATA]"

    patientData = PatientData(mr_path=mr_path, seg_path=seg_path, verbose=True)

    X, Y = patientData.get_data()

    output_path = "output/combined/"
    output_path = os.path.join(base_dir, output_path)

    dicom_reader = DicomReader()
    dicom_reader.write_image(X, Y, out_path=output_path)
    
if __name__ == "__main__":
    main()