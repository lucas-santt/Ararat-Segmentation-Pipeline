import pandas as pd
import SimpleITK as sitk
import os
from radiomics.featureeextractor import RadiomicsFeatureExtractor
from classes.segmentation import Segmentation

class RadiomicsPipeline():
    """ Responsável por extrair features da ROI de determinada imagem ou imagens DICOM """

    def __init__(self, spreadsheet_path, images_base_path, series_id='t2_tse_tra', output_filename="radiomics_features.csv"):
        """
            Arguments:
                spreadsheet_path (str)  - Planilha com as informações ROI de cada imagem
                images_base_path (str)  - Pasta que armazena as imagens DICOM
                _series_id (str)         - Sufixo das imagens, 't2_tse_tra' como default
        """
        self._spreadsheet_path = spreadsheet_path
        self._images_base_path = images_base_path
        self._output_filename = output_filename
        
        self._series_id = series_id.lower().replace(" ", "")
        self.__extractor = RadiomicsFeatureExtractor()

    def _find_dicom_series(self, patient_path):
        """
            Retorna o diretorio de imagens com o sufixo passado no construtor

            Arguments:
                patient_path (str)  - Diretorio do paciente
            Returns:
                str                 - Diretorio resultante, None se não encontrado 
        """
        for root, dirs, _ in os.walk(patient_path):
            for d in dirs:
                d = d.lower().replace(" ", "")
                # TODO Pq a len deve ser maior que 10?
                if (self._series_id in d) and len(os.listdir(os.path.join(root, d))) > 10:
                    print(f"  > Série encontrada: {d}")
                    return os.path.join(root, d)
        return None
    
    def _get_features(self, patient_row):
        """
            Processa as imagens DICOM e extrai suas features

            Arguments:
                patient_row ()       - Linha de informações do paciente
            
            Returns:
                features (pd.Series) - Dataframe com todas as features da lesão
        """
        patient_id = patient_row['PatientID']
        finding_id = patient_row['FindingID']
            
        coords_mm = [float(c) for c in patient_row['WorldCoordinates'].split()]
        patient_path = os.path.join(self._images_base_path, patient_id)
        series_path = self._find_dicom_series(patient_path)

        if series_path == None: 
            raise FileNotFoundError(f"Couldn't find any series '{self._series_id} in {patient_path}")
        
        # Empilha todas as imagens 2D em uma 3D
        images2D_path = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(series_path)
        image_sitk = sitk.ReadImage(images2D_path)

        mask_sitk = Segmentation.create_image_mask(image_sitk, coords_mm, radius_mm=5)
        features = self.__extractor.execute(image_sitk, mask_sitk)
        
        features['PatientID'] = patient_id
        features['FindingID'] = finding_id
        
        # Adiciona colunas extras se for t2tsetra
        if 't2tsetra' in self._series_id:
            for col in ['ggg', 'zone', 'ClinSig']: 
                if col in patient_row and pd.notna(patient_row[col]):
                    features[col] = patient_row[col]

        return features

    def run(self):
        """
            Process all DICOM images, extract its features
            and export them onto a csv file. 
            Uses the class information (spreadsheet, path to images 
            and _series_id) in the process.

            Returns:
                pd.DataFrame - Features dataframe, None if it
                                couldn't find any injury 
                                information successfully
        """
        try:
            df_lesoes = pd.read_csv(self._spreadsheet_path)
            df_lesoes = df_lesoes.rename(columns={'ProxID': 'PatientID', 'pos': 'WorldCoordinates', 'fid': 'FindingID'})
        except FileNotFoundError:
            print(f"! Can't find spreadsheet '{self._spreadsheet_path}' \n")
            return

        all_results = []
        
        for _, row in df_lesoes.iterrows():
            try:
                all_results.append(self._get_features(row))
            except Exception as e:
                print(f"! Error in processing {row['PatientID']} - Injury {row['FindingID']}: \n{e} \n")
                continue
     
        if not all_results:
            print("! Couldn't process any injury successfully\n")
            return None

        df_final = pd.DataFrame(all_results)
        df_final.to_csv(self._output_filename, index=False)
        return df_final
    
    # region Encapsulament
    
    @property
    def spreadsheet_path(self):
        return self._spreadsheet_path
    
    @spreadsheet_path.setter
    def spreadsheet_path(self, spreadsheet_path):
        self._spreadsheet_path = spreadsheet_path

    @property
    def images_base_path(self):
        return self._images_base_path
    
    @images_base_path.setter
    def images_base_path(self, images_base_path):
        self._images_base_path = images_base_path

    @property
    def output_filename(self):
        return self._output_filename
    
    output_filename.setter
    def output_filename(self, output_filename):
        self._output_filename = output_filename

    @property
    def series_id(self):
        return self._series_id

    @series_id.setter
    def series_id(self, series_id):
        self._series_id = series_id.lower().replace(" ", "")

    # endregion