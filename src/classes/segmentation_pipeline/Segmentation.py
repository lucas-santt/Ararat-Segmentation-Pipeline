import SimpleITK as sitk
import numpy as np

class Segmentation():
    """ Segmentação criando uma máscara 3D da região de interesse ROI"""
   
    @staticmethod
    def create_image_mask(image_sitk: sitk.Image, center_mm: list, radius_mm: int = 5) -> sitk.Image:
        """
            Cria uma máscara binária 3D referente a
            presença de cada voxel na esfera passada por argumentos

            Arguments:
                image_sitk (sitk.Image)  - Imagem original
                center_mm (list)         - Centro da esfera
                radius_mm (int)          - Raio da esfera, 5 como default

            Returns:
                mask_itk (sitk.Image)    - Máscara resultante
        """

        spacing = np.array(image_sitk.GetSpacing()) # Valor de um voxel em medida real
        voxel_center_continuous = image_sitk.TransformPhysicalPointToContinuousIndex(center_mm)

        # Geramos três matrizes de indices que contem a coordenada de todos os voxels
        zz, yy, xx = np.mgrid[:image_sitk.GetDepth(), :image_sitk.GetHeight(), :image_sitk.GetWidth()]

        # xx, yy, zz = np.mgrid[:image_sitk.GetWidth(), :image_sitk.GetHeight(), :image_sitk.GetDepth()]        
        
        # Calcula a distancia de cada voxel ao centro dado
        # TODO Não era para ter uma sqrt aqui não?
        distance_sq = (
            (spacing[0] * (xx - voxel_center_continuous[0]))**2 +
            (spacing[1] * (yy - voxel_center_continuous[1]))**2 +
            (spacing[2] * (zz - voxel_center_continuous[2]))**2
        )
        
        # Matriz booleana em 0 e 1 transformada em imagem
        mask_np = (distance_sq <= radius_mm**2).astype(np.uint8)
        mask_itk = sitk.GetImageFromArray(mask_np)
        mask_itk.CopyInformation(image_sitk)
        
        return mask_itk