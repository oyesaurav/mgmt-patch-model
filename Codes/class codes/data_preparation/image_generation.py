import os
import shutil
import glob
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np


class GenerateImages:

    def __init__(self, modalities: List[str],
                 classifications: List[str],
                 block_size: Tuple[int, int],
                 stride: int,
                 inter_dim: Tuple[int, int],
                 dataset_path: str,
                 main_dir_path: str):
        """
        :param modalities: List[str], 
        :param classifications: List[str], 
        :param block_size: Tuple[int, int], 
        :param stride: int, 
        :param inter_dim: Tuple[int, int], 
        :param dataset_path: str, 
        :param main_dir_path: str
        """
        self.modalities = modalities
        self.classifications = classifications
        self.block_h, self.block_w = block_size
        self.stride = stride
        self.inter_dim = inter_dim
        self.dataset_path = dataset_path
        self.main_dir_path = main_dir_path
        self.org_dir = self.main_dir_path + 'Original_Data_Backup/'
        self.work_dir = self.main_dir_path + 'Working_Data/'

    def main(self):
        """
        Main function to generate the images according to the provided block size and inter_dim
        :return: None
        """
        print('Generating Images...')
        try:
            workdir = os.listdir(self.work_dir)
            for mgmt_class in workdir:
                modality_path = os.path.join(self.work_dir, mgmt_class + '/')
                modalitypath = os.listdir(modality_path)
                print("Modality:", modalitypath)
                for modality in modalitypath:
                    patient_path = os.path.join(modality_path, modality + '/')
                    patientpath = os.listdir(patient_path)
                    for patient_folder in patientpath:
                        file_path = os.path.join(patient_path, patient_folder + '/')
                        filepath = os.listdir(file_path)
                        os.chdir(file_path)
                        patient = patient_folder.split('_')[0] + '_' + patient_folder.split('_')[1]
                        list_of_patients: list[str] = []
                        if patient not in list_of_patients:
                            list_of_patients.append(patient)
                            # print(patient)
                            mod = nib.load('{}_{}.nii.gz'.format(patient, modality))
                            mod_data = mod.get_fdata()  # Getting the data from the nifti file
                            seg_mask = nib.load('{}_seg.nii.gz'.format(patient))
                            seg_mask_data = seg_mask.get_fdata()  # Getting the data from the nifti file

                            # Extracting layers from mask that have non-zero values
                            z = np.any(seg_mask_data, axis=(0, 1))
                            zmin, zmax = np.where(z)[0][[0,
                                                         -1]]  # zmin & zmax are the first and last layer number non
                            # zero values in the z axis

                            # Creating a new mask to remove segmentation
                            d = seg_mask_data
                            for layer in range(zmin, zmax + 1):
                                nonzero = np.nonzero(d[:, :, layer])
                                r = nonzero[0]
                                c = nonzero[1]
                                if (r.size == 0) or (c.size == 0):
                                    continue
                                row_min = np.min(r)
                                row_max = np.max(r)
                                col_min = np.min(c)
                                col_max = np.max(c)
                                d[row_min:row_max + 1, col_min:col_max + 1, layer] = 1  # Replacing tumor region
                                # values by 1

                            #  Multiply modality data with the new segmentation mask
                            tumor = np.multiply(mod_data, d)

                            # Removing Zero valued layers
                            tumor_layers = tumor[:, :, ~(tumor == 0).all((0, 1))]

                            # Converting to png files
                            cropped_list = []
                            for lay in range(0, tumor_layers.shape[2]):
                                coords = np.argwhere(tumor_layers[:, :, lay])
                                x_min, y_min = coords.min(axis=0)
                                x_max, y_max = coords.max(axis=0)
                                cropped = tumor_layers[x_min:x_max + 1, y_min:y_max + 1, lay]
                                cropped *= (255.0 / cropped.max())  # Normalizing the values
                                mod_data[:, :, lay] *= (255.0 / mod_data[:, :, lay].max())
                                cropped_list.append(cropped)

                            frame = 0
                            for item in cropped_list:
                                if ((item.shape[0] * item.shape[1]) >= 300):
                                    frame = frame + 1
                                    im = Image.fromarray(item)
                                    im = im.convert('RGB')
                                    im = im.resize(self.inter_dim, Image.Resampling.LANCZOS)  # interpolating
                                    im.save('{}_{}_{}.png'.format(patient, modality, frame))
                                    im.close()
                            # Deleting the nifti files
                            nii_path = os.listdir(file_path)
                            for nii in nii_path:
                                try:
                                    if nii.startswith(patient) and nii.endswith('.gz'):
                                        os.remove(nii)
                                except Exception as e:
                                    print('Error in deleting nifti files')
                                    print(e)
        except Exception as e:
            print('Error in Generate_images()')
            print(e)