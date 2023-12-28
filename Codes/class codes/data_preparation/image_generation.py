import os
import shutil
import glob
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import cv2 as cv
import numpy as np
from file_structuring import FolderStructure


class GenerateImages(FolderStructure):

    def __init__(self, modalities: List[str], classifications: List[str], block_size: Tuple[int, int], stride: int,
                 inter_dim: Tuple[int, int], dataset_path: str, main_dir_path: str):
        """
        :param modalities: List[str], 
        :param classifications: List[str], 
        :param block_size: Tuple[int, int], 
        :param stride: int, 
        :param inter_dim: Tuple[int, int], 
        :param dataset_path: str, 
        :param main_dir_path: str
        """
        super().__init__(modalities, classifications, block_size, stride, inter_dim, dataset_path, main_dir_path)

    def ConnectedComponentsLabelling(self,gray_img: np.array):
            return cv.connectedComponentsWithStats(gray_img, connectivity=8)

    def segmentation_cropping(self):
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
                        list_of_patients: List[str] = []
                        if patient not in list_of_patients:
                            list_of_patients.append(patient)
                            # print(patient)
                            mod = nib.load('{}_{}.nii.gz'.format(patient, modality))
                            mod_data = mod.get_fdata()  # Getting the data from the nifti file
                            seg_mask = nib.load('{}_seg.nii.gz'.format(patient))
                            seg_mask_data = seg_mask.get_fdata()  # Getting the data from the nifti file

                            # Extracting layers from mask that have non-zero values
                            z = np.any(seg_mask_data, axis=(0, 1))
                            zmin, zmax = np.where(z)[0][[0, -1]]  # zmin & zmax are the first and last layer number non
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
                                d[row_min:row_max + 1, col_min:col_max + 1, layer] = 1  # Replacing tumor region with highest pixel value

                            #  Multiply modality data with the new segmentation mask
                            tumor = np.multiply(mod_data, d)

                            # Removing Zero valued layers
                            tumor_layers = tumor[:, :, ~(tumor == 0).all((0, 1))]

                            # Converting to png files
                            cropped_list = []
                            for lay in range(0, tumor_layers.shape[2]):
                                _, binary_img = cv.threshold(lay, 0, 255, cv.THRESH_BINARY)
                                n_labels, labels, stats, centroids = self.ConnectedComponentsLabelling(binary_img)
                                for region in range(1,n_labels):
                                    indices = np.where(labels==region)
                                    r_max = indices[0].max()
                                    r_min = indices[0].min()
                                    c_min = indices[1].min()
                                    c_max = indices[1].max()
                                    # coords = np.argwhere(tumor_layers[:, :, lay])
                                    # x_min, y_min = coords.min(axis=0)
                                    # x_max, y_max = coords.max(axis=0)
                                    cropped = mod_data[r_min:r_max + 1, c_min:c_max + 1, lay]
                                    cropped *= (255.0 / cropped.max())  # Normalizing the values
                                    # mod_data[:, :, lay] *= (255.0 / mod_data[:, :, lay].max())
                                    cropped_list.append(cropped)

                            frame = 0
                            for item in cropped_list:
                                if ((item.shape[0] * item.shape[1]) >= 300):
                                    frame = frame + 1
                                    im = Image.fromarray(item)
                                    im = im.convert('RGB')
                                    # im = im.resize(self.inter_dim, Image.Resampling.LANCZOS)  # interpolating
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

    def generate_patches(self):
        print('Creating patches...')
        try:
            workdir = os.listdir(self.work_dir)
            for mgmt_type in workdir:  # 2 - MGMT_pos and MGMT_neg
                modality_path = os.path.join(self.work_dir, mgmt_type + '/')
                modalitypath = os.listdir(modality_path)
                for modality in self.modalities:  # 4 - FLair t1 t2 gd
                    patient_path = os.path.join(modality_path, modality + '/')
                    patientpath = os.listdir(patient_path)
                    for patient in patientpath:
                        file_path = os.path.join(patient_path, patient + '/')
                        filepath = os.listdir(file_path)

                        os.chdir(modality_path)

                        if not os.path.exists(patient):
                            os.mkdir(patient)
                            # shutil.copytree('{}'.format(modality), '{}'.format(patient))

                        # chdir to the patient folder
                        os.chdir(os.path.join(modality_path, modality + '/' + patient + '/'))
                        for png in tqdm(glob.glob('*.png')):
                            img = Image.open(png)
                            img_w, img_h = img.size

                            file_name, extension = os.path.splitext(png)

                            save_path = os.path.join(modality_path, patient + '/')
                            # print('Save_path:', Save_path)

                            frame_num = 0
                            count_row = 0

                            for row in range(0, img_h, self.stride):
                                if img_h - row >= self.block_h:
                                    count_row += 1
                                    count_col = 0

                                    for col in range(0, img_w, self.stride):
                                        if (img_w - col >= self.block_w):
                                            count_col += 1
                                            frame_num += 1

                                            box = (col, row, col + self.block_w, row + self.block_h)
                                            a = img.crop(box)
                                            a.save(save_path + file_name + '_row_' + str(count_row) + '_col_' + str(
                                                count_col) + '.png')

                            img.close()
                            os.remove(png)

                    print('Patching done for {} modality'.format(modality))

                    # shutil.rmtree(os.path.join(Modality_path, modality))
                print('Patching done for {}'.format(mgmt_type))

        except Exception as e:
            print('Error in Creating_patches() function')
            print(e)

    def update_patch_record(self):
        os.chdir(self.main_dir_path)
        patch_df = pd.read_csv(self.main_dir_path + 'Codes/upenn_data.csv')

        for mgmt_type in os.listdir(self.work_dir):

            # Delete existing Modality empty folders if any
            for modality in self.modalities:
                if modality in os.listdir(self.work_dir + mgmt_type + '/'):
                    shutil.rmtree(self.work_dir + mgmt_type + '/' + modality)

            # update the csv with patches of patients
            for patient in os.listdir(self.work_dir + mgmt_type + '/'):
                patch_df.loc[patch_df.id == patient, 'patches64'] = True  # mention the patches column to update

        patch_df.to_csv(self.main_dir_path + 'Codes/upenn_data.csv', index=False)

    def main(self):
        self.backup_folders()
        self.create_modality_folders()
        self.segmentation_cropping()
        # self.generate_patches()
        self.update_patch_record()