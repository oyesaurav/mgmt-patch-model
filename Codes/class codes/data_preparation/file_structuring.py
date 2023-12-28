import os
import shutil
import glob
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np


class FolderStructure:

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

    def backup_folders(self):
        """
        Backup the data as original directory and working directory. The working dir can be reconstructed from the
        original one.
        :return: None
        """
        print('Creating Backup')
        # Copy data for backup
        try:
            shutil.copytree(self.main_dir_path, self.org_dir)
        except Exception as e:
            print(e)

        # Copy data for working
        try:
            shutil.copytree(self.org_dir, self.work_dir)
        except Exception as e:
            print(e)

        # Deleting folder, listed in the CATEGORIES list, after creating Cases
        for cate in self.classifications:
            try:
                shutil.rmtree(self.main_dir_path + cate)  # Deleting Folders of CATEGORIES list
            except Exception as e:
                print(e)
        print('Backup Created')

    def recreate_folders(self):
        """
        Recreate the data as original directory and working directory.
        :return: None
        """
        print('Reconstructing')
        # Deleting working directory
        shutil.rmtree(self.main_dir_path + "Working_data/")
        # Copying data from backup
        try:
            shutil.copytree(self.org_dir, self.work_dir)
        except Exception as e:
            print(e)
        print('Reconstruction complete')

    def create_modality_folders(self):
        """
        Create the modality folders needed for the training.
        :return:
        """
        print("Copying files into the corresponding modality folders...")
        train_folder_1 = os.listdir(self.work_dir)
        for pos_neg in tqdm(train_folder_1):
            patient_folders = os.listdir(os.path.join(self.work_dir, pos_neg))
            for patient_folder in patient_folders:
                for modality in self.modalities:
                    modality_folder_path = os.path.join(self.work_dir, pos_neg, modality)
                    # print(modality_folder_path)
                    modality_patient_folder_path = os.path.join(modality_folder_path, patient_folder)
                    # print(modality_patient_folder_path)
                    if not os.path.exists(modality_patient_folder_path):
                        os.makedirs(modality_patient_folder_path)

                    modality_file_path = os.path.join(self.work_dir, pos_neg, patient_folder,
                                                      '{}_{}.nii.gz'.format(patient_folder, modality))
                    # print(modality_file_path)
                    seg_file_path = os.path.join(self.work_dir, pos_neg, patient_folder,
                                                 '{}_seg.nii.gz'.format(patient_folder))
                    # print(seg_file_path)
                    if os.path.exists(modality_file_path) and os.path.exists(seg_file_path):
                        shutil.copy(modality_file_path, modality_patient_folder_path)
                        shutil.copy(seg_file_path, modality_patient_folder_path)
                    else:
                        print("Either {} or {} does not exist".format(modality_file_path, seg_file_path))
                # delete patient folder after copying files
                shutil.rmtree(os.path.join(self.work_dir, pos_neg, patient_folder))

        print("Modality folders done.")