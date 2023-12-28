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