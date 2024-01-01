import pandas as pd
import numpy as np
import shutil as sh
from tqdm import tqdm
import os
from typing import List, Tuple


class DataPreparation:
    def __init__(self, patients: int,
                 dataset_path: str,
                 main_dir_path: str):
        """
        :param patients: number of patients to take in dataset
        :param dataset_path: str,
        :param main_dir_path: str
        """
        self.patients = patients
        self.dataset_path = dataset_path
        self.main_dir_path = main_dir_path
        self.org_dir = self.main_dir_path + 'Original_Data_Backup/'
        self.work_dir = self.main_dir_path + 'Working_Data/'
        self.path_structural_images = self.dataset_path + 'images_structural/'
        self.path_automated_seg = self.dataset_path + 'automated_segm/'
        self.path_seg = self.dataset_path + 'images_segm/'

    def sort_patients(self):
        # DF1 stores the ids of patients which have mgmt, auto segm or segm, sutvival and struct_image

        df = pd.read_csv(
            self.main_dir_path + '/Codes/Arbitrary/UPENN-GBM_data_availability.csv')
        df.columns = [c.replace(' ', '_') for c in df.columns]

        df1 = pd.DataFrame({
            "id": [],
            "mgmt": [],
        })

        for i in tqdm(range(len(df))):
            if (df.MGMT[i] == "available" and (
                    df.Automatic_Tumor_Segmentation[i] == "available"
                    or df.Corrected_Tumor_Segmentation[i] == "available")
                    and df.Structural_imaging[i] == "available" and df.Overall_Survival[i] == "available"):
                df2 = pd.DataFrame({'id': [df.ID[i]],
                                    'mgmt': ["available"]},
                                   )
                df1 = df1.append(df2)

        df3 = pd.read_csv(self.main_dir_path + '/Codes/Arbitrary/UPENN-GBM_clinical_info_v1.1.csv')

        df4 = pd.DataFrame({"id": [], "mgmt": [], "age": [], "gender": [], "survival": []})

        arr = df1.id.to_numpy()
        print("Total patients: ", len(arr))
        count = 0

        for i in tqdm(range(len(df3))):

            if df3.ID[i] in arr:

                l: str = "0"

                if df3.MGMT[i] == 'Unmethylated':
                    pass
                else:
                    l: str = "1"
                # print(df3.MGMT[i])
                df5 = pd.DataFrame({'id': [df3.ID[i]],
                                    'mgmt': [l],
                                    'age': [df3.Age_at_scan_years[i]],
                                    'gender': [df3.Gender[i]],
                                    "survival": [df3.Survival_from_surgery_days[i]]
                                    })
                df4 = df4.append(df5)

        df4.to_csv('upenn_data.csv')

    def prepare_data(self):
        """
        Prepare the Data directory for file structuring Step 1
        :return: None
        """
        df6 = pd.read_csv(self.main_dir_path + 'Codes/upenn_data.csv')
        os.mkdir(self.main_dir_path + 'Data/')

        m_neg = self.main_dir_path + 'Data/MGMT_negative/'
        m_pos = self.main_dir_path + 'Data/MGMT_positive/'
        os.mkdir(m_neg)
        os.mkdir(m_pos)

        pos = 0
        neg = 0
        for i in tqdm(range(len(df6))):
            path1 = self.path_structural_images + df6.id[i]
            path_segm = self.path_seg + df6.id[i] + '_segm.nii.gz'
            path_auto_segm = self.path_automated_seg + df6.id[i] + '_automated_approx_segm.nii.gz'
            rename_segm = df6.id[i] + '_seg.nii.gz'

            # if df6.patches64[i] == False :  # patch 64 check
            if df6.mgmt[i] == 0 and neg < self.patients / 2:  # only desired negatives as in config.py
                path2 = m_neg + df6.id[i] + '/'  # create folder for each patient id
                seg_path = path2 + rename_segm  # rename the segm path for both automated and segm
                sh.copytree(path1, path2)  # copy contents of structural images to this patient folder
                if os.path.exists(path_segm):
                    sh.copy(path_segm, seg_path)  # copy the segm file
                elif os.path.exists(path_auto_segm):
                    sh.copy(path_auto_segm, seg_path)  # copy the auto_segm file
                neg += 1
            elif df6.mgmt[i] == 1 and pos < self.patients / 2:  # only desired positives
                path2 = m_pos + df6.id[i] + '/'
                seg_path = path2 + rename_segm
                sh.copytree(path1, path2)
                if os.path.exists(path_segm):
                    sh.copy(path_segm, seg_path)
                elif os.path.exists(path_auto_segm):
                    sh.copy(path_auto_segm, seg_path)
                else:
                    print("No segmentation found for ", df6.id[i])
                pos += 1