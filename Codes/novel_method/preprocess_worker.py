import time
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import cv2
from multiprocessing import Process, current_process

import sys
sys.path.append("..")  # Adds the parent directory to sys.path
import config 

# Define the modalities and classifications
# modalities = ['T1', 'T1GD', 'T2', 'FLAIR']
modalities = ['T1']
classifications = ['MGMT_positive', 'MGMT_negative']

# Define patch size and stride
block_h, block_w = (32, 32)
stride = 2

# Interpolated image dimestions
inter_dim = (110, 90)

# Define paths to the BraTS dataset folders
path = config.MAIN_DIR

Preprocess_Dir = path + 'Preprocessed/layers/'
PATH = path + 'Data/'
Org_Dir = PATH + 'Original_Data_Backup/'
Work_Dir = PATH + 'Working_Data/'

# def worker(patient_path: str):
#     print('Reading Images')
#     img_array = cv2.imread(patient_path)
#     print(img_array)
#     return img_array,patient_path

def worker(patient_path):
    print('Reading Images')
    workdir = os.listdir(Work_Dir)
    # for mgmt_type in workdir:
    # for patient in os.listdir(Work_Dir + mgmt_type + '/'):
    patient_patches = []
    for patch in tqdm(os.listdir(Work_Dir + patient_path + '/')):
        try:
            img_array = cv2.imread(os.path.join(Work_Dir, patient_path+'/'+patch), cv2.IMREAD_GRAYSCALE)
            patient_patches.append(img_array)
        except Exception as e:
            print(e)
    print(patient_path + " ✔")
    pickle.dump(patient_patches, open(Preprocess_Dir + patient_path, 'wb'))
    # print(patient_path + " ✔✔")

    
# def worker (i):
#     time.sleep(1)
#     print (9)
