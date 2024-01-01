import os
import shutil
import glob
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
import random as rn
from multiprocessing import Pool,Process
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, Activation, AveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from tensorflow.keras.metrics import TruePositives,TrueNegatives,FalsePositives,FalseNegatives,AUC,Recall,Precision
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard
from tensorflow.keras.regularizers import l2
import time
import shutil
import sys

patients = 150
modalities = ['T1']
classifications = ['MGMT_positive', 'MGMT_negative']
block_size = (64,64)
stride = 2
inter_dim = (110, 90)
dataset_path = "D:/MGMT research project/NIfTI-files/"
main_dir_path = "D:/mgmt-patch-model/"

class Trainer():
    def __init__(self, modalities, classifications, main_dir_path, min_size, inter_dim) -> None:
        self.modalities = modalities
        self.classifications = classifications
        self.main_dir_path = main_dir_path
        self.inter_dim = inter_dim
        self.block_h, self.block_w = min_size
        self.work_dir = main_dir_path + 'Data/Working_Data/'

    def interpolation(self, df):
        the_y = []
        the_x = []
        for item in tqdm(range(len(df))):
            img = cv.imread(df['img_path'].iloc[item],cv.IMREAD_GRAYSCALE)
            if img.shape[0]<self.block_h or img.shape[1]<self.block_w: continue
            img = cv.resize(img,self.inter_dim)
            # normalized_img = img.astype(np.float32) / 255.0  # Scaling pixel values to [0, 1]
            the_x.append(img)
            the_y.append(df['mgmt'].iloc[item])

        return the_x, the_y
    
    def spliting(self):
        dict = {
            'pat_path':[],
            'mgmt':[],
        }
        for mgmt_type in self.classifications:
            mgmt = 1 if mgmt_type == 'MGMT_positive' else 0
            for mod in self.modalities:
                patient_dir = self.work_dir + mgmt_type + '/' + mod
                for patients in os.listdir(patient_dir):
                    dict['pat_path'].append(patient_dir + '/' + patients)
                    dict['mgmt'].append(mgmt)

        pat_df = pd.DataFrame(dict)
        # Splitting Data into train and test
        train_df,test_df=train_test_split(pat_df[['pat_path','mgmt']], stratify=pat_df['mgmt'], random_state=57, test_size=0.2)
        print(f'Shape of train_data {train_df.shape}')
        print(f'Shape of test_data {test_df.shape}')

        del pat_df, dict
        dict = {
            'img_path':[],
            'mgmt':[],
        }
        for item in tqdm(range(len(train_df))):
            for img in os.listdir(train_df['pat_path'].iloc[item]):
                dict['img_path'].append(os.path.join(train_df['pat_path'].iloc[item], img))
                dict['mgmt'].append(train_df['mgmt'].iloc[item])
        train_df = pd.DataFrame(dict)

        dict = {
            'img_path':[],
            'mgmt':[],
        }
        for item in tqdm(range(len(test_df))):
            for img in os.listdir(test_df['pat_path'].iloc[item]):
                dict['img_path'].append(os.path.join(test_df['pat_path'].iloc[item], img))
                dict['mgmt'].append(test_df['mgmt'].iloc[item])
        test_df = pd.DataFrame(dict)
        
        return train_df, test_df
    
    def model(self):
        model = Sequential()

        # model.add(Conv2D(16, (5,5), padding='same',input_shape=(90,110,1),kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        model.add(Conv2D(8, (5, 5), padding='same',input_shape=(self.inter_dim[0],self.inter_dim[1],1),
                         kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))

        # model.add(Conv2D(8, (3, 3), padding='same',kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.))

        model.add(Conv2D(4, (3, 3), padding='same',kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))
        
        model.add(Conv2D(4, (3, 3), padding='same',kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        # model.add(Dropout(0.1))


        # model.add(Conv2D(48, (3, 3), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.1))

        model.add(Flatten())  # Convert 3D feature map to 1D feature vector.

        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        
        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Dropout(0.1))

        model.add(Dense(10,kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        # model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam', 
                    metrics=['accuracy',TruePositives(),
                             TrueNegatives(),FalsePositives(),
                             FalseNegatives(),AUC(),Recall(),Precision()])
        return model

    def train_model(self, model, train_x, train_y, test_x, test_y):
        model.fit(np.array(train_x),np.array(train_y), 
            validation_data=(np.array(test_x),np.array(test_y)),
            batch_size= 8,
            epochs = 10,
            shuffle= True
        )

    def main(self):
        train_df, test_df = self.spliting()
        train_x, train_y = self.interpolation(train_df)
        test_x, test_y = self.interpolation(test_df)
        model = self.model()
        self.train_model(model, train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    Trainer_obj = Trainer(modalities, classifications, main_dir_path, (30,30), (64,64))
    Trainer_obj.main()