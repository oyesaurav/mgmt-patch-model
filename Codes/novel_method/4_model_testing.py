# %%
#  Import libraries and define variables
import os
import shutil
import glob
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np
import cv2
import pickle
import cv2
import tensorflow as tf
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch

# Define the modalities and classifications
modalities = ['t1', 't1ce', 't2', 'flair']
classifications = ['mgmt_positive', 'mgmt_negative']

# Define patch size and stride
block_h, block_w = 32, 32
stride = 2

# Interpolated image dimestions
inter_dim = (128, 128)

# Loading model
load_model = 'latest_model.h5'

# Define paths to the BraTS dataset folders
path = '/Users/vitthal/Documents/Research/DataScience/MedicalResearch/mgmt/'

PATH = path + 'Data/BRATS/novel_data/'
Test_Dir = PATH + 'Working_Data/test/'
Test_Dir1 = PATH + 'test/'
Org_Dir = PATH + 'Original_Data_Backup/'
Work_Dir = PATH + 'Working_Data/'

# %%
#  Function Defination --> Create backup for test folder
def Backup():
    print('Creating backup for test folder...')
    # Copy data from test folder inside test folder
    try:
        if not os.path.exists(Org_Dir + 'test/'):
            shutil.copytree(Test_Dir1, Org_Dir + 'test/')
    except Exception as e:
        print('Error while creating backup!')
        print(e)
    #  copy data for working folder
    try:
        if not os.path.exists(Work_Dir + 'test/'):
            shutil.copytree(Test_Dir1, Work_Dir + 'test/')
    except Exception as e:
        print('Error while creating backup!')
        print(e)
    # remove test folder
    if os.path.exists(Test_Dir1):
        shutil.rmtree(Test_Dir1)

# %%
# Function Defination --> Reconstruct folder
def Reconstruct():
    print('Reconstructing test folder...')
    try:
        if not os.path.exists(Test_Dir1):
            shutil.copytree(Org_Dir + 'test/', Test_Dir1)
    except Exception as e:
        print('Error while reconstructing test folder!')
        print(e)
    # remove backup folder
    if os.path.exists(Org_Dir + 'test/'):
        shutil.rmtree(Org_Dir + 'test/')
    if os.path.exists(Work_Dir + 'test/'):
        shutil.rmtree(Work_Dir + 'test/')

# %%
# Function Defination --> Create modality folders for independent cohort
def create_modality_folders():
    print('Creating Modality Folders')
    test_folder = os.listdir(Test_Dir)
    if '.DS_Store' in test_folder:
        test_folder.remove('.DS_Store')
        print('Removed .DS_Store from test folder')
    for pos_neg in tqdm(test_folder):
        patient_folders = os.listdir(os.path.join(Test_Dir, pos_neg))
        # print('Patient Folders: {}'.format(patient_folders))
        if '.DS_Store' in patient_folders:
            patient_folders.remove('.DS_Store')
            print('Removed .DS_Store from patient folder')
        for patient in patient_folders:
            for modality in modalities:
                print('Patient: ',patient)
                modality_folder_path = os.path.join(Test_Dir, pos_neg, modality)
                modality_patient_folder_path = os.path.join(modality_folder_path, patient)
                # print(modality_patient_folder_path)
                if not os.path.exists(modality_folder_path):
                    # print('Creating folder: {}'.format(modality_folder_path))
                    os.makedirs(modality_folder_path)
                if not os.path.exists(modality_patient_folder_path):
                    # print('Creating folder: {}'.format(modality_patient_folder_path))
                    os.makedirs(modality_patient_folder_path)

                modality_file_path = os.path.join(Test_Dir, pos_neg, patient, '{}_{}.nii.gz'.format(patient, modality))
                seg_file_path = os.path.join(Test_Dir, pos_neg, patient, '{}_seg.nii.gz'.format(patient))

                if os.path.exists(modality_file_path) and os.path.exists(seg_file_path):
                   shutil.copy(modality_file_path, modality_patient_folder_path)
                   shutil.copy(seg_file_path, modality_patient_folder_path)
                else:
                    print('File not found: {}'.format(modality_file_path))
                    print('File not found: {}'.format(seg_file_path))
            # delete the patient folder
            shutil.rmtree(os.path.join(Test_Dir, pos_neg, patient))

# %%
# Function Defination --> Generating images
def Generate_images():
    try:
        testdir = os.listdir(Test_Dir)
        if '.DS_Store' in testdir:
            testdir.remove('.DS_Store')
            print('Removed .DS_Store from test folder')
        for modality in testdir:
            Modality_path = os.path.join(Test_Dir, modality + '/')
            modalitypath = os.listdir(Modality_path)
            if '.DS_Store' in modalitypath:
                modalitypath.remove('.DS_Store')
                print('Removed .DS_Store from modality folder')
            for patient in modalitypath:
                Patient_path = os.path.join(Modality_path, patient + '/')
                patientpath = os.listdir(Patient_path)
                if '.DS_Store' in patientpath:
                    patientpath.remove('.DS_Store')
                    print('Removed .DS_Store from patient folder')
                for file in patientpath:
                    File_path = os.path.join(Patient_path, file + '/')
                    filepath = os.listdir(File_path)
                    if '.DS_Store' in filepath:
                        filepath.remove('.DS_Store')
                        print('Removed .DS_Store from file folder')
                    os.chdir(File_path)
                    pat = file.split('_')[0]+'_'+file.split('_')[1]

                    list_of_patients = []

                    if pat not in list_of_patients:
                        list_of_patients.append(pat)
                        mod = nib.load('{}_{}.nii.gz'.format(pat, patient))
                        mod_data = mod.get_fdata()
                        seg_mask = nib.load('{}_seg.nii.gz'.format(pat))
                        seg_mask_data = seg_mask.get_fdata()

                        # Extracting layers from mask that have non zero values
                        z = np.any(seg_mask_data, axis=(0,1))
                        zmin, zmax = np.where(z)[0][[0, -1]]  #  zmin & zmax are the first and last layer number non zero values in the z axis

                        # Creating a new mask to remove segmentation
                        d = seg_mask_data
                        for layer in range(zmin, zmax+1):
                             nonzero = np.nonzero(d[:,:,layer])
                             r = nonzero[0]
                             c = nonzero[1]
                             if (r.size == 0) or (c.size == 0):
                                continue
                             rmin = np.min(r)
                             rmax = np.max(r)
                             cmin = np.min(c)
                             cmax = np.max(c)
                             d[rmin:rmax+1, cmin:cmax+1, layer] = 1 #Replacing tumor region values by 1

                        #  Multiply modality data with the new segmentation mask
                        tumor = np.multiply(mod_data, d)

                        # Removing Zero valued layers
                        tumor_layers = tumor[:,:,~(tumor==0).all((0,1))]

                        # Converting to png files
                        Cropped_list = []
                        for lay in range(0, tumor_layers.shape[2]):
                            coords = np.argwhere(tumor_layers[:,:,lay])
                            x_min, y_min = coords.min(axis=0)
                            x_max, y_max = coords.max(axis=0)
                            cropped = tumor_layers[x_min:x_max+1, y_min:y_max+1, lay]
                            cropped *= (255.0/cropped.max()) # Normalizing the values
                            Cropped_list.append(cropped)
                            
                        frame =0
                        for item in Cropped_list:
                            if ((item.shape[0]*item.shape[1])>= 300):
                                frame = frame + 1
                                im = Image.fromarray(item)
                                im = im.convert('L')
                                im.save('{}_{}_{}.png'.format(pat, patient, frame))
                                im.close()
                        
                        # Deleting the nifti files
                        niipath = os.listdir(File_path)
                        if '.DS_Store' in niipath:
                            niipath.remove('.DS_Store')
                            print('Removed .DS_Store from nifti folder')
                        for nii in niipath:
                            try:
                                if nii.startswith(pat) and nii.endswith('.gz'):
                                    os.remove(nii)
                            except Exception as e:
                                print('Error in deleting nifti files')
                                print(e)
    except Exception as e:
        print('Error in Generate_images()')
        print(e)

# %%
# Function Defination --> Interpolation

def Interpolation():
    try:
        testdir = os.listdir(Test_Dir)
        if '.DS_Store' in testdir:
            testdir.remove('.DS_Store')
            print('Removed .DS_Store from test folder')
        for modality in testdir:
            Modality_path = os.path.join(Test_Dir, modality + '/')
            modalitypath = os.listdir(Modality_path)
            if '.DS_Store' in modalitypath:
                modalitypath.remove('.DS_Store')
                print('Removed .DS_Store from modality folder')
            for patient in modalitypath:
                Patient_path = os.path.join(Modality_path, patient + '/')
                patientpath = os.listdir(Patient_path)
                if '.DS_Store' in patientpath:
                    patientpath.remove('.DS_Store')
                    print('Removed .DS_Store from patient folder')
                for file in patientpath:
                    File_path = os.path.join(Patient_path, file + '/')
                    filepath = os.listdir(File_path)
                    if '.DS_Store' in filepath:
                        filepath.remove('.DS_Store')
                        print('Removed .DS_Store from file folder')
                    os.chdir(File_path)

                    
                    pngpath = os.listdir(File_path)
                    if '.DS_Store' in pngpath:
                        pngpath.remove('.DS_Store')
                    for png in pngpath:
                        try:
                            if png.endswith('.png'):
                                image = Image.open(png)
                                image = image.resize(inter_dim, Image.ANTIALIAS)
                                png1 = 'inter_' + png
                                image.save(png1)
                                image.close()
                                os.remove(png)
                        except Exception as e:
                            print('Error in Interpolation() - pngpath')
                            print(e)
    except Exception as e:
        print('Error in Interpolation()')
        print(e)

# %%
# Function Defination --> Creating Patches

def Creating_patches(block_h, block_w, stride):
    try:
        testdir = os.listdir(Test_Dir)
        if '.DS_Store' in testdir:
            testdir.remove('.DS_Store')
            print('Removed .DS_Store from test folder')
        for modality in testdir:
            Modality_path = os.path.join(Test_Dir, modality + '/')
            modalitypath = os.listdir(Modality_path)
            if '.DS_Store' in modalitypath:
                modalitypath.remove('.DS_Store')
                print('Removed .DS_Store from modality folder')
            for patient in modalitypath:
                Patient_path = os.path.join(Modality_path, patient + '/')
                patientpath = os.listdir(Patient_path)
                if '.DS_Store' in patientpath:
                    patientpath.remove('.DS_Store')
                    print('Removed .DS_Store from patient folder')
                for file in patientpath:
                    File_path = os.path.join(Patient_path, file + '/')
                    filepath = os.listdir(File_path)
                    if '.DS_Store' in filepath:
                        filepath.remove('.DS_Store')
                        print('Removed .DS_Store from file folder')
                    
                    # print(File_path)
                    os.chdir(File_path)
                    for png in tqdm(glob.glob('*.png')):
                        img = Image.open(png)
                        img_w, img_h = img.size

                        File_Name, extentions = os.path.splitext(png)

                        Save_path = Modality_path
                        # print('Save',Save_path)

                        frame_num= 0
                        count_row = 0

                        for row in range(0,img_h,stride):
                            if (img_h-row >= block_h):
                                count_row += 1
                                count_col = 0

                                for col in range(0, img_w, stride):
                                    if (img_h - col >= block_w):
                                        count_col += 1
                                        frame_num += 1

                                        box = (col, row, col +
                                               block_w, row+block_h)
                                        a = img.crop(box)
                                        a.save(
                                            Save_path + File_Name + '_row_' + str(count_row) + '_col_' + str(count_col) + '.png')

                        img.close()
                        os.remove(png)

    except Exception as e:
        print('Error in Creating_patches()')
        print(e)

# %%
# Function Defination --> Read images

def read_img():
    print('Reading Images')
    class_num = 0
    data = []
    testdir = os.listdir(Test_Dir)
    if '.DS_Store' in testdir:
        testdir.remove('.DS_Store')
        print('Removed .DS_Store from test folder')
    for pool in testdir:
        pool_dir = Test_Dir + pool + '/'
        pool_dir_list = os.listdir(pool_dir)
        if '.DS_Store' in pool_dir_list:
            pool_dir_list.remove('.DS_Store')
            print('Removed .DS_Store from pool folder')
        for img in tqdm(pool_dir_list):
            try:
                img_array = cv2.imread(os.path.join(pool_dir, img), cv2.IMREAD_GRAYSCALE)
                data.append([img_array, class_num])
            except Exception as e:
                print('Error in read_img()')
                print(e)
        class_num = 1
    return data
        

# %%
# Function Defination --> Initialize features and labels

def Initializing_feature_labels(data):
    print('Initializing Features & Labels')
    X = []
    Y = []

    for features, label in data:
        X.append(features)
        Y.append(label)
    print('List Size: ', len(X), len(Y))
    return X, Y


# %%
# Function Defination --> Reshape the list to numpy array

def Converting(block_h, block_w, X, Y):
    print('Converting to Array')
    global x, y

    # -1 is added to solve dimension mismatch while converting list to an array.
    x = np.array(X).reshape((-1, block_h, block_w, 1))
    y = np.array(Y)

    print('Array Size with Reshape: ', len(X), len(y))
    print('Array Shape with Reshape: ', X.shape, y.shape)

# Function Definition --> Reshape the list to numpy array


# def Converting(block_h, block_w, X, Y):
#     print('Converting to Array')
#     global x, y

#     # Check if number of elements is divisible by block size
#     n_elem = len(X)
#     prod = block_h * block_w
#     if n_elem % prod != 0:
#         n_elem_trunc = n_elem // prod * prod
#         X = X[:n_elem_trunc]
#         Y = Y[:n_elem_trunc]
#         print(
#             f"Truncating arrays to {n_elem_trunc} elements to ensure divisibility by block size")

    # # Reshape arrays
    # x = np.array(X).reshape((-1, block_h, block_w, 1))
    # y = np.array(Y)

    # print('Array Size with Reshape: ', len(X), len(y))
    # print('Array Shape with Reshape: ', x.shape, y.shape)


# %%
# Function Defination --> Creating Pickle files

def create_pickle_files():
    # list for storing preprocessing data
    data = []

    # Read Images
    data = read_img()

    # Print size of data
    print('Size of data: ', len(data))

    # Initializing all features & labels of the processed image in the list X & Y
    X = []
    Y = []

    # Initializing the features and labels
    X,Y = Initializing_feature_labels(data)

    # Converting the list into numpy array
    Converting(block_h, block_w, X, Y)

    # Storing the numpy array in pickle file
    pickle_out = open(Test_Dir + "X_test.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open(Test_Dir + "y_test.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

# %%
# Function defination --> Create Testing files

def Test_file_creation():

    # # Create backup folder
    # print('Creating Backup Folder')
    # Backup()
    
    # # Create modality folders
    # print('Creating Modality Folders')
    # create_modality_folders()

    # # Generate Images 
    # print('Generating Images')
    # Generate_images()

    # # Interpolation of images
    # print('Interpolating Images')
    # Interpolation()

    # # Create patches
    # print('Creating Patches')
    # Creating_patches(block_h, block_w, stride)

    # Create pickle files
    print('Creating Pickle Files')  
    create_pickle_files()

# %%
# Function Defination --> Load pickle files

def Load_Data():
    pickle_in = open(Test_Dir + "X_test.pickle", "rb")
    X = pickle.load(pickle_in) # Features

    pickle_in = open(Test_Dir + "y_test.pickle", "rb")
    y = pickle.load(pickle_in) # Labels

    return X, y

# %%
# Function Defination --> Load Model

def Load_Model():
    model = tf.keras.models.load_model(path + 'Outputs/' + load_model )
    return model

# %%
# Function Defination --> PLotting AUC-ROC Curve

def Plotting_AUC_ROC_Curve(X,y,model):
    # X = X.astype('float32')
    # X = torch.from_numpy(X)
    probs = model.predict(X)
    y_preds = np.argmax(model.predict(X), axis=-1)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, color='red')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--', color='blue')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.show()
    acc = metrics.accuracy_score(y, y_preds)
    print('Accuracy: ', acc)
    print('AUC: ', roc_auc)

# %%
# Main cell to execute the functions
# # Reconstruction of folders
# Reconstruct()

# # Test file creation for computation on independant cohort
Test_file_creation()

# Loading data
X, y = Load_Data()

# Print lenght of data
print('Length of data: ', len(X), len(y))

results = []

# Load model
model = Load_Model()

# Plotting the AUC-ROC curve
Plotting_AUC_ROC_Curve(X,y,model)



# %%
print(type(X))
print(type(y))
print(X.shape)
print(y.shape)
