# #  Import libraries and define variables
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from tensorflow.keras.metrics import TruePositives,TrueNegatives,FalsePositives,FalseNegatives,AUC,Recall,Precision
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import pickle as pkl
import config
import time
from multiprocessing import Pool,Process
import pandas as pd
import numpy as np
from tqdm import tqdm
# import h5py





# Variables intialisation
(img_h,img_w)=(32,32)
epoch=1

# Function Definition --> Reads the cv index split data
def load_global_data():
    global cv_idx_dict,train_df
    cv_idx_dict = pkl.load(open(config.MAIN_DIR+'results/cross_validation_indexes.pkl','rb'))
    train_df = pd.read_csv(config.MAIN_DIR+'results/train_data.csv')
    print("Global data was loaded")

def making_all_arrays_to_list(idx: list,x: list,y: list):
    for row in tqdm(idx[:2]):
        if train_df['mgmt'].iloc[row]==0:
            label = 'MGMT_negative/'
        else: label = 'MGMT_positive/'
        pkl_file = pkl.load(open(config.MAIN_DIR+'preprocessed/32x32/'+label+train_df['id'].iloc[row],'rb'))
        for arr in pkl_file:
            x.append(arr)
            y.append(train_df['mgmt'].iloc[row])
    # x=np.array(x)
    # y=np.array(y)

def data_arr(indexes: list):
    train_idx,val_idx = indexes
    train_x, train_y = [], []
    val_x, val_y = [], []
    making_all_arrays_to_list(idx = train_idx,x = train_x,y = train_y)
    making_all_arrays_to_list(idx = val_idx,x = val_x,y = val_y)

    return train_x, train_y, val_x, val_y
    



#  Funtion Defination --> train the model and stores the history
def model_training(data_idx: list):
    cv, cv_indexes = data_idx

    # Model intialisation
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))

    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))

    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))

    model.add(Flatten())  # Convert 3D feature map to 1D feature vector.

    model.add(Dense(1096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy',TruePositives(),TrueNegatives(),FalsePositives(),FalseNegatives(),AUC(),Recall(),
                                             Precision()])

    # Selecting the data from train_idx and test_idx
    X_train, y_train, X_val, y_val = data_arr(indexes = cv_indexes)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)

    # Model Checkpoints
    checkpoint_filepath = config.MAIN_DIR+f'results/model checkpoints/model(k={cv+1})'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath+'_epoch-{epoch:02d}_valloss-{val_loss:.4f}.h5',
                                                                   monitor='val_loss', 
                                                                   mode='min',
                                                                   save_best_only=True)
    # Reduce LRPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.0001)

    # Early stoping
    early_stoping =EarlyStopping(monitor="val_loss",patience=5,mode="min") 
    # Model training
    print(f"Model Training for {cv} was started")
    history = model.fit(X_train[:100], y_train[:100], batch_size=8, epochs=epoch,
                        validation_data=(X_val[:100], y_val[:100]), shuffle=True,callbacks=[model_checkpoint_callback,reduce_lr,early_stoping])
    
    # Stores the history in pickle
    pkl.dump(history.history,open(config.MAIN_DIR+f'results/history/history_k={cv}.pkl','wb'))
    print(f"Model Training for cv-{cv} was completed")

    # del train_idx,val_idx,X_train,X_val,y_train,y_val
    




if __name__=="__main__":
    start=time.time()
    load_global_data()
    pool = Pool(processes=2)
    args=[]
    for cv in cv_idx_dict:
        # model_training([cv,cv_idx_dict[cv]])
        args.append([cv,cv_idx_dict[cv]])
    pool.map(model_training,args)
    pool.close()
    pool.join()
    end=time.time()
    print(f'Total Time taken by cross validation - {end-start}')
# del X,y,cv_idx_dict

# # global l 
# def func1():
#     global l 
#     l = [j for j in range(2000)]
#     print(f"Address {id(l)}")
# def func(l):
#     x,a=l
#     # l = [j for j in range(2)]
#     # print(f"Address {id(l)}")
#     print(a[x])
#     print(f"For {x} Address {id(a)}")
# # func1()

# if __name__=="__main__":
#     start=time.time()
#     # l = [j for j in range(2000)]
#     func1()
#     print('Loaded')
#     p=Pool(processes=9)
#     p.map(func,[[6,l],[0,l],[9,l],[16,l],[10,l],[19,l],[26,l],[20,l],[29,l]])
#     p.close()
#     p.join()
#     end=time.time()
#     print(f'Total Time taken by cross validation - {end-start}')