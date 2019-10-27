from MyResnet import myModel
import numpy as np
import os,glob
from DataGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

base_path = './Data'

x_train_path = os.path.join(base_path,'X_Train')
y_train_path = os.path.join(base_path,'Y_Train')
x_valid_path = os.path.join(base_path,'X_Valid')
y_valid_path = os.path.join(base_path,'Y_Valid')

x_train_data = sorted(glob.glob(os.path.join(x_train_path,'*.npy')))
y_train_data = sorted(glob.glob(os.path.join(y_train_path,'*.npy')))
x_valid_data = sorted(glob.glob(os.path.join(x_valid_path,'*.npy')))
y_valid_data = sorted(glob.glob(os.path.join(y_valid_path,'*.npy')))

train_datas = DataGenerator(input_x = x_train_data,output_y = y_train_data, batch_size=64, dim=(128,128),n_channels=3,shuffle=True)
valid_datas = DataGenerator(input_x = x_valid_data,output_y = y_valid_data, batch_size=64, dim=(128,128),n_channels=3,shuffle=True)

X,y = valid_datas[0]

print(y)

def train_age() :
	model = myModel()
	model.compile(optimizer='adam', loss='mse')
	model.summary()
	history = model.fit_generator(train_datas,validation_data=valid_datas, epochs = 50, verbose=1, callbacks=[ModelCheckpoint('models/model.h5',monitor='val_loss',verbose=1,save_best_only=True)])

train_age()	





