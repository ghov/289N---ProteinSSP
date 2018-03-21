import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing

from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import losses
from keras.models import h5py
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import optimizers
import tensorflow as tf
from keras import backend as K

from keras.models import load_model

import matplotlib.pyplot as plt


data_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/Data/cullpdb+profile_6133.npy.gz'
this_directory_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/saved_models/'
model_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/saved_models/fc_model_44_params_val_0.022727562233805655.h5'

number_of_labels = 9
number_of_features = 22
reshaped_imput_dimension = (700, 44)
numer_of_examples = 6133
vectorized_dimensions = 39900

all_data_np = np.load(data_file_path)

# reshape
reshaped_data = np.reshape(all_data_np,(6133,700,57))

print(reshaped_data.shape)

X_Train = np.concatenate((reshaped_data[:,:,0:22],reshaped_data[:,:,35:57]), axis=2)
Y_Train = reshaped_data[:, :, 22:31]

np.set_printoptions(threshold=np.inf)

model = load_model(model_file_path)
model.summary()

print('Training')

temp_list = list()

print(X_Train.shape)
print(Y_Train.shape)

my_history = model.fit(X_Train[0:5000, :, :], Y_Train[0:5000, :, :], epochs=200, verbose=1, batch_size=256, validation_split=0.2)

# save the model
loss_history = np.array(my_history.history['loss'])
model.save(this_directory_path + 'fc_model_44_params_val_' + str(np.amin(loss_history)) +'.h5')

#print(loss_history.shape)
# epoch_history = np.zeros((1000,))
# for i in range(0,epoch_history.shape[0]):
#     epoch_history[i] = i+1

#plt.plot(epoch_history, loss_history)

predictions = model.predict(X_Train[:, :, :])

print(predictions[0][0].shape)

print('the lowest loss is: ' + str(np.amin(loss_history)))
