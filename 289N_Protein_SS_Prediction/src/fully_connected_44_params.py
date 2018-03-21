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

import matplotlib.pyplot as plt


data_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/Data/cullpdb+profile_6133.npy.gz'
this_directory_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/saved_models/'

def customLoss(y_true, y_pred):
    error_counter = 0
    total_counter = 0
    #print(K.get_value(y_pred))
    # sess = tf.InteractiveSession()
    #
    # with sess.as_default():
    #     print(len(yTrue.eval()))
    # for i in range(int(yTrue.shape[0])):
    #     for j in range(700):
    #         if (np.argmax(yTrue[i][j][:]) != 8):
    #             total_counter += 1
    #             if (np.argmax(yPred[i][j][:]) != np.argmax(yTrue[i][j][:])):
    #                 error_counter += 1

    #for val in y_pred:
        #print(val)
    #print(K.argmin(y_pred, axis=-1))
    print('the loss is: ' + str(K.mean(K.square(y_pred - y_true), axis=-1)))
    return K.mean(K.square(y_pred - y_true), axis=-1)
    #return error_counter/total_counter

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

# Get the training set
#traning_set = all_data_np[0:5600]
#validation_set = all_data_np[5600:5877]
#testing_set = all_data_np[5877:6133]

np.set_printoptions(threshold=np.inf)

model = Sequential()
model.add(Dense(44, input_shape = (reshaped_imput_dimension), activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(number_of_labels))
#model.add(Dense(number_of_labels, activation='softmax'))

adam = optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.999, decay=0)

model.compile(loss='mse', optimizer=adam)
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
