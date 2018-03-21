import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from keras.models import load_model




model_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/model1_0.024.h5'
data_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/Data/cullpdb+profile_6133.npy.gz'

my_data = np.load(data_file_path)
my_model = load_model(model_file_path)

reshaped_data = np.reshape(my_data, (6133,700,57))
X_Train = reshaped_data[:,:,0:22]
Y_Train = reshaped_data[:,:,22:31]

predictions  = my_model.predict(X_Train[:,:,:])

protein_error_list = list()
protein_error_counter = 0
for i in range(6133):
    aa_error_counter = 0
    protein_error_list.append(list())
    for j in range(700):
        if(np.argmax(predictions[i][j][:]) == np.argmax(Y_Train[i][j][:])):
            protein_error_list[i].append(1)            


total_error = 0
for val in protein_error_list:
    total_error += len(val)

print(total_error)
