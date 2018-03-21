from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import MaxPool1D
from keras.layers import MaxPool2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras import backend as K
from keras import optimizers
import numpy as np

this_directory_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/saved_models/'
data_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/Data/cullpdb+profile_6133.npy.gz'

number_of_labels = 9
number_of_features = 22
reshaped_input_dimension = (700, 44)
numer_of_examples = 6133
vectorized_dimensions = 39900

all_data_np = np.load(data_file_path)

# reshape
reshaped_data = np.reshape(all_data_np,(6133,700,57))

print(reshaped_data.shape)

X_Train = np.concatenate((reshaped_data[:,:,0:22],reshaped_data[:,:,35:57]), axis=2)
Y_Train = reshaped_data[:, :, 22:31]

np.set_printoptions(threshold=np.inf)

model = Sequential()

model.add(Conv1D(filters=3, kernel_size=1, strides=1, activation='relu', input_shape=(reshaped_input_dimension)))
model.add(Conv1D(filters=6, kernel_size=1, strides=1, activation='relu'))
model.add(Conv1D(filters=9, kernel_size=1, strides=1, activation='relu'))

model.summary()

adam = optimizers.Adam(lr=0.001, beta_1=0, beta_2=0, decay=0)

model.compile(loss='mse', optimizer=adam)
#model.summary()

print('Training')

temp_list = list()

print(X_Train.shape)
print(Y_Train.shape)

my_history = model.fit(X_Train[0:5000, :, :], Y_Train[0:5000, :, :], epochs=200, verbose=1, batch_size=256, validation_split=0.2)

loss_history = np.array(my_history.history['loss'])

model.save(this_directory_path + 'cnn1d_model_44_params_val_' + str(np.amin(loss_history)) +'.h5')

print('the lowest loss is: ' + str(np.amin(loss_history)))