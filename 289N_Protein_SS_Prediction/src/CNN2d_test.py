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
import numpy as np

this_directory_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/saved_models/'
data_file_path = '/home/greghovhannisyan/PycharmProjects/289N_Protein_SS_Prediction/Data/cullpdb+profile_6133.npy.gz'

number_of_labels = 9
number_of_features = 22
reshaped_input_dimension = (700, 22, 1)
numer_of_examples = 6133
vectorized_dimensions = 39900

all_data_np = np.load(data_file_path)

# reshape
reshaped_data = np.reshape(all_data_np,(6133,700,57,1))

print(reshaped_data.shape)

X_Train = reshaped_data[:, :, 0:22, :]
Y_Train = reshaped_data[:, :, 22:31, :]

np.set_printoptions(threshold=np.inf)

model = Sequential()

# Need to calculate the number of layers needed to get to 3 by 3 by 700 filters

model.add(Conv2D(filters=700, kernel_size=(698,20), strides=1, activation='relu', input_shape=(reshaped_input_dimension)))
#model.add(Conv2D(filters=700, kernel_size=(299,9), strides=1, activation='relu'))
#model.add(Dropout(rate=0.6))
model.add(Reshape((700,9,1)))
#model.add(Activation(activation=K.softmax(axis=-1)))

model.summary()

model.compile(loss='mse', optimizer='adam')
#model.summary()

print('Training')

temp_list = list()

print(X_Train.shape)
print(Y_Train.shape)

my_history = model.fit(X_Train[:, :, :, :], Y_Train[:, :, :, :], epochs=1000, verbose=1, batch_size=256, validation_split=0.2)

loss_history = np.array(my_history.history['loss'])

model.save(this_directory_path + 'cnn2d_model_val_' + str(np.amin(loss_history)) +'.h5')

predictions = model.predict(X_Train[:, :, :])

print(predictions[0][0].shape)

print('the lowest loss is: ' + str(np.amin(loss_history)))