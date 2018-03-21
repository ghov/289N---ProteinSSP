from keras.models import load_model
import numpy as np
import csv


number_of_examples = 100
report_number_list = list()

Fc_model = load_model(model_save_path + 'full_model_new_no_floats2_val_0.04737289249897003.h5')

X_train = np.load(numpy_array_save_path + 'X_train_new_no_floats.npy')
Y_train = np.load(numpy_array_save_path + 'Y_train_new_no_floats.npy')

predictions = Fc_model.predict(X_train.transpose()[:,:])

# Read the reportnumbers through the csv file
with open(no_floats_path) as csv_read_file:
    my_reader = csv.DictReader(csv_read_file, delimiter=",")
    for row in my_reader:
        report_number_list.append(row['REPORTNUMBER'])

# Write to the csv file
with open(prediction_file_path, 'a') as csv_write_file:
    fieldnames = ['REPORTNUMBER', 'PREDICTED PM', 'TRUE PM', 'DIFFERENCE']
    my_writer = csv.DictWriter(csv_write_file, fieldnames=fieldnames)

    for i in range(predictions.shape[0]):
        my_writer.writerow({'REPORTNUMBER' : report_number_list[i],
                            'PREDICTED PM' : predictions[i][0],
                            'TRUE PM' : Y_train[i:i+1,:1],
                            'DIFFERENCE' : abs(predictions[i][0] - Y_train[i:i+1,:1])
        })

#
# for i in range(vals.shape[0]):
#     print(vals[i][0])
#     print(Y_train[i:i+1,:1])
# print('\n')
#print(Y_train[:56, :1])

#plt.plot(epoch_history, loss_history)
#plt.show()





