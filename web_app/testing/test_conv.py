import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

matplotlib.use('Agg')


def test_cnn(data_type):
    test_x = []
    test_y = []
    title = ""
    if data_type == 'open_close_test':
        test_x = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/fist_and_relax/us_test.npy').flatten()
        test_y = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/fist_and_relax/ang_test.npy')
        title = 'CNN: Open and Close Test Data:'
    elif data_type == 'open_close_train':
        test_x = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/fist_and_relax/us_train.npy').flatten()
        test_y = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/fist_and_relax/ang_train.npy')
        title = 'CNN: Open and Close Train Data:'
    elif data_type == 'pinch_relax_test':
        test_x = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/pinch_relax/us_test.npy').flatten()
        test_y = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/pinch_relax/ang_test.npy')
        title = 'CNN: Pinch and Relax Test Data:'
    elif data_type == 'pinch_relax_train':
        test_x = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/pinch_relax/us_train.npy').flatten()
        test_y = np.load('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/labeled_data/pinch_relax/ang_train.npy')
        title = 'CNN: Pinch and Relax Train Data:'

    test_x = np.reshape(test_x, [-1, 310, 128, 1])

    model = tf.keras.models.load_model('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/testing/models/open_close_convolutional.h5')
    # model = tf.keras.models.load_model('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/testing/models/both_datasets_convolutional.h5')
    # model = tf.keras.models.load_model('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/testing/models/pinch_relax_convolutional.h5')

    # model.summary()

    # print(np.shape(test_x))

    print('predictions...')
    preds = model.predict(test_x)
    # print(preds)

    # address ending error:
    for i in range(len(preds)):
        if preds[i] < 0 or preds[i] > 2:
            preds[i] = 0.25

    plt.clf()
    plt.plot(test_y, 'b')
    plt.plot(preds, 'r')
    # plt.show()
    plt.savefig('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/static/cnn_performance')

    cnn_mse = mean_squared_error(test_y, preds)
    score = 'CNN Mean Squared Error: ' + str(cnn_mse)

    return score, title

