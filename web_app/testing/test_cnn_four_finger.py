import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches

matplotlib.use('Agg')


def plot_results(preds, test_y, index, finger, file_name):
    preds[-1, index] = 0.25
    preds[-2, index] = 0.25

    plt.title(finger)
    plt.axis(ymin=0, ymax=2)
    plt.plot(test_y[:, index], 'b')
    plt.plot(preds[:, index], 'r')
    blue_patch = mpatches.Patch(color='blue', label='Actual')
    red_patch = mpatches.Patch(color='red', label='Predicted')
    plt.legend(handles=[blue_patch, red_patch])
    plt.savefig(file_name)
    plt.clf()


def predict(preds, test_y, plot_loc):
    plot_results(preds, test_y, 0, 'Index Finger', (plot_loc + '/index'))
    plot_results(preds, test_y, 1, 'Middle Finger', (plot_loc + '/middle'))
    plot_results(preds, test_y, 2, 'Ring Finger', (plot_loc + '/ring'))
    plot_results(preds, test_y, 3, 'Pinky Finger', (plot_loc + '/pinky'))

    index_mse = mean_squared_error(test_y[:, 0], preds[:, 0])
    middle_mse = mean_squared_error(test_y[:, 1], preds[:, 1])
    ring_mse = mean_squared_error(test_y[:, 2], preds[:, 2])
    pinky_mse = mean_squared_error(test_y[:, 3], preds[:, 3])

    print('Index Mean Squared Error: ' + str(index_mse))
    print('Middle Mean Squared Error: ' + str(middle_mse))
    print('Ring Mean Squared Error: ' + str(ring_mse))
    print('Pinky Mean Squared Error: ' + str(pinky_mse))

    return [index_mse, middle_mse, ring_mse, pinky_mse]


def test_cnn_four_finger(data_type, model_type):
    test_x = []
    test_y = []
    title = ""
    if data_type == 'open_close_test':
        test_x = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/fist_relax/four_fingers/us_test.npy').flatten()
        test_y = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/fist_relax/four_fingers/ang_test.npy')
        title = 'CNN: Open and Close Test Data:'
    elif data_type == 'open_close_train':
        test_x = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/fist_relax/four_fingers/us_train.npy').flatten()
        test_y = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/fist_relax/four_fingers/ang_train.npy')
        title = 'CNN: Open and Close Train Data:'
    elif data_type == 'pinch_relax_test':
        test_x = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/pinch_relax/four_fingers/us_test.npy').flatten()
        test_y = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/pinch_relax/four_fingers/ang_test.npy')
        title = 'CNN: Pinch and Relax Test Data:'
    elif data_type == 'pinch_relax_train':
        test_x = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/pinch_relax/four_fingers/us_train.npy').flatten()
        test_y = np.load('/home/mitch/Documents/ultrasound_neural_networks/labeled_data/pinch_relax/four_fingers/ang_train.npy')
        title = 'CNN: Pinch and Relax Train Data:'

    test_x = np.reshape(test_x, [-1, 310, 128, 1])

    # model = None
    if model_type == 'fr':
        model = tf.keras.models.load_model('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/testing/models/four_finger/fist_relax_convolutional.h5')
    elif model_type == 'pr':
        model = tf.keras.models.load_model('/home/mitch/Documents/ultrasound_neural_networks/web_app/testing/models/four_finger/pinch_relax_convolutional.h5')
    elif model_type == 'both':
        model = tf.keras.models.load_model('/home/mitch/Documents/ultrasound_neural_networks/web_app/testing/models/four_finger/both_datasets_convolutional.h5')

    print('predictions...')
    preds = model.predict(test_x)

    errors = predict(preds, test_y, '/home/mitch/Documents/ultrasound_neural_networks/web_app/static/four_finger')

    return errors, title

