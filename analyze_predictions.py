import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

test_x = np.load('labeled_data/fist_and_relax/us_test.npy')
test_x = np.reshape(test_x, [-1, 310, 128, 1])
test_y = np.load('labeled_data/fist_and_relax/ang_test.npy')

model = tf.keras.models.load_model('models/open_close_convolutional.h5')
pred = model.predict(test_x)

cnn_mse = mean_squared_error(test_y, pred)
for i in range(len(pred)):
    if pred[i] < 0 or pred[i] > 2:
        pred[i] = 0.25

plt.plot(test_y, 'b')
plt.plot(pred, 'r')
# plt.plot(svm_pred, 'g')
plt.show()
# plt.savefig('plots/cnn_train_performance')

# svm_mse = mean_squared_error(test_y.npy, svm_pred)
# print('SVM Mean Squared Error: ' + str(svm_mse))
print('CNN Mean Squared Error: ' + str(cnn_mse))


# def plot_results(index, finger, file_name):
#
#     preds[-1, index] = 0.25
#     preds[-2, index] = 0.25
#
#     plt.title(finger)
#     plt.axis(ymin=0, ymax=2)
#     plt.plot(test_y[:, index], 'b')
#     plt.plot(preds[:, index], 'r')
#     blue_patch = mpatches.Patch(color='blue', label='Actual')
#     red_patch = mpatches.Patch(color='red', label='Predicted')
#     plt.legend(handles=[blue_patch, red_patch])
#     plt.savefig(file_name)
#     plt.clf()
#
#
# def predict(preds, test_y, plot_loc):
#     plot_results(0, 'Index Finger', (plot_loc + '/index'))
#     plot_results(1, 'Middle Finger', (plot_loc + '/middle'))
#     plot_results(2, 'Ring Finger', (plot_loc + '/ring'))
#     plot_results(3, 'Pinky Finger', (plot_loc + '/pinky'))
#
#     index_mse = mean_squared_error(test_y[:, 0], preds[:, 0])
#     middle_mse = mean_squared_error(test_y[:, 1], preds[:, 1])
#     ring_mse = mean_squared_error(test_y[:, 2], preds[:, 2])
#     pinky_mse = mean_squared_error(test_y[:, 3], preds[:, 3])
#
#     print('Index Mean Squared Error: ' + str(index_mse))
#     print('Middle Mean Squared Error: ' + str(middle_mse))
#     print('Ring Mean Squared Error: ' + str(ring_mse))
#     print('Pinky Mean Squared Error: ' + str(pinky_mse))
#
#
# print('loading data...')
# test_x = np.load('labeled_data/fist_and_relax/four_fingers/us_test.npy')
# test_y = np.load('labeled_data/fist_and_relax/four_fingers/ang_test.npy')
# test_x = np.reshape(test_x, [-1, 310, 128, 1])
#
# print('making predictions...')
# model = tf.keras.models.load_model('models/four_fingers/both_datasets_convolutional.h5')
# preds = model.predict(test_x)
#
# print('fist and relax...')
# predict(preds, test_y, 'plots/four_finger/fist_relax')
#
#
# print('loading data...')
# test_x = np.load('labeled_data/pinch_relax/four_fingers/us_test.npy')
# test_y = np.load('labeled_data/pinch_relax/four_fingers/ang_test.npy')
# test_x = np.reshape(test_x, [-1, 310, 128, 1])
#
# print('making predictions...')
# preds = model.predict(test_x)
#
# print('pinch and relax...')
# predict(preds, test_y, 'plots/four_finger/pinch_relax')
