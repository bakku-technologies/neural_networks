import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

test_x = np.load('../labeled_data/us_test.npy').flatten()
test_y = np.load('../labeled_data/ang_test.npy')

test_x = np.reshape(test_x, [-1, 310, 128, 1])

model = tf.keras.models.load_model('open_close_convolutional.h5')

# model.summary()

# print(np.shape(test_x))

print('predictions...')
preds = model.predict(test_x)
# print(preds)

# plt.plot(test_y, 'b')
# plt.plot(preds, 'r')
# # plt.show()
# plt.savefig('../plots/cnn_train_performance')

cnn_mse = mean_squared_error(test_y, preds)
print('CNN Mean Squared Error: ' + str(cnn_mse))

# print(test_y)
# print(preds)
