import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import to_categorical
from scipy.signal import argrelextrema

# ang_train = np.load('labeled_data/ang_train.npy')
# ang_test = np.load('labeled_data/ang_test.npy')
# us_train = np.load('labeled_data/us_train.npy')
# us_test = np.load('labeled_data/us_test.npy')
#
#
#
# # Calculate the maximum values (return matrix, each row is start of maximum and then end of maximum)
# def calc_maxima(data):
#     all_max = argrelextrema(data, np.greater)[0]
#     # print(all_max)
#
#     ranges = []
#     prev = data[all_max[0]]
#     prev_ind = 0
#     maxima = []
#     for i in all_max:
#         if len(maxima) == 1:
#             if data[maxima[0]] - 0.5 > data[i]:
#                 maxima.append(prev_ind)
#                 ranges.append(maxima)
#                 maxima = []
#         elif data[i] > prev + 0.5:
#             maxima.append(i)
#         prev = data[i]
#         prev_ind = i
#
#     return ranges
#
#
# # calculate the minimum values (return matrix, each row is start of minimum and then end of minimum
# def calc_minima(data):
#     all_min = argrelextrema(data, np.less)[0]
#     # print(all_max)
#
#     ranges = []
#     prev = data[all_min[0]]
#     prev_ind = 0
#     minima = []
#     for i in all_min:
#         if len(minima) == 1:
#             if data[minima[0]] + 0.5 < data[i]:
#                 minima.append(prev_ind)
#                 ranges.append(minima)
#                 minima = []
#         elif data[i] < prev - 0.5:
#             minima.append(i)
#         prev = data[i]
#         prev_ind = i
#
#     return ranges
#
#
# test_open_indices = calc_maxima(ang_test)
# train_open_indices = calc_maxima(ang_train)
# test_close_indices = calc_minima(ang_test)
# train_close_indices = calc_minima(ang_train)
#
# # Merged Test data starting with opening hand coordinates
# merged_test = [None]*(len(test_open_indices)+len(test_close_indices))
# merged_test[::2] = test_open_indices
# merged_test[1::2] = test_close_indices
#
# # Merged training data starting with open hand coordinates
# merged_train = [None]*(len(train_open_indices)+len(train_close_indices))
# merged_train[::2] = train_close_indices
# merged_train[1::2] = train_open_indices
#
# OPENING = 0
# CLOSING = 1
# OPEN = 2
# CLOSED = 3
#
# # NOTE: training data starts with opening whereas test data starts with closed #
# test_x = [us_test[:180]]
# test_y = [CLOSED]
# train_x = []
# train_y = []
#
#
# prev = [0, 180]
# for i in range(len(merged_test)):
#     cur = merged_test[i]
#     if i % 2 == 0:
#         test_y.append(OPENING)
#         test_x.append(us_test[prev[1]: cur[0]])
#         test_y.append(OPEN)
#         test_x.append(us_test[cur[0]: cur[1]])
#     else:
#         prev = merged_test[i-1]
#         test_y.append(CLOSING)
#         test_x.append(us_test[prev[1]: cur[0]])
#         test_y.append(CLOSED)
#         test_x.append(us_test[cur[0]: cur[1]])
#         prev = merged_test[i]
# test_y.append(CLOSING)
# test_x.append(us_test[cur[1]:950])
# test_y.append(CLOSED)
# test_x.append(us_test[950:])
#
# prev = [0,0]
# for i in range(len(merged_train)):
#     cur = merged_train[i]
#     if i % 2 == 0:
#         train_y.append(OPENING)
#         train_x.append(us_train[prev[1]: cur[0]])
#         train_y.append(OPEN)
#         train_x.append(us_train[cur[0]: cur[1]])
#     else:
#         prev = merged_train[i-1]
#         train_y.append(CLOSING)
#         train_x.append(us_train[prev[1]: cur[0]])
#         train_y.append(CLOSED)
#         train_x.append(us_train[cur[0]: cur[1]])
#         prev = merged_train[i]
# train_y.append(CLOSING)
# train_x.append(us_train[cur[1]:2500])
# train_y.append(CLOSED)
# train_x.append(us_train[2500:])
#
# np.save('labeled_data/recurrent/train_x', train_x)
# np.save('labeled_data/recurrent/train_y', train_y)
# np.save('labeled_data/recurrent/test_x', test_x)
# np.save('labeled_data/recurrent/test_y', test_y)

train_x = np.load('labeled_data/recurrent/train_x.npy')
train_y = np.load('labeled_data/recurrent/train_y.npy')
test_x = np.load('labeled_data/recurrent/test_x.npy')
test_y = np.load('labeled_data/recurrent/test_y.npy')

temp = []
for i in range(len(train_x)):
    temp.append(train_x[i][:80][1::3, 1::3])  # take first 10 frames, take only even rows and cols of the image to reduce size
train_x = temp

print(np.shape(train_x))

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

reshape_y = np.shape(train_x)[1] * np.shape(train_x)[2]
train_x = np.reshape(train_x, (len(train_x), reshape_y))

print(np.shape(train_x))

model = tf.keras.Sequential()

# embedding layer
model.add(layers.Embedding(len(train_x), 4, input_length=reshape_y))

model.add(layers.SpatialDropout1D(0.2))

model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=5, batch_size=64, validation_split=0.1)

model.save('models/recurrent.h5')

