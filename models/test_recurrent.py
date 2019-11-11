import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

test_x = np.load('../labeled_data/recurrent/test_x.npy')
test_y = np.load('../labeled_data/recurrent/test_y.npy')

temp = []
for i in range(len(test_x)):
    temp.append(test_x[i][:20][1::2, 1::2])  # take first 10 frames, take only even rows and cols of the image to reduce size
test_x = temp

reshape_y = np.shape(test_x)[1] * np.shape(test_x)[2]
test_x = np.reshape(test_x, (len(test_x), reshape_y))

model = tf.keras.models.load_model('recurrent.h5')

# model.summary()

print(np.shape(test_x))

print('predictions...')
preds = model.predict(test_x)
# preds = np.load('../predictions/recurrent_pred.npy')
preds = np.argmax(preds, axis=1)

# plt.plot(test_y, 'b')
# plt.plot(preds, 'r')
# plt.show()
# plt.savefig('../plots/recurrent_performance')

print(test_y)
print(preds)

# np.save('../predictions/recurrent_pred', preds)

