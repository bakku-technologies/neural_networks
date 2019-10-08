import numpy as np
import sys
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def create_cnn(width, height, depth, filters=(16, 21, 64), regress=False):
    input_shape = (height, width, depth)
    chan_dim = -1

    inputs = Input(shape=input_shape)

    # loop over the number of filters
    for (i, filter) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(filter, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


ang_train = np.load('labeled_data/ang_train.npy')
us_train = np.load('labeled_data/us_train.npy').flatten()  # .8 gigabyte
us_test = np.load('labeled_data/us_test.npy').flatten()
ang_test = np.load('labeled_data/ang_test.npy')

us_train = np.reshape(us_train, [-1, 310, 128, 1])
us_test = np.reshape(us_test, [-1, 310, 128, 1])

model = create_cnn(128, 310, 1, regress=True)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print('training model...')
model.fit(us_train, ang_train, validation_split=0.3, epochs=10, batch_size=8, verbose=2)

print('predictions...')
preds = model.predict(us_test)

plt.plot(ang_test, 'b')
plt.plot(preds, 'r')
# plt.savefig('cnn_acc.png')
plt.show()
