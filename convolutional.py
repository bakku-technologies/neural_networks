import numpy as np
import sys
import os

from tensorflow.keras.models import Model, Sequential
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
        x = Dense(4, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


print('loading data...')

ang_train_all = {dir_content: np.load(dir_content)
                 for dir_content in os.listdir('data_2_5_2020/new_train_data/')
                 if dir_content.split('.')[-1] in ['npy']
                 if dir_content.endswith('-angles.npy')}

us_train_all = {dir_content: np.load(dir_content)
                for dir_content in os.listdir('data_2_5_2020/new_train_data/')
                if dir_content.split('.')[-1] in ['npy']
                if dir_content.endswith('-images.npy')}

# ang_train = np.load('labeled_data/fist_and_relax/four_fingers/ang_train.npy')
# us_train = np.load('labeled_data/fist_and_relax/four_fingers/us_train.npy').flatten()  # .8 gigabyte
# us_train = np.reshape(us_train, [-1, 310, 128, 1])

# pinch_ang = np.load('labeled_data/pinch_relax/four_fingers/ang_train.npy')
# pinch_us = np.load('labeled_data/pinch_relax/four_fingers/us_train.npy').flatten()  # .8 gigabyte
# pinch_us = np.reshape(pinch_us, [-1, 310, 128, 1])

print('creating model...')
model = create_cnn(128, 310, 1, regress=True)

opt = Adam(lr=1e-2, decay=1e-3 / 200)  # base decay rate is 1e-3
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print('training model...')
model.fit(us_train, ang_train, validation_split=0.3, epochs=20, batch_size=32,
          verbose=2)  # base epochs=10, batch_size=8 (best: epochs=20, batch_size=20)

model.save('models/05022020/05022020_convolutional.h5')
