import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import to_categorical

ang_train = np.load('labeled_data/ang_train.npy')
ang_test = np.load('labeled_data/ang_test.npy')
us_train = np.load('labeled_data/us_train.npy')
us_test = np.load('labeled_data/us_test.npy')

# us_train = np.array(us_train).flatten()
# us_test = np.array(us_test).flatten()

# shape_train = us_train.shape
# shape_test = us_test.shape

ang_train = to_categorical(ang_train, 4776)
# TODO: Create network with: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=2583,
              input_length=39680,
              output_dim=4776,
              # weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Input Layer
# model.add(Dense(39680, activation='relu', input_shape=(39680, 2583)))  # input layer

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False,
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

angles_len = 4776  # from 'store_data'
# Output layer
model.add(Dense(angles_len, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=us_train, y=ang_train, validation_split=0.3, epochs=10, batch_size=128, verbose=2)
