import tensorflow as tf
import numpy as np


def predict_from_image(image):
    image = np.reshape(image, [-1, 310, 128, 1])  # reshape image to matrix
    model = tf.keras.models.load_model('/Users/cormac/Documents/WPIStuff/MQP/nn_old_data/web_app/testing/models/four_finger/both_datasets_convolutional.h5')
    preds = model.predict(image)
    preds = preds.flatten().tolist()  # convert one row matrix to a json serializable list
    return preds