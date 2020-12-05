"""
This file contains classes which implement deep neural networks namely CNN and LSTM
"""
import sys

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D


class CNN(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, **params):
        params['name'] = 'CNN'
        super(CNN, self).__init__(**params)

    def make_default_model(self):
        """
        Makes a CNN keras model with the default hyper parameters.
        """
        self.model.add(Conv2D(8, (13, 13),
                              input_shape=(
                                  self.input_shape[0], self.input_shape[1], 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

class LSTM(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)

    def make_default_model(self):
        """
        Makes the LSTM model with keras with the default hyper parameters.
        """
        self.model.add(
            KERAS_LSTM(128,
                       input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))