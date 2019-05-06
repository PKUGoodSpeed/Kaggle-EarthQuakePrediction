import os
import gc
import sys
import numpy as np
import pandas as pd

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import LearningRateScheduler
from keras import regularizers

class CustomizedCNNModel:
    NAME = "ccnn"
    IMG_HEIGHT = 288
    IMG_WIDTH = 192
    NUM_CHANNEL = 1
    _model = None
    
    def __init__(self):
        pass

    def buildModel(self):
        print("Constructing {} Model".format(self.NAME))
        model = Sequential()
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            input_shape=(
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                self.NUM_CHANNEL)))

        model.add(
            Conv2D(64, kernel_size = (7, 7), padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(0.12))

        model.add(
            Conv2D(128, kernel_size = (5, 5), padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(
            Conv2D(256, kernel_size = (3, 3), padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(
            Conv2D(512, kernel_size = (3, 3), padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation="linear"))
        model.summary()
        self._model = model

    def getModel(self):
        self.buildModel()
        return self._model
    
    def getModelName(self):
        return self.NAME

    def loadWeightsFromFile(self, model_file=None):
        if model_file is None:
            model_file = self.getModelName() + ".h5"
        model_file = os.path.join(ROOT, "nnmodels", model_file)
        assert os.path.exists(model_file)
        print("Loading model from {FILE} ...".format(FILE=model_file))
        self._model.load_weights(model_file)