import json

import numpy as np
import tensorflow.python as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.trackable.data_structures import ListWrapper
import lstmp

HIDDEN = 0
CELL = 1


# wrapper class for DLSTM-P model
class DLSTMP():
    def __init__(self, units, depth, batch_input_shape, *args, **kwargs):
        super(DLSTMP, self).__init__()
        self.__model = keras.Sequential()
        for i in range(depth - 1):
            self.__model.add(lstmp.LSTMP(units, batch_input_shape))  # add returning layers

        # non returning layer
        self.__model.add(lstmp.LSTMP(units, batch_input_shape, return_sequences=False))
        # condense output // not needed TODO remove
        # self.__model.add(layers.Dense(units))

    def __call__(self, inputs, *args):
        return self.__model(inputs)

    def c(self):
        self.__model.compile(
            optimizer=optimizers.adam_v2.Adam(learning_rate=1e-3),
            loss=losses.MeanSquaredError(),
            metrics=metrics.Accuracy()
        )

        print("not implemented")

