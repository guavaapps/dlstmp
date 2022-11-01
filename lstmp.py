import logging

import numpy as np
import tensorflow.python as tf
import tensorflow.python.framework.ops
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.ops import variables
from tensorflow.python.ops.array_ops import *
from tensorflow.python.trackable.data_structures import ListWrapper

HIDDEN = 0
CELL = 1


# single LSTM-P layer
class LSTMP(keras.Model):
    def __init__(self, units: int, batch_input_shape, return_sequences=True, name="", *args, **kwargs):
        super(LSTMP, self).__init__()

        self.a = None
        self.lstm = layers.LSTM(
            units,
            batch_input_shape=batch_input_shape,
            stateful=True,
            name=name,
            return_sequences=return_sequences
            # return_state=True
        )

        self.proj = layers.Dense(units, name="denseLayer")

    def call(self, inputs, training=False):
        self.a = self.lstm(inputs)
        self.a = self.proj(self.a)
        c = self.lstm.states[CELL].numpy().reshape(1, 4)
        h = (self.a[-1][-1] if self.lstm.return_sequences else self.a[-1]).numpy().reshape(1, 4)
        self.lstm.reset_states([h, c])

        return self.a
