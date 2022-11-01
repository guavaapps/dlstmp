import keras
import numpy as np
import json
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.trackable.data_structures import ListWrapper


class LSTMPLayer(layers.LSTM):
    def __init__(self, units, return_sequences=False):
        super(LSTMPLayer, self).__init__(units, return_sequences=return_sequences, stateful=True)

        self.w = tf.Variable(tf.random.normal((units, units)), trainable=True)
        self.w = tf.Variable ([[0., 0.], [1., 1.]])

    def call(self, inputs, training=None, mask=None):
        o = super(LSTMPLayer, self).call(inputs, training, mask)

        h_states = self.states [0]
        h_type = type (h_states)

        o_type = type (o)

        var = tf.Variable (h_states.numpy ())
        print (h_states)
        print(var)

        return tf.matmul(o, self.w)

