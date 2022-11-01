import random
import time

import numpy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf
import tensorflow.python.framework.ops
from tensorflow import keras

layers = keras.layers

look_back = 5
features = 2


def create_batches(dataset, look_back=1):
    x, y = [], []

    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]

        x.append(a)
        y.append(dataset[i + look_back])

    return numpy.array(x), numpy.array(y)


def scale(dataset):
    print(dataset.shape)

    temp = dataset.reshape(features, len(dataset))
    scaled = []

    for feature in temp:
        min = np.array(feature).min()
        max = np.array(feature).max()

        delta = max - min

        for timestep in feature:
            scaled.append((timestep - min) / delta)

    scaled = numpy.array(scaled).reshape(len(dataset), features)

    return scaled


def run(model):
    dataset = pd.read_csv("pollution.csv", usecols=["DEWP", "TEMP"], engine="python")
    dataset = dataset.values.astype("float32")
    dataset = dataset[:100]

    dataset = scale(dataset)

    trainX, trainY = create_batches(dataset, look_back)

    print(trainX.shape)

    model.compile(optimizer="adam", loss="mse")
    model.fit(trainX, trainY, epochs=2, batch_size=5, verbose=2)

    model.save("model2")

    converter = tf.lite.TFLiteConverter.from_saved_model("model2")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    with open("model2.tflite", "wb") as m:
        m.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path="model2.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output = []

    for batch in trainX:
        interpreter.set_tensor(
            input_details[0]["index"],
            batch.astype("float32").reshape(1, look_back, features)
        )

        shape = interpreter.get_tensor(
            input_details[0]["index"]
        ).shape

        print(f"input tensor shape - {shape}")

        interpreter.invoke()

        o = interpreter.get_tensor(output_details[0]["index"])

        print(o)

        output.append(o)

    y_temp = []
    y_pres = []

    temp = []
    pres = []

    print(trainY.shape)

    for batch in trainY:
        y_temp.append(np.array(batch).flatten().tolist()[0])
        y_pres.append(np.array(batch).flatten().tolist()[1])

    for batch in output:
        temp.append(np.array(batch).flatten().tolist()[0])
        pres.append(np.array(batch).flatten().tolist()[1])

    plt.plot(y_temp, label="y_temp")
    plt.plot(y_pres, label="y_pres")
    plt.plot(temp, label="temp")
    plt.plot(pres, label="pres")
    plt.legend(loc="upper left")
    plt.show()

    return None
