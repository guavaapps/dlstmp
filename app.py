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

import test2

layers = keras.layers

# keras more like ker-ass

tbatch = np.array([[
    [0.4, 0.3, 0.1, 0.2],
    [0.4, 0.3, 0.1, 0.2],
    # [0.4, 0.3, 0.1, 0.2]
]])

target = 572.1

"""
"1-01",266.0
"1-02",145.9
"1-03",183.1
"1-04",119.3
"1-05",180.3
"1-06",168.5
"1-07",231.8
"1-08",224.5
"1-09",192.8
"1-10",122.9
"1-11",336.5
"1-12",185.9
"2-01",194.3
"2-02",149.5
"2-03",210.1
"2-04",273.3
"2-05",191.4
"2-06",287.0
"2-07",226.0
"2-08",303.6
"2-09",289.9
"2-10",421.6
"2-11",264.5
"2-12",342.3
"3-01",339.7
"3-02",440.4
"3-03",315.9
"3-04",439.3
"3-05",401.3
"3-06",437.4
"3-07",575.5
"3-08",407.6
"3-09",682.0
"3-10",475.3
"3-11",581.3
"3-12",646.9
"""


class LSTMPCell(layers.LSTMCell):
    def __init__(self, units):
        super(LSTMPCell, self).__init__(units)

        self.units = units
        self.proj_w = tf.Variable(tf.random.normal([units, units]), trainable=True, name="proj_w")

    def call(self, inputs, states, training=None):
        h, c = states

        o, [_, c] = super(LSTMPCell, self).call(inputs, states, training)

        proj_h = tf.matmul(h, self.proj_w)

        return o, [proj_h, c]

    def get_config(self):
        config = super(LSTMPCell, self).get_config()
        config.update({
            "units": self.units,
            "proj_w": self.proj_w.numpy()
        })
        return config


class LSTMP(layers.RNN):
    def __init__(self, units, return_sequences=False):
        self.cell = LSTMPCell(units)
        super(LSTMP, self).__init__(cell=self.cell, return_sequences=return_sequences, stateful=False)

    def get_config(self):
        base_config = super(LSTMP, self).get_config()
        base_config.update({
            # "cell": self.cell,
            # "units": self.units,
            # "return_sequences": self.return_sequences
        })

        return base_config


input = layers.Input(shape=(4, 2))  # 2 ts 4 f // 1, 4
lstmp1 = LSTMP(64, return_sequences=True)(input)
lstmp2 = LSTMP(64)(lstmp1)
output = layers.Dense(2)(lstmp2)

model = keras.Model(inputs=input, outputs=output, name="dlstmp")

# new model ts=1, f=2
lb = 5
f = 2

input = layers.Input(shape=(lb, f))  # 2 ts 4 f // 1, 4
lstmp1 = LSTMP(64, return_sequences=True)(input)
lstmp2 = LSTMP(64)(lstmp1)
output = layers.Dense(f)(lstmp2)

model = keras.Model(inputs=input, outputs=output, name="dlstmp")

test2.run(model)

exit(1)

"""
dlstmp

12-dim vector input 

input (1, *, 12)
lstmp (_, _, 128)
lstmp (_, _, 128)
dense (_, _, 12)

"""

dataset = pd.read_csv("airline-passengers.csv", usecols=[1], engine="python")


def scale(data):
    min = data.min()
    max = data.max()

    delta = max - min

    shape = data.shape

    scaled = []

    for d in data.flatten():
        new_d = (d - min) / delta
        scaled.append(new_d)

    return numpy.array(scaled).reshape(shape)


tsbatch = dataset.values.astype("float32")
tsbatch = scale(tsbatch)

print(f"shape of data - {tsbatch.shape}")

# data is 144 timesteps of 1-feature data

# use first 2/3 of the data to train
# use last 1/3 of the data to test

train_size = int(len(tsbatch) * 0.67)
test_size = len(tsbatch) - train_size
train, test = tsbatch[0:train_size, :], tsbatch[train_size:len(tsbatch), :]
print(len(train), len(test))


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]  # add ,0 before last ]

        a_list = a.tolist ()

        for j in range (len(a_list)):
            ts = a_list [j] # timestep
            ts.append (1.)

            a_list[j] = np.array(ts)

        a = np.array(a_list)

        dataX.append(a)

        y = dataset [i + look_back].tolist ()
        y.append (1.)
        y = np.array(y)
        print (f"y - {dataset [i + look_back]}")

        dataY.append(y)#dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX.shape, trainY.shape)
print(trainX[0], trainY[0])
print(trainX[1], trainY[1])

# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 2))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 2))

print(f"trainX shape - {trainX.shape}")
print(f"trainX - {trainX}")

model.compile(optimizer="adam", loss="mse")
start = time.time()
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
end = time.time()

delta = end - start

print(f"fit completed in {delta} millis")

tpred = model.predict(trainX)
tspred = model.predict(testX)

plt.plot(tsbatch)
plt.plot(tspred)
plt.plot(tpred)

print(f"dataset {dataset.shape}")
print(f"trainX {trainX.shape}")

model.save("model")

converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

tflite_model = converter.convert()

with open("model.tflite", "wb") as m:
    m.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print(input_details)

# print (f"shape {trainX.shape}") [0] # first batch
index = random.randint(0, len(trainX) - 1)
# trainXbatch = trainX[index].astype("float32").reshape(1, 1, 1)
trainYbatch = trainY[index]

output = []

i = input_details[0]["index"]
print(i)

flat_output = []

for batch in trainX:
    interpreter.set_tensor(
        input_details[0]["index"],
        batch.astype("float32").reshape(1, 4, 2)  # 1, 1, 1
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"output_data - {output_data}")

    output.append(output_data [0] [0])
    flat_output.append(output_data[0] [1])

pltY = []

for item in trainY:
    pltY.append(item [0])

pltY = np.array(pltY).flatten().tolist()

flat_pltY = []

for item in trainY:
    flat_pltY.append(item [1])

flat_pltY = np.array(flat_pltY).flatten().tolist()

plt.cla()
plt.plot(pltY)
plt.plot(flat_pltY)
plt.plot(numpy.array(output).flatten().tolist())
plt.plot(numpy.array(flat_output).flatten().tolist())
plt.show()

print(f"{output_data} {trainYbatch}")

androidTest = np.array([[0.01544402, 1],
                        [0.02702703, 1],
                        [0.05405406, 1],
                        [0.04826255, 1]])

interpreter.set_tensor(
    input_details[0]["index"],
    androidTest.astype("float32").reshape(1, 4, 2)
)

interpreter.invoke()

androidOutput = interpreter.get_tensor(output_details[0]["index"])

print(androidOutput)