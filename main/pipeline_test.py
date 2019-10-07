import sys
import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import utility
import preprocessor
import data

import utility.io as io

## READ DATA

train_path = os.path.join(data.PATH, "sign_mnist_train.csv")
test_path = os.path.join(data.PATH, "sign_mnist_test.csv")

train_data = io.read_data(train_path)
train_X, train_Y = train_data[:, 1:], train_data[:, 0]
print(train_X.shape)
print(train_Y.shape)

test_data = io.read_data(test_path)
test_X, test_Y = test_data[:, 1:], test_data[:, 0]
print(test_X.shape)
print(test_Y.shape)


## DATA DISTRIBUTION

train_labels = list(set(train_Y))
print("Train Labels: ".format(train_labels))
print("# of train label: {}".format(len(train_labels)))

print("\n")

test_labels = list(set(test_Y))
print("Test Labels: ".format(test_labels))
print("# of train label: {}".format(len(test_labels)))

train_labels_plotting = {label: train_Y[train_Y==label].shape[0] for label in train_labels}

test_labels_plotting = {label: test_Y[test_Y==label].shape[0] for label in test_labels}

fig, axs = plt.subplots(1, 1, figsize=(15, 8), sharey=True)
axs.bar(list(train_labels_plotting.keys()), list(train_labels_plotting.values()))
axs.bar(list(test_labels_plotting.keys()), list(test_labels_plotting.values()))
fig.suptitle('Data distribution Plotting')
# plt.show()


## VISUALIZE DATA

sample = train_X[0:10]
sample = np.reshape(sample, newshape=(-1, 28, 28))
fig = plt.figure(figsize=(10, 10))

for i in range(sample.shape[0]):
  fig.add_subplot(3, 4, i+1)
  plt.imshow(sample[i, :, :], cmap='gray')

# plt.show()

## BUILDING PIPELINE

hyper_param = {
    "lr":0.01
}


# import tensorflow as tf
# import numpy as np
class image_preprocessor():
    def __init__(self):
        return

    @classmethod
    def to3D(cls, images):
        return tf.reshape(images, shape=(-1, 28, 28, 1))

    @classmethod
    def normalize(cls, images):
        return tf.multiply(images, 1 / 255.0)

    @classmethod
    def flatten(cls, images):
        return tf.reshape(images, shape=(-1, 784))

    @classmethod
    def flatten2image(cls, x):
        return tf.reshape(x, shape=(-1, 28, 28))

    @classmethod
    def resize(cls, images, size):
        return tf.image.resize(images, size=size, method=tf.image.ResizeMethod.BILINEAR)

    @classmethod
    def padding(cls, images, pad_size):
        return tf.image.pad_to_bounding_box(
            images,
            offset_height=pad_size,
            offset_width=pad_size,
            target_height=tf.shape(images)[1] + 2 * pad_size,
            target_width=tf.shape(images)[2] + 2 * pad_size
        )


## INIT DATASET

# Load full data set
def load_full_dataset(X, Y):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))

    return ds


def load_tf_recorder(file_path):
    return


def load_from_file(file_path):
    return


def load_from_files(folder_path):
    return


def load_from_image(folder_path):
    return


def load_custom(path):
    return


def config_dataset(dataset, batch_size=64, is_shuffle=True):
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(100)
    return dataset

## TRAINING
class Model():

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.model = None
        return

    def build_model(self):
        input = tf.keras.Input(shape=(784,))
        X = input
        logit = tf.keras.layers.Dense(25)(X)
        prob = tf.keras.layers.Activation('softmax')(logit)
        model = tf.keras.Model(inputs=X, outputs=prob)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
        print(model.summary())
        self.model = model


train_X = train_X.astype(float) / 255.0
# print(np.mean(train_X))
# train_Y = train_Y.astype(float)
train_ds = load_full_dataset(train_X, np.reshape(train_Y, newshape=(-1, 1)))
train_ds = config_dataset(train_ds)
test = Model()
test.build_model()

# iterator = tf.data.D
# data = iterator.get_next()
# init_op = iterator.
# count=0
# with tf.Session() as sess:
#     # sess.run(init_op)
#     while True:
#         try:
#             print("START AT TRYING")
#             output = sess.run(data)
#             # print(np.mean(output[0]))
#             print(output[0].shape)
#             print(output[1].shape)
#             count += 1
#         except:
#             print("GOT SOME ERROR")
#             break

# print(train_ds)
# count = 0
# n_data = 0
# for element in train_ds:
#     print(element[0].numpy().shape)
#     count += 1
#     n_data += element[0].numpy().shape[0]
#
# print(count)
# print(n_data)
# # print(train_ds)
# print(train_X.shape[0] / 64 + 1)
# print(train_X.shape[0])

history = test.model.fit(
    train_ds,
    steps_per_epoch=int(train_X.shape[0] / 64 + 1),
    epochs=100,
    verbose=1)
# print(history)
test.model.evaluate(x=test_X, y=test_Y)