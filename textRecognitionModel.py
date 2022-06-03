import os
import cv2
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



class CTCLayer(layers.Layer):

    def __init__(self, name=None):

        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
        
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding)(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output


def Model(epochs=0):
    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32, 128, 1), name="image")

    labels = layers.Input(name="label", shape=(20,), dtype="float32")

    conv_1 = convolutional(inputs, (1, 1,  3,  32))
    conv_1 = convolutional(conv_1, (3, 3,  32,  32))
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)


    conv_2 = convolutional(pool_1, (1, 1,  32,  64))
    conv_2 = convolutional(conv_2, (3, 3,  64,  64))
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)


    conv_3 = convolutional(pool_2, (1, 1,  64,  128))
    conv_3 = convolutional(conv_3, (3, 3,  128,  128))
    conv_3 = convolutional(conv_3, (3, 3,  128,  128))
    pool_3 = MaxPool2D(pool_size=(2, 2))(conv_3)



    conv_4 = convolutional(pool_3, (1, 1,  128,  256))
    conv_4 = convolutional(conv_4, (3, 3,  256,  256))
    conv_4 = convolutional(conv_4, (3, 3,  256,  256))
    pool_4 = MaxPool2D(pool_size=(2, 2))(conv_4)



    conv_5 = convolutional(pool_4, (1, 1,  256,  512))
    conv_5 = convolutional(conv_5, (3, 3,  512,  512))
    conv_5 = convolutional(conv_5, (3, 3,  512,  512))
    pool_5 = MaxPool2D(pool_size=(2, 1))(conv_5)


    dense_out = Dense(512,activation="relu",name="dense_1")(pool_5)

    dense_out = Dense(256,activation="relu",name="dense_3")(dense_out)

    dense_out = Dense(512,activation="relu",name="dense_4")(dense_out)

    dense_add = pool_5 + dense_out


    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(dense_add)

    squeezed = tf.squeeze(batch_norm_5,1)

    squeezed = tf.reshape(squeezed,(-1,32,128))


    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    softmax_output = Dense(88, activation = 'softmax', name="dense")(blstm_2)

    output = CTCLayer(name="ctc_loss")(labels, softmax_output)


    #model to be used at training time
    model = keras.models.Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer = "adam")

    prediction_model = keras.models.Model(inputs=inputs, outputs=softmax_output)
    return prediction_model

