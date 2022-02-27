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

char_list = ['a', 'A', 'b', 'B', 'c', 'C', 'ç', 'Ç', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'ğ', 'Ğ', 'h', 'H', 'ı', 'I', 'i', 'İ', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ö', 'Ö', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 'ş', 'Ş', 't', 'T', 'u', 'U', 'ü', 'Ü', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', ',', '.', '#', '+', '%', ';', '=', '(', ')', "'"]


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


def Model(epochs=0):
    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32, 128, 1), name="image")
    #maxlen
    labels = layers.Input(name="label", shape=(20,), dtype="float32")

    conv_1 = Conv2D(32, (3,3), activation = "selu", padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(64, (3,3), activation = "selu", padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(128, (3,3), activation = "selu", padding='same')(pool_2)
    conv_4 = Conv2D(128, (3,3), activation = "selu", padding='same')(conv_3)

    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(256, (3,3), activation = "selu", padding='same')(pool_4)

    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(256, (3,3), activation = "selu", padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(64, (2,2), activation = "selu")(pool_6)

    squeezed = tf.squeeze(conv_7,1)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    softmax_output = Dense(len(char_list) + 1, activation = 'softmax', name="dense")(blstm_2)

    output = CTCLayer(name="ctc_loss")(labels, softmax_output)


    #model to be used at training time
    model = Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer = "adam")

    prediction_model = Model(inputs=inputs, outputs=softmax_output)
    return prediction_model

    
    # # input with shape of height=32 and width=128 
    # inputs = Input(shape=(32, 128, 1), name="image")

    # labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # conv_1 = Conv2D(32, (3,3), activation = "selu", padding='same')(inputs)
    # pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)
    
    # conv_2 = Conv2D(64, (3,3), activation = "selu", padding='same')(pool_1)
    # pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)

    # conv_3 = Conv2D(128, (3,3), activation = "selu", padding='same')(pool_2)
    # conv_4 = Conv2D(128, (3,3), activation = "selu", padding='same')(conv_3)

    # pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    
    # conv_5 = Conv2D(256, (3,3), activation = "selu", padding='same')(pool_4)
    
    # # Batch normalization layer
    # batch_norm_5 = BatchNormalization()(conv_5)
    
    # conv_6 = Conv2D(256, (3,3), activation = "selu", padding='same')(batch_norm_5)
    # batch_norm_6 = BatchNormalization()(conv_6)
    # pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    
    # conv_7 = Conv2D(64, (2,2), activation = "selu")(pool_6)
    
    # squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    
    # # bidirectional LSTM layers with units=128
    # blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    # blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    
    # softmax_output = Dense(len(char_list) + 1, activation = 'softmax', name="dense")(blstm_2)

    # output = CTCLayer(name="ctc_loss")(labels, softmax_output)


    # optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

    

    # #model to be used at training time
    # model = keras.models.Model(inputs=[inputs, labels], outputs=output)
    # #model.load_weights('/content/TextRecognition_best.hdf5')
    # model.compile(optimizer = optimizer)
    

    # # file_path = "TextRecognition_lower_Char_best.hdf5"
    
    # # checkpoint = ModelCheckpoint(filepath=file_path, 
    # #                             monitor='val_loss', 
    # #                             verbose=1, 
    # #                             save_best_only=True, 
    # #                             mode='min')

    # # callbacks_list = [checkpoint, 
    # #                   #PlotPredictions(frequency=1),
    # #                   #EarlyStopping(patience=3, verbose=1)
    # # ]

    # # history = model.fit(#train_dataset, 
    # #                     epochs = epochs,
    # #                     #validation_data=validation_dataset,
    # #                     verbose = 1,
    # #                     callbacks = callbacks_list,
    # #                     shuffle=True)

    
    
    # return model