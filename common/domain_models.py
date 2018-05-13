import os
import sys
import shutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('common')
sys.path.append('external')
import util

import keras
import keras.backend as K
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D,
                          Activation, Dropout, BatchNormalization, Flatten)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

# Half part of AlexNet based, matched to input shape (40, 501)
def model_cnn_alexnet(input_shape, num_classes, lr=0.0001):
    model = Sequential()
 
    model.add(Conv2D(48, 11,  input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])
    return model

#  Based on [TensorFlow Speech Recognition Challenge 2nd place solution by Thomas O'Malley
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47715
def model_cnn7_for_event_pattern(input_shape, num_classes, lr):
    model = Sequential()
    #  > I thought of this as a "denoising" and basic feature extraction step
    model.add(Conv2D(64, (7, 3),  input_shape=input_shape, strides=(1,2), activation='relu', padding='same'))
    # > Getting back down to the standard 40 frequency features
    model.add(MaxPooling2D((4, 1), strides=(2, 1)))
    model.add(BatchNormalization())
    #  > Look for local patterns across frequency bands
    model.add(Conv2D(128, (7, 1),  strides=(1,1), activation='relu', padding='same'))
    # Allow for speaker variation, similar to what worked here:
    # https://link.springer.com/content/pdf/10.1186%2Fs13636-015-0068-3.pdf
    model.add(MaxPooling2D((4, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # This allows it to treat each remaining freq band very differently, and compress the frequency dimension entirely.
    # I think of this as detecting phoneme-level features
    model.add(Conv2D(128, (5, 1),  input_shape=input_shape, strides=(1,1), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    # I think of this as looking for connected components of a short keyword at different points in time
    model.add(Conv2D(128, (1, 5),  input_shape=input_shape, strides=(1,1), activation='relu', padding='same'))
    # Collect all the components
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))
    # Because why not, and seemed to work well
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])
    return model

# Half part of AlexNet based, matched to input shape (40, 501)
# Input will soon be shrinked to 40, 1 at the first layer
def model_cnn5_timewise_mean(input_shape, num_classes, lr):
    model = Sequential()
    
    # This computes time-wise mean
    model.add(AveragePooling2D((1, input_shape[1]), input_shape=input_shape, strides=1))
 
    # Followings are based on half-AlexNet
    model.add(Conv2D(48, (11, 1),  input_shape=input_shape, strides=(2,1), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 1), strides=(1,1)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (5, 1), strides=(2,1), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 1), strides=(2, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 1), strides=1, activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 1), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 1), strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 1), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 1), strides=(1,1)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])
    return model
