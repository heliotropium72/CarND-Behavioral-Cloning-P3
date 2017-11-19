# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:09:33 2017

@author: asd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Check if tensorflow (and thus keras) is using the GPU
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


###############################################################################
# Read the data 
# (recorded in self-driving mode)
'''
# all at once into memory
import pandas as pd

datalog = '../data/driving_log.csv'
data = pd.read_csv(datalog)

images = []
for camera in ['center', 'left', 'right']:
    for idx in range(len(data)):
        source_path = data[camera].values[idx]
        filename = source_path.split('/')[-1]
        # use the path on the server
        current_path = '../data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)

X_train = np.array(images)

correction_angle = 0.2
#y_train = data['steering'].values

# all three cameras
y_train = np.concatenate([data['steering'].values,
                     data['steering'].values + correction_angle,
                     data['steering'].values - correction_angle])
'''

#################
# Data augementation
# TODO: flip all images and add with negative steering angle   


#################
# Use a generator instead of loading all data into memory
# A generator will load the data only when it is needed

#import os
import csv

data_rows = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data_rows.append(line)
# delete header of logfile
del data_rows[0]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data_rows, test_size=0.2)

from random import shuffle #in-place shuffle

def generator(samples, batch_size=32):
    ''' Loads a batch of images from samples
    Parameters:
    -----------
        samples : list
            list of rows containg "center", "left", "right", "steering",
            "throttle", "brake", "speed" from logfile of the simulator
        batch_size : int (default 32)
            number of images in return batch
    Yields:
    -------
    tuple of 2 ndarray
        shuffled X_train with corresponding y_train
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = '../data/IMG/'+batch_sample[0].split('/')[-1]
                #name_left = './IMG/'+batch_sample[1].split('/')[-1]
                #name_right = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function

BATCH_SIZE = 32            

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

#ch, row, col = 3, 80, 320  # Trimmed image format

#################
# Prediction model: Regression model
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
# Flatten : Reduce dimensions to 1D array
# Dense : Fully connected later
# Lambda : Arbitrary function e.g. preprocessing
# Dropout : Fraction of input to be dropped (to prevent overfitting)
# Cropping2D : Crop image to reduce disturbing information (sky, trees,...) (This is parallised on the GPU so faster than outside the model)
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# TODO: Implement deeper network from nvidia
# TODO: Record more data
# Use a generator and batches to speed up data procession

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Normalisation to -0.5 to 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.1))
model.add(Dense(84))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# This is the classical model when all data is in memory
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
# This is the model with generator functions
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/BATCH_SIZE,
                    epochs=3, verbose=1)

model.save('model.h5')


##################
# Visualisation
#from keras.models import Model

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()