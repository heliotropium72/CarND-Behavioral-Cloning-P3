# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:09:33 2017

@author: asd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import shuffle #in-place shuffle
from os.path import join, split, exists
import csv

# Check if tensorflow (and thus keras) is using the GPU
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


###############################################################################
#
# Load file paths and steering angles

# Select data set (I recorded different ones)
#DIR_DATA = 'C:\\Users\\asd\\Documents\\5_CourseWork\\sdc\\training_data'
DIR_DATA = 'C:\\Users\\asd\\Documents\\5_CourseWork\\sdc\\data'



data_rows = []
with open(join(DIR_DATA, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data_rows.append(line)
        
# delete header of logfile (only needed for data provided by udacity)
del data_rows[0]


###############################################################################
#
# Data preparation

# Reduce data amount for fast testing
#import random
#data_rows = random.sample(data_rows, 1500)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data_rows, test_size=0.2)


###############################################################################
#
# Load image data (using a generator)
#
# A generator will load the image data only when it is needed
# This generator is performing also data augmentation
# 1) Flipping the image (and angle)
# 2) Adding image from right and left camera and adding an offset to the angle

def generator(samples, batch_size=32, angle_corr=0.25):
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
                file_center = join(DIR_DATA, 'IMG', split(batch_sample[0])[1])
                file_left = join(DIR_DATA, 'IMG', split(batch_sample[1])[1])
                file_right = join(DIR_DATA, 'IMG', split(batch_sample[2])[1])
                if not exists(file_center):
                    print('{}  does not exist'.format(file_center))
                
                center_angle = float(batch_sample[3])
                
                # Image from central camera (converted from BGR to RGB)
                images.append(cv2.imread(file_center)[...,::-1])
                angles.append(center_angle)
                # Flipped Image from central camera
                images.append(cv2.flip(images[-1],1))
                angles.append(-1*center_angle)
                # Image from right camera
                images.append(cv2.imread(file_right)[...,::-1])
                angles.append(center_angle - angle_corr)
                # Image from left camera
                images.append(cv2.imread(file_left)[...,::-1])               
                angles.append(center_angle + angle_corr)
            
            # convert to array
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

IMAGE_INPUT = 32
AUGMENTATION = 4 # How many images are created form a single logfile row         
BATCH_SIZE = IMAGE_INPUT * AUGMENTATION
EPOCHS = 5
DROPOUT = 0.2

print('The data set contains {} samples, resulting in {} after augmentation'.format(len(train_samples), AUGMENTATION * len(train_samples)))

train_generator = generator(train_samples, batch_size=IMAGE_INPUT)
validation_generator = generator(validation_samples, batch_size=IMAGE_INPUT)

###############################################################################
#
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

# LeNet
model1 = Sequential()
model1.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Normalisation to -0.5 to 0.5
model1.add(Lambda(lambda x: x / 255.0 - 0.5))
model1.add(Conv2D(6, (5, 5), activation="relu"))
model1.add(MaxPooling2D())
model1.add(Conv2D(6, (5, 5), activation="relu"))
model1.add(MaxPooling2D())
model1.add(Flatten())
model1.add(Dense(120))
model1.add(Dropout(DROPOUT))
model1.add(Dense(84))
model1.add(Dropout(DROPOUT))
model1.add(Dense(1))


# Nvidia model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Normalisation to -0.5 to 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(DROPOUT))
model.add(Dense(50))
model.add(Dropout(DROPOUT))
model.add(Dense(10))
model.add(Dropout(DROPOUT))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
# This is the classical model when all data is in memory
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
# This is the model with generator functions
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/IMAGE_INPUT,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/IMAGE_INPUT,
                    epochs=EPOCHS, verbose=1)

model.save('model.h5')

###############################################################################
#
# Visualisation

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