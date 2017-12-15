# Behavioral Cloning Project
## Work in Progress: 19/11/2017

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


### Overview
---

In this project the steering angles of an autonomous vehicle are predicted based on camera images. The project is written in python 3.6. using the deep learning library keras (API 2).
The approach is based on behavioral cloning. This means that a simulated car is driven in a simulator recording images from three frontal cameras (left, center, right) and the steering angles.
Based on this dataset, a model is created imitating the behavior in the simulator. Later in autonomous mode, the vehicle predicts the steering angle based on the (frontal) camera images and keeps on track in the test courses.
The project fulfills the [rubric points](https://review.udacity.com/#!/rubrics/432/view) for submission.

---

This repository contains the following files:

* model.py (script used to create and train the model)
* drive.py (script to drive the car (not changed))
* video.py (script to create a video out of the output images from drive.py (not changed))
* model.h5 (a trained Keras model)
* video.mp4 (a video recording of the vehicle driving autonomously around the track for one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section and contains the writeup of the project.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

If the starter-kit is not installed, the following packages needs to be installed additionally to keras:
* socketio
* eventlet
* flask

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.


## Results of the project

[//]: # (Image References)

[image1]: ./Figures/loss.png "Loss"
[center]: ./Figures/center.jpg "Center"
[left]: ./Figures/left.jpg "Left"
[right]: ./Figures/right.jpg "Right"
[recovery1]: ./Figures/center_2017_11_24_13_23_12_576.jpg "Recovery 1"
[recovery2]: ./Figures/center_2017_11_24_13_23_15_884.jpg "Recovery 2"
[recovery3]: ./Figures/center_2017_11_24_13_23_16_708.jpg "Recovery 3"

### Training data
A set of example training data was already provided by Udacity. However, additional data needed to be recorded to reach a sufficient amount of images.
The hardest part of this project was to get good training data from the simulator. The controls of the car in the simulator was difficult leading to recordings of "bad" steering angles.
Thus the performance of the model highly depended on the quality of the training data. I recorded the following driving situations on the first track with rather low speed:
- normal lap
- lap in the opposite direction
- recovery to the lane center when the car was driving on the right/left edge

I carefully went through the whole data set again and cleaned it: Data recorded with the keyboard was deleted completely as the steering angles were not continuously changing but rather jumpy. During the full laps, periods of "bad driving"
e.g. over-steering to one side of the road were cut out as well. I got observe an increase in performance in the video: the car wiggled much lesser and the steering angles became more stable.

Here is an example of center lane driving from the three cameras:
![alt text][left]
![alt text][center]
![alt text][right]

and an example of a recovery from the side of the road:
![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

Further, the training data set was further augmented by flipping the images (steering angle is the same but with the opposite sign) and by including also the images from the left and right camera. In this case an offset of 0.25 was added/subtracted to the steering angle, respectively.
The offset was found empirically.

The total amount of recorded images was 22954 which was augmented by a factor of 4 to 91816 images.

### Model Architecture and Training Strategy

#### 1. Preprocessing
The images are read-in using a generator to save memory. Inside the generator the data augmentation (see section above) happens. Further, the image are read-in usging OpenCV and are therefore in the RGB order.
They are converted into BGR because the images are read-in as BGR by ´drive.py´ while driving in autonomous mode. This is important as yellow lane marks are only detected successfully if the colour order is correct.
Inside the model architecture, the images are cropped to the relevant scene and normalized.

#### 2. Nvidia architecture for steering angle prediction
`model.py` contains two model implementations: A LeNet and the Nividia architecture. I started with the simpler LeNet to get an overview of the data but then switched to the Nividia model.

The model is based on the CNN architecture of [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with some modifications.
The Nvidia model consits of five convolutional layers, and three fully connected layers. 
The Nvidia model was extent with dropout layers to reduce overfitting.

#### 3. Training
The training data set was divided into 80% training and 20% validation datasets.
After training of the model, the model was tested in the simulator and inspected by eye how well the autonomous mode of the car steering performed (center of lane driving, going off-track, ...).
More training data was recorded for parts of the track, where the model had problems (see section above).

The model was trained by minimizing the loss using an Adam optimizer, so the learning rate was adapted automatically.

I used the following hyperparameters for training

| Hyperparameter | Value  | 
|:--------------:|:------:|
| Learning rate  | adam opt. |
| Dropout	     | 0.80   | 
| Batch size     | 32*4 = 128    |
| Epochs         | 5     |

The model performed well on training and validation set suggesting that there is no major overfitting.
![alt text][image1]


### Performance on track
After training, the car suffesfully stays on track 1. There is still room for improvement as the car is wiggeling around the center of the lane.
Also the recorded data is all from the same track, using also data from the second track would generalize the model to new situations. However, for me this was out of questions due to the simulator controls.

