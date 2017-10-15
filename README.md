# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

## Results summary

The video shows that [NVidia End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) seem to work well and I was able to get a complete lap for both Track1 and Track2.


[//]: # (Image References)

[image1]: ./examples/track1_center.png "Track1 Center Image"
[image2]: ./examples/track1_center_flip.png "Track1 Center Flip Image"
[image3]: ./examples/track2_left.png  "Track2 Left Image "
[image4]: ./examples/track2_right.png  "Track2 Right Image"
[image5]: ./examples/track1_center_cropped.png  "Track1 Center Cropped Image"
[image6]: ./examples/track2_left_cropped.png  "Track2 Left Cropped Image"

## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Clear functions with comments are provided.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model closely follows [NVidia End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture with respect to number of layers and not using any pooling between the Convolutions but rather use larger strides.

Padding is set to VALID and this seems to clearly outperform SAME padding option for the model size.

I have used larger fully connected layers and incorporated dropout for the first two fully connected layers

The model includes RELU layers to introduce nonlinearity (code line 107), and the data is normalized in the model using a Keras lambda layer (code line 101).

Image is initially cropped with removing top 40% and bottom 15%.

The training and test data images have CLAHE transform applied and converted back to RGB.

drive.py is modified to ensure that images stream has the same CLAHE transform applied.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 120 and 123). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 136-140). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137).

#### 4. Appropriate training data

I used XBox controller to generate training data. Resolution is 800x600 and graphics quality is set to "Fantastic".

Training data was chosen to keep the vehicle driving on the road. I used a combination of training data from both the tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try smaller CNN model and then increase the depth and size of it.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it performs image classification, it is simple and seem to work.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the first model after few epocs the training loss stopped decreasing. 

I then generated more data tried with and withot left,right images and still the car would not get very far.

Then I implemented [NVidia End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) exactly to spec and the only after 5 to 10 epochs it seems to do well.

I generated more data using track2 and performace on track1 remarkably improved.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I continued to take more sampling data from Track2 and improved performance and made the model work for both Track1 and Track2.

#### 2. Final Model Architecture

The final model architecture (model.py lines 107-127) consisted of a convolution neural network with the following layers and layer sizes:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_3 (Cropping2D)    (None, 72, 320, 3)        0         
_________________________________________________________________
lambda_3 (Lambda)            (None, 72, 320, 3)        0         
_________________________________________________________________
average_pooling2d_3 (Average (None, 72, 80, 3)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 34, 38, 32)        2432      
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 15, 17, 48)        38448     
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 6, 7, 64)          76864     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 4, 5, 96)          55392     
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 3, 96)          83040     
_________________________________________________________________
flatten_2 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 200)               115400    
_________________________________________________________________
dropout_3 (Dropout)          (None, 200)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 100)               20100     
_________________________________________________________________
dropout_4 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1010      
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 11        
=================================================================
Total params: 392,697
Trainable params: 392,697
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded for the second track:

Left camera 

![alt text][image3]

Right camera

![alt text][image4]


To augment the data set, I also flipped images and angles so that the steering value is still correct and makes sure that lap orientation does not create a bias in training data. For example, here is an image that has then been flipped:

![alt text][image2]

The images are cropped with removing top 40% and bottom 15% for the above images:
Track1

![alt text][image5]

Track2

![alt text][image6]

Images are then resized down by scale factor of 4 for x-axis.

After the collection process, I had 36228 number of data points. All images are preprocessed by applying CLAHE transformation as show above.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
