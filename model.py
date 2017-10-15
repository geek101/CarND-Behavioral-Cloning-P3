import argparse
import csv
import os
import numpy as np
from PIL import Image
import cv2

import keras
from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K

import warnings
warnings.simplefilter('ignore', UserWarning)

clahe = cv2.createCLAHE()
def _rbg_to_CLAHE(img):
    l,a,b = cv2.split(img)
    l_clahe =  clahe.apply(l)
    img = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

def process_image(img):
    return _rbg_to_CLAHE(img)

def process_raw_input(track_path, csv_fname = "driving_log.csv", img_dir = "IMG", correction=0.2, include_left=True, include_right=True):
    """
    Take the raw data and process them i.e crop and run the image
    processing pipeline if it is necessary
    """
    car_images = []
    steering_angles = []
    with open(track_path + "/" + csv_fname) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            img_path = track_path + "/" + img_dir
            img_center = process_image(cv2.cvtColor(cv2.imread((img_path + "/" + os.path.basename(row[0])), 
                cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab))
            image_center_flipped = np.fliplr(img_center)
            car_images.extend([img_center, image_center_flipped])
            steering_angles.extend([steering_center, -steering_center])
            if include_left is True:
                img_left = process_image(cv2.cvtColor(cv2.imread(img_path + "/" + os.path.basename(row[1]), 
                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab))
                car_images.append(img_left)
                steering_angles.append(steering_left)
            
            if include_right is True:
                img_right = process_image(cv2.cvtColor(cv2.imread(img_path + "/" + os.path.basename(row[2]), 
                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab))
                car_images.append(img_right)
                steering_angles.append(steering_right)
                  
    return car_images, steering_angles

def process_raw_inputs(track_paths, correction=0.2, include_left=True, include_right=True):
    """
    Process multiple directories of track training data.
    """
    car_images = []
    steering_angles = []

    for track_path in track_paths:
        print("Preparing data for training from: {}".format(track_path))
        x, y = process_raw_input(track_path, correction=correction, include_left=include_left, 
            include_right=include_right)
        print("Collected data size: {} from: {}".format(len(x), track_path))
        car_images.extend(x)
        steering_angles.extend(y)

    print("Total data size: {}".format(len(car_images)))
    return np.array(car_images), np.array(steering_angles)


def CarND(input_shape, top_crop=0.40, bottom_crop=0.85):
    """
    Model is based on the following:
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

    We do use more filters for all Convolutional layers. This seems to perform better
    specially for track2.

    Dropout only for the first 2 hidden layers.
    """
    model = Sequential()
    
    # Crop the top and bottom of the image.
    model.add(Cropping2D(cropping=((int(input_shape[0]*top_crop), int(input_shape[0]*(1-bottom_crop))), (0, 0)), 
        input_shape=input_shape))

    # Normalize the pixel values to mean.
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    # Reduce the input size, similar to image resize perhaps use TensorFlow Image API?
    model.add(AveragePooling2D(pool_size=(1,4), trainable=False))
    
    # Layer 1 Conv with relu, more filters than recommended
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', strides=2, padding='valid'))
    # Layer 2 Conv with relu, more filters than recommended
    model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', strides=2, padding='valid'))
    # Layer 3 Conv with relu, more filters than recommended
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', strides=2, padding='valid'))
    # Layer 4 Conv with relu, more filters than recommended
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', strides=1, padding='valid'))
    # Layer 5 Conv with relu, more filters than recommended
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', strides=1, padding='valid'))
    # Layer 6 Flatten for fully connected layers
    model.add(Flatten())
    # Layer 7 Fully connected with 200 Units (100 more than recommeded hence the dropout)
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    # Layer 8 Fully connected with 100 Units (50 more than recommended hence the dropout)
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    # Layer 9 Fully connected with 10 Units same as recomended
    model.add(Dense(10, activation='relu'))
    # Output Layer
    model.add(Dense(1))

    return model

def train_model(model, x_train, y_train, epochs=32, batch_size=64, verbose=1):
    """
    Train the model.
    Using mean squared error as loss function and Adam optimizer, default learning rate is .001
    """
    model.compile(loss=keras.losses.mean_squared_error, 
        optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
        verbose=verbose, validation_split=0.2, shuffle=True)

def run_model(X_train, y_train, epochs=32, batch_size=64, verbose=1, model_name="model.h5"):
    """
    Top level helper to create the model, train it and then save it
    """
    print ("Using keras {}".format(str(keras_version).encode('utf8')))

    # Get the model
    model = CarND(X_train[0].shape)

    # Train the model
    train_model(model, X_train, y_train)

    # Save the model
    print("Saving model ... {}".format(model_name))
    model.save(model_name)

def run(data_paths, include_left, include_right, correction, epochs, batch_size, model_name, verbose):
    """
    Prepare the training data then call then top level model API to create, run and save the model.
    """
    X_train, y_train = process_raw_inputs(data_paths, correction, include_left, include_right)

    run_model(X_train, y_train, epochs, batch_size, verbose, model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Trainer')
    parser.add_argument("-d", "--data", type=str, nargs='+', help="List of directories of training data", default="")
    parser.add_argument("-c", "--correction", type=float, help="Steering correction for left/right", default=0.2)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train for", default=32)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size to use", default=64)
    parser.add_argument("-s", "--model_name", type=str, help="Save model with this name", default="model.h5")
    parser.add_argument("-v", "--verbose", type=int, help="Verbosity for logging", default=1)
    parser.add_argument("-x", "--exclude_lr", help="Exclude left and right data", action='store_false')
    args = parser.parse_args()

    if args.data != "":
        include_left = args.exclude_lr
        include_right = args.exclude_lr
        run(args.data, include_left, include_right, args.correction, args.epochs, args.batch_size, 
            args.model_name, args.verbose)





