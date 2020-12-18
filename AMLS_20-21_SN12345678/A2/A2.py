'''
    Applied Machine Learning Systems - ELEC0132 Assignment
    Task A2 - Emotion Classification
'''

# Importing all the relevant libraries and packages used
import pandas as pd
import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

import tensorflow as tf
import scipy.misc

# Defining the image size and batch size used to feed into the CNN model
IMG_WIDTH = 89
IMG_HEIGHT = 109
batch_size = 16


def mainA2(path1, path2):
    # Define the folders from the images and the labels are stored
    data_folder = path1#"../Datasets/dataset_AMLS_20-21/celeba/"
    test_folder = path2#"../Datasets/dataset_AMLS_20-21/celeba_test/"
    img_folder = data_folder + "img/"
    test_img_folder = test_folder + "img/"

    # Read the .csv files into a Pandas data frame structure for both the data set and the testing set
    data_labels = pd.read_csv(data_folder + "labels.csv")
    test_labels = pd.read_csv(test_folder + "labels.csv")

    # Split the data frame into three columns so that the file name, gender label and emotion label are stored separately
    data_labels2 = data_labels['\timg_name\tgender\tsmiling'].apply(lambda x: pd.Series(x.split('\t')))

    # Deleting the index column and emotion column since they are not needed for this task
    del data_labels2[0]
    del data_labels2[2]
    data_labels2.columns = ['img_name', 'smiling']

    # Replacing string values with integer values, also replace -1 label with 0 label
    data_labels2.replace(to_replace='-1', value=0, inplace=True)
    data_labels2.replace(to_replace='1', value=1, inplace=True)

    # Repeat for the test set
    test_labels2 = test_labels['\timg_name\tgender\tsmiling'].apply(lambda x: pd.Series(x.split('\t')))
    del test_labels2[0]
    del test_labels2[2]
    test_labels2.columns = ['img_name', 'smiling']
    test_labels2.replace(to_replace='-1', value=0, inplace=True)
    test_labels2.replace(to_replace='1', value=1, inplace=True)

    # Split the data set into training data and validation data using a ratio of 4:1
    data_set = data_labels2.copy()
    training_data = data_set.sample(frac=0.8, random_state=0)
    validation_data = data_set.drop(training_data.index)

    # Create numpy arrays for storing the training and validation labels
    training_labels = np.array([])
    for smile in training_data['smiling']:
        training_labels = np.append(training_labels, smile)
    validation_labels = np.array([])
    for smile in validation_data['smiling']:
        validation_labels = np.append(validation_labels, smile)

    # Loop through the file names in the data frame, extract each image and stack them into a numpy array
    training_set = np.dstack(tuple(
        np.array(cv2.resize(cv2.imread(img_folder + image, 0), (IMG_WIDTH, IMG_HEIGHT))) for image in
        training_data['img_name']))
    validation_set = np.dstack(tuple(
        np.array(cv2.resize(cv2.imread(img_folder + image, 0), (IMG_WIDTH, IMG_HEIGHT))) for image in
        validation_data['img_name']))

    # Swap the axes of the numpy data sets so that it can be fed to the model
    validation_set = np.swapaxes(np.swapaxes(validation_set, 2, 0), 2, 1)
    training_set = np.swapaxes(np.swapaxes(training_set, 2, 0), 2, 1)

    # Repeat the same for the test label and test set
    test_data = test_labels2.copy()
    test_labels = np.array([])
    for smile in test_data['smiling']:
        test_labels = np.append(test_labels, smile)

    test_set = np.dstack(tuple(
        np.array(cv2.resize(cv2.imread(test_img_folder + image, 0), (IMG_WIDTH, IMG_HEIGHT))) for image in
        test_data['img_name']))
    test_set = np.swapaxes(np.swapaxes(test_set, 2, 0), 2, 1)

    # Resize them into the shapes defined above and normalize the values so they are between 0 and 1
    training_set = training_set.reshape(4000, IMG_HEIGHT, IMG_WIDTH, 1)
    validation_set = validation_set.reshape(1000, IMG_HEIGHT, IMG_WIDTH, 1)
    test_set = test_set.reshape(1000, IMG_HEIGHT, IMG_WIDTH, 1)
    training_set, validation_set, test_set = training_set / 255.0, validation_set / 255.0, test_set / 255.0

    # Build the sequential CNN model and add the relevant layers to it, then print the model's summary
    model = Sequential()
    model.add(layers.Conv2D(10, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(layers.Conv2D(20, 3, activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()

    # Defining a stopping criterion for the model, and the optimizer with new learning rate
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001, name="Adam")
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', min_delta=0.01)

    # Compile the model using Adam optimizer and sparse categorical cross-entropy
    model.compile(optimizer=opt_adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model using the training and validation data, also add the callback
    history = model.fit(training_set, training_labels, callbacks=[es],
                        validation_data=(validation_set, validation_labels),
                        epochs=10, batch_size=16)

    # Plotting the validation and training accuracy as the training progresses
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    #plt.show()

    # Evaluate our model using the test data and print the loss and the accuracy
    test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=2)
    return test_loss, test_acc


if __name__ == "__main__":
    mainA2("../Datasets/celeba/", "../Datasets/celeba_test/")