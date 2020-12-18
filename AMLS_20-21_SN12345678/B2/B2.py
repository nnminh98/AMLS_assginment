'''
    Applied Machine Learning Systems - ELEC0132 Assignment
    Task A1 - Gender Classification
'''

# Importing all the relevant libraries and packages used
import pandas as pd
import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
#import seaborn as sns
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
from tqdm import tqdm_notebook as tqdm

# Defining the image size and batch size used to feed into the CNN model
IMG_WIDTH = 100
IMG_HEIGHT = 100
batch_size = 4


def mainB2(path1, path2):
    # Define the folders from the images and the labels are stored
    data_folder = path1#"../Datasets/dataset_AMLS_20-21/cartoon_set/"
    test_folder = path2#"../Datasets/dataset_AMLS_20-21/cartoon_set_test/"
    img_folder = data_folder + "img/"
    test_img_folder = test_folder + "img/"

    # Read the .csv files into a Pandas data frame structure for both the data set and the testing set
    data_labels = pd.read_csv(data_folder + "labels.csv")
    test_labels = pd.read_csv(test_folder + "labels.csv")

    # Split the data frame into three columns so that the eye color, face shape and file name are stored separately
    data_labels2 = data_labels['\teye_color\tface_shape\tfile_name'].apply(lambda x: pd.Series(x.split('\t')))

    # Deleting the index column and eye color column since they are not needed for this task
    del data_labels2[0]
    del data_labels2[2]
    data_labels2.columns = ['eye_color', 'file_name']

    # Repeat for the test set
    test_labels2 = test_labels['\teye_color\tface_shape\tfile_name'].apply(lambda x: pd.Series(x.split('\t')))
    del test_labels2[0]
    del test_labels2[2]
    test_labels2.columns = ['eye_color', 'file_name']

    # Split the data set into training data and validation data using a ratio of 4:1
    data_set = data_labels2.copy()
    test_data = test_labels2.copy()
    training_data = data_set.sample(frac=0.8, random_state=0)
    validation_data = data_set.drop(training_data.index)

    # Create numpy arrays for storing the training, validation and test labels
    training_labels = np.array([])
    for eye in training_data['eye_color']:
        training_labels = np.append(training_labels, eye)

    validation_labels = np.array([])
    for eye in validation_data['eye_color']:
        validation_labels = np.append(validation_labels, eye)

    test_labels = np.array([])
    for eye in test_data['eye_color']:
        test_labels = np.append(test_labels, eye)

    # Loop through the file names in the data frame, extract each image and append them to a list
    training_set = []
    for image in training_data['file_name']:
        training_set.append(cv2.resize(cv2.imread(img_folder + image, 1), (IMG_WIDTH, IMG_HEIGHT)))

    validation_set = []
    for image in validation_data['file_name']:
        validation_set.append(cv2.resize(cv2.imread(img_folder + image, 1), (IMG_WIDTH, IMG_HEIGHT)))

    test_set = []
    for image in test_data['file_name']:
        test_set.append(cv2.resize(cv2.imread(test_img_folder + image, 1), (IMG_WIDTH, IMG_HEIGHT)))

    # Convert list to numpy arrays
    test_set = np.array(test_set)
    training_set = np.array(training_set)
    validation_set = np.array(validation_set)

    # Normalize the values so they are between 0 and 1
    # Convert label data type
    training_set, test_set, validation_set = training_set / 255.0, test_set / 255.0, validation_set / 255.0
    training_labels, test_labels, validation_labels = training_labels.astype('uint8'), test_labels.astype('uint8'), \
                                                      validation_labels.astype('uint8')

    # Build the sequential CNN model and add the relevant layers to it, then print the model's summary
    model = Sequential()
    model.add(layers.Conv2D(18, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.Conv2D(36, 5, activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.summary()

    # Creating Adam optimizer with adjusted learning rate
    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.0001, name="Adam")

    # Defining a stopping criterion for the model
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', min_delta=0.01)

    # Compile the model using Adam optimizer and sparse categorical cross-entropy
    model.compile(optimizer=opt_adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model using the training and validation data, also add the callback
    history = model.fit(training_set, training_labels, callbacks=[es],
                        validation_data=(validation_set, validation_labels),
                        epochs=20, batch_size=16)

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
    mainB2("../Datasets/cartoon_set/", "../Datasets/cartoon_set_test/")


