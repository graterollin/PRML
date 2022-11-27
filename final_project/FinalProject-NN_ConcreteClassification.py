# ----------------------------
# Andres Graterol 
# EEL 5678 
# 4031393
# ---------------------------
# NN solution for the Concrete Cracks classification 
# ---------------------------
import os
#import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# TODO: See if want to reduce image size or perform PCA
def data_preprocessing(negative_pics, positive_pics):
    # Class 0 => No cracks 
    # Class 1 => Cracks 

    # Before reshape => (20000, 227, 227)
    # Want => (20000, 51529, 1)
    negative_images = []

    # Iterate over the folder with no cracks  
    for picture in os.scandir(negative_pics):
        if picture.is_file():
            pic_path = negative_pics + '/' + picture.name
            image = cv2.imread(pic_path)
            # Remove 3 channels 
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Each image is (227, 227, 3)
            # Grayscale => (227, 227, 1)
            data = np.asarray(gray_image)
            data = data.reshape((227*227))
            # Add extra last column to help with labeling
            data = np.append(data, 0)
            negative_images.append(data)

    positive_images = []

    # Iterative over the folder with cracks
    for picture in os.scandir(positive_pics):
        if picture.is_file():
            pic_path = positive_pics + '/' + picture.name
            image = cv2.imread(pic_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data = np.asarray(gray_image)
            data = data.reshape((227*227))
            data = np.append(data, 1)
            positive_images.append(data)

    all_images = np.concatenate((negative_images, positive_images))

    # TODO: Seed the shuffle??
    np.random.shuffle(all_images)
    # 75 and 25 split (30000) and (10000)
    x_train, x_test = all_images[:30000], all_images[30000:]

    y_train = []
    y_test = []

    for point in x_train:
        if (point[-1] == 0):
            # OHE for class 0
            y_train.append([1, 0])
        if (point[-1] == 1):
            # OHE for class 1
            y_train.append([0, 1])

    # TODO: See if this needs to be done before reshaping!!!
    # drop the label point 
    x_train = np.delete(x_train, obj=-1, axis=1)
    x_train = x_train.astype('float32')
    # Normalize the data
    x_train /= 255
    x_train = x_train[:, :, np.newaxis]

    for point in x_test:
        if (point[-1] == 0):
            y_test.append([1, 0])
        if (point[-1] == 1):
            y_test.append([0, 1])
        
    # drop the label point 
    x_test = np.delete(x_test, obj=-1, axis=1)
    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = x_test[:, :, np.newaxis]

    return (x_train, y_train), (x_test, y_test)

def main():
    negative_pics = 'ConcreteCracks/Negative'
    positive_pics = 'ConcreteCracks/Positive'

    #(x_train, y_train), (x_test, y_test) = 
    #data_preprocessing(negative_pics, positive_pics)
    
    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_cuda())
    return None

main()