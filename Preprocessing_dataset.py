#####----------------------------------Preprocessing Dataset -----------------------

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import gdal
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

traindirectory = "/content/drive/MyDrive/PhD_Thesis_Data/DEEP LEARNING BASED PATCH-WISE LAND COVER LAND USE CLASSIFICATION: A NEW SMALL BENCHMARK SENTINEL-2 IMAGE DATASET/CLCL2DATASET_7Class"
validdirectory = "/content/drive/MyDrive/PhD_Thesis_Data/DEEP LEARNING BASED PATCH-WISE LAND COVER LAND USE CLASSIFICATION: A NEW SMALL BENCHMARK SENTINEL-2 IMAGE DATASET/CLCL2DATASET_Test_7Class_png"

traindatagen = ImageDataGenerator(rescale=1./255)
validdatagen = ImageDataGenerator(rescale=1./255) 

train_corine_ds = traindatagen.flow_from_directory(directory=traindirectory, 
                                        target_size = (100,100),
                                       color_mode ="rgb",
                                       batch_size=1,
                                       shuffle = True,
                                       subset="training",
                                       class_mode = "categorical",
                                       interpolation="bicubic")

valid_corine_ds = validdatagen.flow_from_directory(directory=validdirectory,
                                                  target_size=(100, 100),
                                                  color_mode="rgb",
                                                  batch_size=1,
                                                  class_mode="categorical",
                                                  shuffle=True,
                                                  interpolation="bicubic")



class_names_train = train_corine_ds.class_indices
class_names_valid = valid_corine_ds.class_indices
print(class_names_train,class_names_valid )

x_train = []
y_train = []
x_valid = []
y_valid = []

for i in range(len(train_corine_ds)):
    x_train.append(train_corine_ds[i][0])
    y_train.append(train_corine_ds[i][1])


x_train = tf.concat(x_train, axis=0)
y_train = tf.concat(y_train, axis=0)


for i in range(len(valid_corine_ds)):
    x_valid.append(valid_corine_ds[i][0])
    y_valid.append(valid_corine_ds[i][1])


x_valid = tf.concat(x_valid, axis=0)
y_valid = tf.concat(y_valid, axis=0)

