# -*- coding: utf-8 -*-
"""
You can **use only these** 3rd party **packages:** `cv2, keras, matplotlib, numpy, sklearn, skimage, tensorflow`. (And `torchvision` for dowloading the dataset.)

### Description
In this assignment you have to build and train a simple car and motorbike detection model using tf.keras. To do so, first we crop the images, the classify each crop one-by-one. For architecture we will use a simple convolutional network.

## Prepare dataset

* Download the PascalVOC2012 detection dataset. Here you can find more information about the labels: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html *Hint: you don't need to load the data from scrach, you can iterate over the `data` variable, it will automatically read the image and the label.*
* To see some samples, plot 5 random images and print their corresponding bounding boxes (bboxes).
* Crop each image to be an NxN image.
* Select only those, that contains cars or motorbikes.
* Split the image into 9 overlapping regions with a crop size of `<img_size> // 2` and a stride of `<img_size> // 4`, and classify each of them into 3 categories: `0 -- background, 1 -- motorbike, 2 -- car`.
  * if there isn't any (car or motorbike) bbox that overlaps more than by 50% with the crop, then it is a background crop.
  * otherwise it is a car/motorbike crop.
* Then resize the cropped regions to 64x64.
* Split the datasets into train-val-test sets (ratio: 60-20-20), without shuffling.
* Print the size of each set and plot 5 training images and their corresponding masks.
* Normalize the datasets. Input values should be between -1.0 and 1.0. *Note: you don't have to use standardization, you can just divide them by 255.*
"""

pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html

!mkdir -p ./voc2012
!curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=19Mh6P8sXJzD_j0O2AN_StB2fbqBXJNWA" > /dev/null
!curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=19Mh6P8sXJzD_j0O2AN_StB2fbqBXJNWA" -o ./voc2012/VOCtrainval_11-May-2012.tar

# Import all the library
from google.colab.patches import cv2_imshow 
import cv2
from PIL import Image
import keras
from skimage.transform import rescale, resize, downscale_local_mean
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import skimage
import sklearn
import torchvision
import os
from torchvision.datasets import VOCDetection
from torchvision.datasets.voc import DATASET_YEAR_DICT
import sys
import random
import tensorflow as tf
from matplotlib import pyplot

# download the data
data = torchvision.datasets.VOCDetection('./voc2012/', year='2012', image_set='trainval', download=True)

# check the length of the trainval
print((type(data)))

# checking the 
o1=next(iter(data))
o1

#Define necessary functions
def show_object_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0,255,255), 2)
def show_object_name(image: np.ndarray, name: str, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 0, 0))
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

def crop_img(image=[],*args):
  list_cropped_img=[]
  list_cropped_label=[]
  image_height,image_width,_=image.shape
  # minimun=min(image_height,image_width)

  if min(image_height,image_width)%2==0:
    minimum=min(image_height,image_width)
    img=image[0:minimun,0:minimun]

  else:
    minimum=min(image_height,image_width)-1
    img=image[0:minimun,0:minimun]

  image_height,image_width,_=img.shape
  # cv2_imshow(img)
  interval = image_width//2
  stride = image_width//4


  for i in range(0, img.shape[0], stride):
      bounding_box_crop=[]

      if (abs(i-minimun))>=interval:
          for j in range(0, img.shape[1], stride):
              if (abs(j-minimun))>=interval:

                  cropped_img = img[i:i + interval, j:j + interval] 
                  bounding_box_crop.append([i,j,i + interval,j+interval])

                  # determine the coordinates of the intersection rectangle
                  x_left = max(x_min,bounding_box_crop[0][0])
                  y_top = max(y_min, bounding_box_crop[0][1])
                  x_right = min(x_max, bounding_box_crop[0][2])
                  y_bottom = min(y_max,bounding_box_crop[0][3])

                  # The intersection of two axis-aligned bounding boxes is always an
                  intersection_area = (x_right - x_left) * (y_bottom - y_top)

                  # compute the area of the bounding box and crop image
                  bb1_area = (x_max - x_min) * (y_max - y_min)
                  bb2_area = (bounding_box_crop[0][2] - bounding_box_crop[0][0]) * (bounding_box_crop[0][3] - bounding_box_crop[0][1])
                  bounding_box_crop=[]

                  # compute the intersection over union by taking the intersection
                  # area and dividing it by the sum of prediction + ground-truth
                  # areas - the interesection area
                  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
                  if iou>0.5:

                    if object_name=="car":

                      cropped_img = array_to_img(cropped_img)
                      cropped_img=cropped_img.resize((64,64))
                       
                      cropped_img=img_to_array(cropped_img)
                      

                      list_cropped_img.append(cropped_img)
                      list_cropped_label.append(np.uint8(1))
                    else:
                      if object_name=="motorbike":

                        cropped_img = array_to_img(cropped_img)
                        cropped_img=cropped_img.resize((64,64))
                        cropped_img=img_to_array(cropped_img)
             
                        list_cropped_img.append(cropped_img)
                        list_cropped_label.append(np.uint8(2))
  
                    break
                  else:
                    cropped_img = array_to_img(cropped_img)
                    cropped_img=cropped_img.resize((64,64))
                    cropped_img=img_to_array(cropped_img)
                

                    list_cropped_img.append(cropped_img)
                    list_cropped_label.append(np.uint8(0))  

                
              else:
                  continue
      else :
          break
  return list_cropped_img,list_cropped_label

print('print 5 random pictures ')

count=0
for i in randomly(range(len(data))):
  image, annotation = data[i][0], data[i][1]['annotation']
  objects = annotation['object']
  show_image = np.array(image)
  if not isinstance(objects,list):
    object_name = objects['name']
    object_bndbox = objects['bndbox']
    x_min = int(object_bndbox['xmin'])
    y_min = int(object_bndbox['ymin'])
    x_max = int(object_bndbox['xmax'])
    y_max = int(object_bndbox['ymax'])  
    show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
  else:
    for j in objects:

        object_name = j['name']
        object_bndbox = j['bndbox']
        x_min = int(object_bndbox['xmin'])
        y_min = int(object_bndbox['ymin'])
        x_max = int(object_bndbox['xmax'])
        y_max = int(object_bndbox['ymax'])
        show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
  height,width,channel=show_image.shape

  minimun=min(height,width)
  crop_img=show_image[0:minimun,0:minimun]

  cv2_imshow(crop_img)
  cv2.waitKey(0)

  count=count+1
  if count==5:
    break

import sys
import random
array_crop=[]
array_label=[]
count=0
for i in range(len(data)):
  image, annotation = data[i][0], data[i][1]['annotation']
  objects = annotation['object']
  show_image = np.array(image)
  if not isinstance(objects,list):
    object_name = objects['name']
    if object_name=="car" or object_name=="motorbike": 
      object_bndbox = objects['bndbox']
      x_min = int(object_bndbox['xmin'])
      y_min = int(object_bndbox['ymin'])
      x_max = int(object_bndbox['xmax'])
      y_max = int(object_bndbox['ymax'])
      crop_np,crop_label=crop_img(show_image,x_min,y_min,x_max,y_max,object_name)
      array_crop.append(crop_np)
      array_label.append(crop_label)
           
  else:
    for j in objects:
        object_name = j['name']
        if object_name=="car" or object_name=="motorbike":
          object_bndbox = j['bndbox']
          x_min = int(object_bndbox['xmin'])
          y_min = int(object_bndbox['ymin'])
          x_max = int(object_bndbox['xmax'])
          y_max = int(object_bndbox['ymax'])
          crop_np,crop_label=crop_img(show_image,x_min,y_min,x_max,y_max,object_name)
          array_crop=array_crop+crop_np
          array_label=array_label+crop_label

print (array_label.count(0))
print (array_label.count(1))
print (array_label.count(2))
print ((array_label))

print (array_crop)

print (array_label)

# one hot encoding labels
unique_category_count = 3
array_label = tf.one_hot(array_label, unique_category_count)
array_label=array_label.numpy()
print ((array_label))

# Train Data
total_data_length_X=int((len(array_crop))*0.6)
print (total_data_length_X)
X_train=np.stack(array_crop[0:total_data_length_X])
Y_train=np.stack(array_label[0:total_data_length_X])
print (Y_train)
# #validation
X_valid=np.stack(array_crop[total_data_length_X:total_data_length_X+5515])
Y_valid=np.stack(array_label[total_data_length_X:total_data_length_X+5515])
print (len(X_valid))
# Test data
X_test=np.stack(array_crop[total_data_length_X+5515:])
Y_test=np.stack(array_label[total_data_length_X+5515:])
print ("test_length",len(X_test))
# # for i in range(5):
# #   cv2_imshow(X_train[i])

"""## Data augmentation
  * Augment the training set using `ImageDataGenerator`. The parameters should be the following: `featurewise_center=False, featurewise_std_normalization=False, rotation_range=90., width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True`.
  * When creating the generator(s), use shuffling with a seed value of 1 and batch size of 32.
  * To validate that the augmentation is working, plot 5 original images with their corresponding transformed (augmented) images and labels.
"""

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,featurewise_std_normalization=False,rotation_range=90, width_shift_range=0.1,
    height_shift_range=0.1,rescale=1./255,zoom_range=0.2,horizontal_flip=True)

train_gen_iter=datagen.flow(X_train,Y_train,batch_size=32,shuffle=False)
for x,y in train_gen_iter:
  for i in range(len(x)):
    a=x[i]*255
    cv2_imshow(a)
    cv2_imshow(X_train[i])
    print (Y_train[i])
    if i==10:
      break
  break

train_generator_iter = datagen.flow(
        X_train,
        Y_train,
        seed=1,
        shuffle=True,
        batch_size=32)

valid_generator_iter = datagen.flow(
        X_valid,
        Y_valid,
        seed=1,
        shuffle=True,
        batch_size=32)

"""## Define the model
Define the following convolutional network:
* It has 6 conv layers with 3x3 kernes
* Filter sizes are 32, 32, 64, 64, 256, 256 (in this order)
* The 1., 3. and 5. layers are padded such that the output is the same as the input
* After every convolution comes a BatchNormalisation and ReLU activation layer
* After every second convolution comes a MaxPool layer with 2x2 pool size and a Dropout layer with rate 20%
* After the last convolutional layer add a GlobalAveragePooling2D layer and a final dense layer
* For optimizer use Adam, and add accuracy to the metrics. *Note: For multi-class classification tasks, use cross-entropy loss.*
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def classnet(input_size=(64, 64, 3)):


  model=Sequential([
          Conv2D(filters=32, kernel_size=(3,3), input_shape=(64, 64,3),padding="same", activation='relu'),
          BatchNormalization(),

          Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
          BatchNormalization(),
          MaxPool2D(pool_size=(2,2)),
          Dropout(0.2),



          Conv2D(filters=64, kernel_size=(3,3),padding="same", activation='relu'),
          BatchNormalization(),

          Conv2D(filters=64, kernel_size=(3,3),activation='relu'),
          BatchNormalization(),
          MaxPool2D(pool_size=(2,2)),
          Dropout(0.2),



          Conv2D(filters=256, kernel_size=(3,3),padding="same", activation='relu'),
          BatchNormalization(),

          Conv2D(filters=256, kernel_size=(3,3),activation='relu'),
          BatchNormalization(),
          GlobalAveragePooling2D(),
          Dropout(0.2),

          Flatten(),
          Dense(3, activation='softmax')])
  return model

model = classnet(input_size=(64, 64, 3))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer ='adam',
              metrics= ['acc'])

"""## Training and evaluation 
  * Train the model for 50 epochs and use early stopping with patience of 10.
  * Plot the training curve (train/validation loss and train/validation accuracy).
  * Evaluate the trained model on the test set and plot some (5) predictions with their corresponding GT labels.
  * Print the classification report and the multilabel confusion matrix. *Note: check out the following functions: `sklearn.metrics.classification_report` and `sklearn.metrics.multilabel_confusion_matrix`*
"""

my_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]



history=model.fit(train_generator_iter,
                    epochs=50,
                    validation_data=valid_generator_iter,
                    callbacks=my_callbacks)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(['training', 'validation'], loc='upper left')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(['training', 'validation'], loc='upper right')

plt.show()

results = model.evaluate(X_test,Y_test , batch_size=32)
print (results)

predictions = model.predict(X_test)
predictions=np.argmax(predictions, axis=1)
Y_test=np.argmax(Y_test,axis=1)
print (type(predictions))
print (Y_test)

for i in range(50,60,2):
  print ("ground truth : ",Y_test[i])
  print ("predictions :",predictions[i])
  cv2_imshow(X_test[i])

from sklearn import metrics

metrics.multilabel_confusion_matrix(Y_test,predictions)

print (metrics.classification_report(Y_test, predictions))

