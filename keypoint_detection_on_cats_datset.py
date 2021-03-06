"""

You can **use only these** 3rd party **packages:** `cv2, keras, matplotlib, numpy, pandas, sklearn, skimage, tensorflow`.

### Description
In this assignment you have to build and train a cat eyes and nose keypoint detector model using tf.keras. We will use an autoencoder like architecture, which first encodes the data, then decodes it to its original size. To implement such kind of models, you should take a look at the following classes and methods: `Funcitonal API, MaxPooling2D, Conv2DTranspose`.

## Prepare dataset

* Download the Cats dataset. We will only use a subset of the original dataset, the CAT_00 folder. Here you can find more information about the dataset: https://www.kaggle.com/crawford/cat-dataset
* Preprocess the data. You can find some help here: https://github.com/kairess/cat_hipsterizer/blob/master/preprocess.py
  * Following the steps in the link above, read and resize the images to be 128x128.
  * Keep only the left eye, right eye and nose coordinates.
  * Create a keypoint heatmap from the keypoints. A 128x128x3 channel image, where the first channel corresponds to left eye heatmap, the sencond channel corresponds to the right eye heatmap and the third channel corresponds to the nose heatmap. To do this:
    1. At each keypoint, draw a circle with its corresponding color. For this, use the following method: `cv2.circle(<heatmap>, center=<keypoint_coord>, radius=2, color=<keypoint_color>, thickness=2)`
    2. Then smooth the heatmap with a 5x5 Gauss filter: `<heatmap> = cv2.GaussianBlur(<heatmap>, (5,5), 0)` 
  * Then crop each image and heatmap:
    1. Define the bounding box, select the min and max keypoint coordinates: `x1, y1, x2, y2`.
    2. Add a 20x20 border around it: `x1, y1, x2, y2 = x1-20, y1-20, x2+20, y2+20`.
    3. Then crop the image and the heatmap using the extended bounding box.  
  * Finally, resize the images and the heatmaps to be 64x64.
* Split the datasets into train-val-test sets (ratio: 60-20-20), without shuffling.
* Print the size of each set and plot 5 training images and their corresponding masks.
* Normalize the datasets. All value should be between 0.0 and 1.0. *Note: you don't have to use standardization, you can just divide them by 255.*
"""

!curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1wGwNi8t-UKAKs-LQL3dG-D8dzGVPHv2w" > /dev/null
!curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1wGwNi8t-UKAKs-LQL3dG-D8dzGVPHv2w" -o CAT_00.zip

!unzip CAT_00.zip

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
!pwd

import random
import dlib, cv2, os
import sys
from google.colab.patches import cv2_imshow
im_size = 128
dirname = 'CAT_00'
base_path = '/content/CAT_00' 
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list)

dataset = {
  'imgs': [],
  'heatmaps': [],
  'lmks': [],
  # 'bbs': []
}

def resize_img(im, img_size=128):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left

for f in file_list:
  if '.cat' not in f:
    continue

  #read landmarks
  pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
  landmarks = (pd_frame.values[0][1:-1]).reshape(-1, 2)[:3]
 
  # load image
  img_filename, ext = os.path.splitext(f)
  img = cv2.imread(os.path.join(base_path, img_filename))

  # resize image and relocate landmarks
  img, ratio, top, left = resize_img(img)
  landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.uint8)
  bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

  c1, r1 = bb[0][0] - 20, bb[0][1] - 20
  c2, r2 = bb[1][0] + 20, bb[1][1] + 20

  if c1 < 0 :
    c1 = 0
  if r1 < 0 :
    r1 = 0
  if c2 >= im_size :
    c2 = im_size - 1
  if r2 >= im_size :
    r2 = im_size - 1

  heatmap = np.zeros((128, 128, 3), dtype=np.uint8)
  lm = landmarks.flatten()
  cv2.circle(heatmap, center=(lm[0], lm[1]), radius=2, color=(255, 0, 0), thickness=2)
  cv2.circle(heatmap, center=(lm[2], lm[3]), radius=2, color=(0, 255, 0), thickness=2)
  cv2.circle(heatmap, center=(lm[4], lm[5]), radius=2, color=(0, 0, 255), thickness=2)

  heatmap = cv2.GaussianBlur(cv2.resize(heatmap[r1:r2, c1:c2], (64, 64)), (5, 5), 0)
  img = resize_img(img[r1:r2, c1:c2], 64)[0]

  # cv2_imshow(img)
  # cv2_imshow(heatmap)

  dataset['imgs'].append(img)
  dataset['heatmaps'].append(heatmap)
  dataset['lmks'].append(landmarks.flatten())
  # dataset['bbs'].append(bb.flatten())

  # for l in landmarks:
  #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

  # cv2_imshow(img)
  # if cv2.waitKey(0) ==0xFF & ord('q'):
  #    break

# np.save('dataset/%s.npy' % dirname, np.array(dataset))
print('Dataset has been successfully resized and scaled.')

train_x, test_x, train_y, test_y = train_test_split(np.stack(dataset['imgs']), np.stack(dataset['heatmaps']), train_size=0.8, shuffle=False)
print("train_x size:", len(train_x), "\ttrain_y size:", len(train_y),
      "\ntest_x size:", len(test_x), "\ttest_y size:", len(test_y), "\n")

for i in range(5):
  cv2_imshow(train_x[i])
  cv2_imshow(train_y[i])

train_x = train_x / 255.0
train_y = train_y / 255.0
test_x = test_x / 255.0
test_y = test_y / 255.0

"""## Data augmentation
  * Augment the training set using `ImageDataGenerator`. The parameters should be the following: `featurewise_center=False, featurewise_std_normalization=False, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2`.
  * When creating the generator(s), use shuffling with a seed value of 1 and batch size of 32.
  * To validate that the augmentation is working, plot 5 original images with their corresponding transformed (augmented) images and masks.

**Keep in mind:** To augment the inputs and targets the same way, you should create 2 separate generator, then you can zip them together.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, validation_split=0.25,
                                     width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

trial_batch = generator.flow(train_x[:5], train_y[:5], shuffle=False)
for i in range(5):
  cv2_imshow(train_x[i] * 255)
  cv2_imshow(trial_batch[0][0][i] * 255)
  cv2_imshow(trial_batch[0][1][i] * 255)

"""## Define the model
Define the following architecture in tf.keras:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 64, 64, 3)]       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 64, 64)        1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 64, 64)        36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 32, 32, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 32, 32, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 32, 32, 128)       147584    
_________________________________________________________________
bottleneck_1 (Conv2D)        (None, 32, 32, 160)       5243040   
_________________________________________________________________
bottleneck_2 (Conv2D)        (None, 32, 32, 160)       25760     
_________________________________________________________________
upsample_1 (Conv2DTranspose) (None, 64, 64, 3)         1920      
=================================================================
Total params: 5,530,880
Trainable params: 5,530,880
Non-trainable params: 0
_________________________________________________________________
```
* Use relu and `padding='same'` for each layer.
* Use a 2x2 `Conv2DTranspose` layer without bias to upsample the result. 
* `block1_conv1`, `block1_conv2`, `block2_conv1` and `block2_conv2` are 3x3 convolutions.
* `bottleneck_1` is a 16x16 and `bottleneck_2` is a 1x1 convolution.
* For optimizer use RMSProp, and MSE as loss function.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

def fcnet(input_size=(64, 64, 3)):
  input = tf.keras.Input(shape=input_size)
  block1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
  block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1_conv1)
  block1_pool = MaxPooling2D((2, 2), padding='same')(block1_conv2)
  block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(block1_pool)
  block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(block2_conv1)
  bottleneck_1 = Conv2D(160, (16, 16), activation='relu', padding='same')(block2_conv2)
  bottleneck_2 = Conv2D(160, (1, 1), activation='relu', padding='same')(bottleneck_1)

  # output = UpSampling2D(3, (2, 2), activation='relu', padding='same', use_bias=False)(bottleneck_2)
  upsampling = UpSampling2D(input_shape=input_size, interpolation='bilinear')(bottleneck_2)
  output = Conv2DTranspose(3, (2, 2), activation='relu', padding='same', use_bias=False)(upsampling)

  return tf.keras.Model(input, output, name='autodecoder')

model = fcnet(input_size=(64, 64, 3))
model.summary()

"""## Training and evaluation 
  * Train the model for 30 epochs without early stopping.
  * Plot the training curve (train/validation loss).
  * Evaluate the trained model on the test set.
  * Plot some (5) predictions with their corresponding GT heatmap and input. You should mark the location of each keypoint on the image. *Note: it might be worth to take a look at this answer: https://stackoverflow.com/a/17386204*
"""

train_batch = generator.flow(train_x, train_y, seed=1, subset='training')
valid_batch = generator.flow(train_x, train_y, seed=1, subset='validation')
test_batch = generator.flow(test_x, test_y, seed=1)

model.compile(
    optimizer='rmsprop', 
    loss='mean_squared_error', 
    metrics=['accuracy'])
history = model.fit(train_batch, epochs=30, verbose=1, validation_data=valid_batch)
model.evaluate(test_batch, verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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

output = model.evaluate(test_x, test_y)
print(output)

output = model.evaluate(train_x, train_y)
images = model.predict(train_x[:10])
for i in range(len(images)):
  cv2_imshow(train_y[i] * 255)
  cv2_imshow(images[i] * 255)
