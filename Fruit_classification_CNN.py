#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Task description
# Your task is to implement a deep learning classifier of fruit images. The dataset contains segmented images of 60 different fruits. You'll have to implement a convolutional network.
# 
# 

# In[ ]:


# Keras 2.3.1 has a bug in evalutation, downgrade it
get_ipython().run_line_magic('tensorflow_version', '1.x')
get_ipython().system('pip install -U keras==2.2.5')


# In[ ]:


# Some useful imports
import matplotlib.pyplot as plt
import numpy as np
import pickle


# ## Dataset preparations
# First download the data and extract it:

# In[ ]:


get_ipython().system('wget http://vegesm.web.elte.hu/fruits_small.zip')
get_ipython().system('unzip fruits_small.zip > /dev/null')


# This will download and extract the dataset into `/content/fruits-small`. You can inspect the files in the sidebar on the left, under the *Files* tab. The dataset contains 100x100 pixel images of fruits, grouped by classes into folders. 
# 
# Notice that the dataset does not define a validation set, you have to split it yourself. Split the training set into a training and validation set. Make sure in the validation set the classes have a similar distribution to the training set.
# 
# 

# Now that you have set up the dataset, it's time to look at some of the images. Create a function that randomly selects 4 images and prints them with the class names.

# In[ ]:


import os
from PIL import Image

from PIL import Image
MAIN_PATH='/content/fruits-small/train/'
from skimage import io
read_train_data=os.listdir('/content/fruits-small/train')
choose_random_picture=np.random.permutation(len(read_train_data))
for i in choose_random_picture[0:4]:

  path=MAIN_PATH +read_train_data[i]
  pth=os.listdir(path)

  choose_random_picture=np.random.permutation(len(pth))[0]
  image_path=path+ '/' +pth[choose_random_picture]
  img=io.imread(image_path)
  im = Image.fromarray(img)
  im = im.rotate(30)
  plt.figure(figsize=(3,3))
  plt.imshow(im)
  title=read_train_data[i]+":-"+pth[choose_random_picture]
  plt.title(title)

print(read_train_data)


  

  


# ### Splitting the dataset
# 
# Notice that the dataset does not define a validation set, you have to split it yourself. Split the training set into a training and validation set. 

# ### Preprocess the dataset
# 
# We need to augment the data, since we do not have many images per classes. Create an augmentation mechanism, data automatically does the following transformations during training:
# - flip images horizontally
# - rotates them
# - performs zooming

# In[ ]:


import keras
train_dir_path = '/content/fruits-small/train'
test_dir_path= '/content/fruits-small/test' 
datagen_train_valid = keras.preprocessing.image.ImageDataGenerator(rotation_range=0.1,rescale=1./255,horizontal_flip=True,zoom_range=0.3,validation_split=0.15)


train_generator_iter = datagen_train_valid.flow_from_directory(
        train_dir_path,
        target_size=(100, 100),
        batch_size=15,
        class_mode='categorical',
        subset='training')
val_generator_iter= datagen_train_valid.flow_from_directory(train_dir_path,
        target_size=(100, 100),
        batch_size=15,
        class_mode='categorical',
    subset='validation')

test_images_iter = datagen_train_valid.flow_from_directory(test_dir_path,
        target_size=(100, 100),  # all images will be resized to 100x100
        batch_size=15,
        class_mode = 'categorical')


# ## Training the network
# 
# Implement and train the following architecture. It has the following layers:
# 
# - A convolutional layer with 5x5 kernel and 32 filters
# - A 2x2 MaxPooling layer
# - Two convolutional layers with 3x3 kernels and 64 filters each
# - A MaxPooling layer
# - Another 3x3 convolutional layer with 128 filters, followed by a MaxPooling layer
# - A fully connected layer of 512 units
# - A final softmax layer
# 
# All layers have ReLU activations. Train the network for 15 epochs.

# In[ ]:


import keras
from keras import layers
from keras import models
from keras import optimizers

import keras
from keras import layers
from keras import models
from keras import optimizers

fruit_model = models.Sequential()
fruit_model.add(keras.layers.Conv2D(filters=32,kernel_size=(5, 5), activation='relu',input_shape=(100, 100, 3)))
#fruit_model.add(keras.layers.BatchNormalization())

fruit_model.add(keras.layers.MaxPooling2D((2, 2)))
fruit_model.add(keras.layers.Conv2D(filters=64,kernel_size=(3, 3), activation='relu'))
#fruit_model.add(keras.layers.BatchNormalization())

fruit_model.add(keras.layers.Conv2D(filters=64,kernel_size=(3, 3), activation='relu'))

fruit_model.add(keras.layers.MaxPooling2D(2,2))
fruit_model.add(keras.layers.Conv2D(filters=128,kernel_size=(3, 3), activation='relu'))
#fruit_model.add(keras.layers.BatchNormalization())

fruit_model.add(keras.layers.MaxPooling2D(2,2))
fruit_model.add(keras.layers.Flatten())
fruit_model.add(keras.layers.Dense(512, activation='relu'))
fruit_model.add(keras.layers.Dense(60, activation='softmax'))
 
fruit_model.summary()


# In[ ]:


fruit_model.compile(loss='categorical_crossentropy',
              optimizer = keras.optimizers.RMSprop(lr = 1e-4, decay = 1e-6),
              metrics= ['acc'])


# In[ ]:


history = fruit_model.fit_generator(train_generator_iter,
                          epochs=15,
                          validation_data = val_generator_iter,
                          verbose=1,callbacks = [
#early stopping in case the loss stops decreasing
keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
])


# Now, that the model has finished training, plot the accuracy and loss over time, both for training and validation data:

# In[ ]:


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


# How did the loss and accuracy curves change over time? What does it mean regarding the training process (i.e. overfit, underfit, etc.)? Is that a problem and how would you solve it?
# 
# **Write your answers below**
# 
# Notice the training loss(Training and validation loss graph) decreases with each epoch and the training accuracy (Training and validation accuracy graph) increases with each epoch.This isn’t the case for the validation loss and accuracy — they seem to peak after about 5-6 epochs. This is an example of overfitting.The model performs better on the training data than it does on data it has never seen before.
# 
# For this particular case, we could prevent overfitting by simply stopping the training after 5-6 epoches.Apply regularization method, increasing the traing data set and decreasing the nural network layers

# Finally, calculate the performance of your model on the test set:

# In[ ]:


# ADD YOUR CODE HERE
pred=fruit_model.predict_generator(test_images_iter,
                                   steps=500, 
                                   callbacks=None,
                                   max_queue_size=10,
                                   workers=1,
                                   use_multiprocessing=False, 
                                   verbose=0)
                                                  


# In[ ]:


loss,acc = fruit_model.evaluate_generator(generator=test_images_iter,steps=500,use_multiprocessing=False)
print('test loss: {}'.format(loss))
print('test accuracy: {:.2%}'.format(acc))


# In[ ]:


#PICKING 5 DATA IMAGES FROM TEST SET AND CHECKINH=G THE TEST ACCURACY
List=['/content/fruits-small/test/Tamarillo/22_100.jpg','/content/fruits-small/test/Plum/38_100.jpg','/content/fruits-small/test/Potato White/109_100.jpg','/content/fruits-small/test/Pineapple Mini/122_100.jpg']

for i in List:

  loaded_image = keras.preprocessing.image.load_img(path=i, target_size=(100,100,3))
  img_array = keras.preprocessing.image.img_to_array(loaded_image) / 255

  imag_array = np.expand_dims(img_array, axis = 0)

  predictions = fruit_model.predict(imag_array)

  classidx = np.argmax(predictions[0])

  label = list(train_generator_iter.class_indices.keys())[classidx]

  pred= ["{:.2f}%".format(j * 100) for j in predictions[0] ]

  print((label, classidx, pred[classidx])) 

  plt.figure(figsize=(3,3))
  plt.imshow(img_array)
  #title=a[i]+": "+b[ran]
  #plt.title(title)
  plt.xticks([])
  plt.yticks([])
  plt.show()

