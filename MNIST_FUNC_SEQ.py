#!/usr/bin/env python
# coding: utf-8

# # BLOG ON CNN .
# explains different ways of training our data using different method

# In[34]:


# Load the TensorBoard notebook extension

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/ ')


# In[35]:


import tensorflow as tf
import datetime


# # Notes
# --matplotlib inline
# If you want to have all of your figures embedded in your session, instead of calling display(), you can specify --matplotlib inline when you start the console, and each time you make a plot, it will show up in your document, as if you had called display(fig)().
# 
# The inline backend can use either SVG or PNG figures (PNG being the default). It also supports the special key 'retina', which is 2x PNG for high-DPI displays. To switch between them, set the InlineBackend.figure_format configurable in a config file, or via the %config magic:

# In[36]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[37]:


import tensorflow.keras.datasets.mnist as mnist
import tensorflow
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt


import datetime
import numpy as np
from random import randint


# # prepare the mnist dataset

# In[133]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=np.expand_dims(x_train, 3)
x_test=np.expand_dims(x_test, 3)


# In[134]:


print (x_train.shape)
print (type(x_train))


# In[ ]:





# In[135]:


# Normalize the training and test data, helps the training algorithm
x_train = x_train / 255.0
x_test = x_test / 255.0


# In[136]:


#Visualize the mnist dataset 
for _ in range(10):
    value=randint(0,10)
    image=x_train[value]

    fig=plt.figure
    plt.imshow(image,cmap='gray')
    plt.show()


# # Note
# we can always convert an array to an image and image to an array. For that we can use different types of library according to our task. For detailed information follow the link "https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays"

# # Create Keras models
# 
# There are two ways to build Keras models : Sequential & Functional (API). In this tutorial we will go through both implementation. For more details refer to the link "https://jovianlin.io/keras-models-sequential-vs-functional/"

# # Sequential API : A linear Stack of layers
# I expect you to have some basic knowledge about Neural network layers(CNN)
# 

# In[44]:


def create_model_Sequential():
    model=models.Sequential([
        k.layers.Conv2D(filters=24, kernel_size=(3,3), input_shape=(28, 28,1), activation='relu'),
        k.layers.MaxPool2D(pool_size=(2,2)),
        
        k.layers.Conv2D(filters=24, kernel_size=(3,3), activation='relu'),
        k.layers.MaxPool2D(pool_size=(2,2)),
        
        #**you can add more layers if you want
        #.
        #.
        #.
        #
        k.layers.Flatten(),
        k.layers.Dense(512,activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(10, activation='softmax')
        
    ])
    return model


# In[45]:


model_sequential=create_model_Sequential()
model_sequential.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[29]:


model_sequential.summary()


# In[46]:


model_sequential.fit(x_train,y_train,epochs=5)


# In[52]:


model_sequential.evaluate(x_train,y_train)


# # Functional API : Models with non-linear topology

# In[53]:


inputs=k.Input(shape=(28,28,1))
inputs.shape


# In[48]:


conv2D = layers.Conv2D(filters=24, kernel_size=(3,3),activation="relu")(inputs)

Maxpool=k.layers.MaxPool2D(pool_size=(2,2))(conv2D)

Flatten=k.layers.Flatten()(Maxpool)

Dense=k.layers.Dense(512,activation="relu")(Flatten)

output= k.layers.Dense(10, activation='softmax')(Dense)


# In[49]:


model_functional = k.Model(inputs=inputs, outputs=output, name="mnist_model")


# In[50]:


model_functional.summary()


# In[55]:


model_functional.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[137]:


model_functional.fit(x_train,y_train,epochs=5)


# In[138]:


model_functional.evaluate(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




