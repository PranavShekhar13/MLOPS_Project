#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LeakyReLU 
import os


# In[2]:
(X_train, y_train), (X_test, y_test) = mnist.load_data('mymnist.db')
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)



# In[23]:
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

X_train.shape


number_of_classes = 10

Y_train = utils.to_categorical(y_train, number_of_classes)
Y_test =  utils.to_categorical(y_test, number_of_classes)

y_train[0], Y_train[0]

# In[34]:

model = keras.Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(10))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))


model.summary()
# In[42]:




model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy']
              )


# In[44]:


h = model.fit(X_train, Y_train, epochs=2)


# In[64]:

test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print(test_accuracy)



# In[72]:


text = h.history
accuracy = text['acc'][1] * 100
accuracy = int(accuracy)
f = open("./accuracy.txt", "w+")
f.write(str(accuracy))
f.close()

print("Accuracy = ", accuracy , "%")


# In[ ]:
