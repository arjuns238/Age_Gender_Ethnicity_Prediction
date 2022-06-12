#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import json
import sys
import tensorflow as tf
from keras.layers import Input
import numpy as np
import argparse
#from keras_applications.resnext import ResNeXt50
from keras.utils.data_utils import get_file
#import face_recognition
from tensorflow.keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# In[ ]:


data = pd.read_csv("age_gender.csv")
df = pd.DataFrame(data)
trainingExamples = df.shape[0]

X = np.zeros(shape = (trainingExamples, 48, 48))

#Splitting the data into 48x48 pictures 
df['pixels'] = df.pixels.apply(lambda x: np.reshape(np.array(x.split(), dtype = 'float32'), (48, 48)))

#Normalizing the data
df['pixels'] = df['pixels']/255
for i in range (df.shape[0]):
    X[i] = df['pixels'][i]
X = np.asarray(X).astype('float32')
print("Type of X = ", X.dtype)
print(X[0])

#converting age into different brackets  - 0-20, 21-40, 41-60,61-80,80+
age_temp = np.zeros(df.shape[0])
for i in range(0, df.shape[0]):
    if df['age'][i] <= 20:
        val = 0
    elif df['age'][i] <= 40:
        val = 1
    elif df['age'][i] <= 60:
        val = 2
    elif df['age'][i] <= 80:
        val = 3
    else:
        val = 4
    age_temp[i] = val



age_temp = np.array(age_temp)
age = to_categorical(age_temp, num_classes = 5)
age = age.astype('float32')

#gender and ethnicity
gender = np.array(df['gender'])
#gender = to_categorical(gender)
gender = gender.astype('int64')

ethnicity = np.array(df['ethnicity'])
ethnicity = to_categorical(ethnicity)
ethnicity = ethnicity.astype('float32')




# In[ ]:


#Splitting data into train, test, and cross validation 
import tensorflow as tf
from sklearn.model_selection import train_test_split
#First, we split the data into train and remainder

def splitData (in1, output):
    X_train, X_rem, Y_train, Y_rem = train_test_split(in1, output, train_size = 0.6, shuffle = True)


    #Next we split the remainder data into test and cross validation
    X_test, X_cv, Y_test, Y_cv = train_test_split(X_rem, Y_rem, train_size = 0.5, shuffle = True)
    
    return X_train, X_test, X_cv, Y_train, Y_test, Y_cv

X_train_age, X_test_age, X_cv_age, age_train, age_test, age_cv = splitData(X, age)
X_train_gender, X_test_gender, X_cv_gender, gender_train, gender_test, gender_cv = splitData(X, gender)
X_train_ethnicity, X_test_ethnicity, X_cv_ethnicity, ethnicity_train, ethnicity_test, ethnicity_cv = splitData(X, ethnicity)


print(len(X_train_gender))


# In[ ]:


#Gender Model

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def convolution(input_size, filters):
    x = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', strides = (1,1), kernel_regularizer = l2(0.001))(input_size)
    x = Activation('relu')(x)
    return x
def model(input_shape, output_shape, loss_function):
#Converting the input into a tensor    
    inputs = Input((input_shape))
    #conv1
    conv_1 = convolution(inputs, 64)
    maxp_1 = MaxPooling2D(pool_size = (2,2))(conv_1)
    #Conv2
    conv_2 = convolution(maxp_1, 128)
    maxp_2 = MaxPooling2D(pool_size = (2, 2))(conv_2)
    #Conv3
    conv_3 = convolution(maxp_2, 256)
    maxp_3 = MaxPooling2D(pool_size = (2, 2))(conv_3)
    
    flatten = Flatten()(maxp_3)
    #Fully connected layer 1
    dense_1 = Dense(128, activation = 'relu')(flatten)
    #Fully connected layer 2
    dense_2 = Dense(64, activation = 'relu')(dense_1)    
    #output layer 
    output = Dense(output_shape, activation = 'sigmoid')(dense_2)    
    
    model = Model(inputs = [inputs], outputs = output)
    model.compile(loss = loss_function, optimizer = "Adam", metrics = ["Accuracy"])
    return model

gender_model = model((48, 48, 1), 1, 'binary_crossentropy')
gender_model.summary()
trained_gender_model = gender_model.fit(X_train_gender, gender_train, batch_size = 32, validation_data=(X_cv_gender, gender_cv), epochs = 20)


# In[ ]:


gender_model.save("gender_model_new")



# In[ ]:


#Age Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def convolution(input_size, filters):
    x = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', strides = (1,1), kernel_regularizer = l2(0.001))(input_size)
    x = Activation('relu')(x)
    return x
def model(input_shape, output_shape, loss_function):
#Converting the input into a tensor    
    inputs = Input((input_shape))
    
    conv_1 = convolution(inputs, 64)
    maxp_1 = MaxPooling2D(pool_size = (2,2))(conv_1)
    
    conv_2 = convolution(maxp_1, 128)
    maxp_2 = MaxPooling2D(pool_size = (2, 2))(conv_2)

    conv_3 = convolution(maxp_2, 256)
    maxp_3 = MaxPooling2D(pool_size = (2, 2))(conv_3)

    flatten = Flatten()(maxp_3)
    
    dense_1 = Dense(128, activation = 'relu')(flatten)

    dense_2 = Dense(64, activation = 'relu')(dense_1)    
    
    output = Dense(output_shape, activation = 'softmax')(dense_2)    
    
    model = Model(inputs = [inputs], outputs = output)
    model.compile(loss = loss_function, optimizer = "Adam", metrics = ["Accuracy"])
    return model

#change to ethnicity model cuz this is ethnicity model
age_model = model((48, 48, 1), 5, 'mse')
age_model.summary()
trained_age_model = age_model.fit(X_train_age, age_train, batch_size = 32, validation_data=(X_cv_age, age_cv), epochs = 40)


# In[ ]:


age_model.save("age_model_new")


# In[ ]:


#Age Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def convolution(input_size, filters):
    x = Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same', strides = (1,1), kernel_regularizer = l2(0.001))(input_size)
    x = Activation('relu')(x)
    return x
def model(input_shape, output_shape, loss_function):
#Converting the input into a tensor    
    inputs = Input((input_shape))
    
    conv_1 = convolution(inputs, 64)
    maxp_1 = MaxPooling2D(pool_size = (2,2))(conv_1)
    
    conv_2 = convolution(maxp_1, 128)
    maxp_2 = MaxPooling2D(pool_size = (2, 2))(conv_2)

    conv_3 = convolution(maxp_2, 256)
    maxp_3 = MaxPooling2D(pool_size = (2, 2))(conv_3)

    flatten = Flatten()(maxp_3)
    
    dense_1 = Dense(128, activation = 'relu')(flatten)

    dense_2 = Dense(64, activation = 'relu')(dense_1)    
    
    output = Dense(output_shape, activation = 'softmax')(dense_2)    
    
    model = Model(inputs = [inputs], outputs = output)
    model.compile(loss = loss_function, optimizer = "Adam", metrics = ["Accuracy"])
    return model

#change to ethnicity model cuz this is ethnicity model
ethnicity_model = model((48, 48, 1), 5, 'mse')
ethnicity_model.summary()
trained_ethnicity_model = ethnicity_model.fit(X_train_ethnicity, ethnicity_train, batch_size = 32, validation_data=(X_cv_ethnicity, ethnicity_cv), epochs = 40)


# In[ ]:


ethnicity_model.save("ethnicity_model_new")


# In[59]:


#precision, recall, and F-score
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
gender_model = load_model("gender_model_new")
plt.imshow(X_test_gender[640])
print(X_test_gender[640].shape)
bleh = X_test_gender[640].flatten().reshape(1, 48, 48)
print(X_test_gender[640])
pred = gender_model.predict(bleh)

threshold = 0.5
print(gender_test[636:669])
if pred > 0.5:
    print(1)
else:
    print (0)
print(bleh)
#precision = true positives/predicted positives
#recall = true positives/Actual postives

#pred = [1 if  i > threshold else 0 for i in pred]
#truepos = 0
#predpos = 0
#actualpos = 0
#for i in range(0, len(gender_test)):
#    if gender_test[i] == 1 and pred[i] == 1:
#        truepos += 1
#        
#    if pred[i] == 1:
#        predpos += 1
#    if pred[i] == 1:
#        actualpos += 1
#precison = truepos/predpos
#recall = truepos/actualpos

#F-score = 2PR/(P+R) ; Higher the F-Score, better the model

#Fscore = (2*precision*recall)/(precision + recall)
#print(Fscore)      

