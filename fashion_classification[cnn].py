# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:35:55 2018

@author: Anil
"""
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

#importing dataset
fashion_train = pd.read_csv('fashion-mnist_train.csv')
fashion_test = pd.read_csv('fashion-mnist_test.csv')

#converting dataframe to numpy array
fashion_train = np.array(fashion_train, dtype='float32')
fashion_test = np.array(fashion_test, dtype='float32')

#separating features and labels 
features = fashion_train[0:,1:]
labels = fashion_train[0:,0]

#visualization
fig, axes = plt.subplots(5,5,figsize=(8,8))
plt.subplots_adjust(hspace=0.6)
axes = axes.ravel()
for i in range(0,25):
    r = random.randint(0,len(features))
    axes[i].imshow(features[r,0:].reshape(28,28))
    axes[i].set_title(labels[r])
    axes[i].axis('off')

#creating training and test set 
X_train = fashion_train[0:,1:]/255
y_train = fashion_train[0:,0]

X_test = fashion_test[0:,1:]/255
y_test = fashion_test[0:,0]

#converting columns in the form of 28*28 and 1 for b&w channel
X_train = X_train.reshape(X_train.shape[0],*(28,28,1))
X_test = X_test.reshape(X_test.shape[0],*(28,28,1))

#split the training data to training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train,y_validate = train_test_split(X_train,y_train,test_size = 0.2, random_state = 101)

#import keras libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import adam
from keras.models import load_model

#creating convolutional neural network
classifier = Sequential()
classifier.add(Conv2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))
classifier.compile(optimizer = adam(lr = 0.001),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train,epochs = 50, batch_size=512, verbose = 1, validation_data=(X_validate, y_validate))

#saving the trained model for future use
classifier.save('fashion_trained.h5')

#loading the model
classifier_load  = load_model('fashion_trained.hdf5')

#checking the accuracy of the model
evaluation = classifier_load.evaluate(X_test,y_test)
print('Accuracy : {}%'.format(evaluation[1] * 100))

#predicted values
y_pred = classifier_load.predict_classes(X_test)

#visualizing the predicted and actual value
fig, axes = plt.subplots(3,3,figsize=(8,8))
axes = axes.ravel()
plt.subplots_adjust(hspace=.9)
for i in range(0,9):
    r = random.randint(0,len(features))
    axes[i].imshow(features[r,0:].reshape(28,28))
    axes[i].set_title('predicted : {} \n actual : {:.0f}'.format(y_pred[i],y_test[i]))
    axes[i].axis('off')

#checking accuracy of each label
from sklearn.metrics import classification_report
classes = ['Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
print(classification_report(y_test,y_pred,target_names=classes))

