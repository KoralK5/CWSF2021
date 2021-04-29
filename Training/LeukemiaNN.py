print('numpy'); import numpy as np
print('os'); import os
print('cv2'); import cv2
print('matplotlib.pyplot'); import matplotlib.pyplot as plt
print('random'); import random
print('pandas'); import pandas as pd
print('Image'); from PIL import Image
print('tensorflow'); import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

print('\nImports Sucessfull')

model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\nModel Created')

def fix(img):
    return np.array(cv2.resize(img, (96,96)))/255.0
 
path = 'enterYourPath\\CWSF\\LeukemiaData\\C-NMC_Leukemia\\training_data'
size = 96
data = []
categories = ['hem','all']
folders = ['fold_0','fold_1','fold_2']

s = 0
for folder in folders:
    for category in categories:
        pt = f'{path}\\{folder}'
        p = f'{pt}\\{category}'
        output = int(category=='hem')
        for img in os.listdir(p):
            data.append([fix(cv2.imread(os.path.join(p,img))),output])

random.shuffle(data)

print('\nData Formatted')

acc = []
for epoch in range(5):
    newData = data
    for row in range(int(len(newData)/7)):
        X, Y = [], []
        for features, label in newData[:1523]:
            X.append(np.array(features))
            Y.append(label)
        newData = newData[1523:]
        
        try:
            history = model.fit(np.array(X), np.array(Y), batch_size=1)
            model.save('LeukNN.model')
            acc.append(history.history['accuracy'])
        except:
            pass

print('\nModel Trained')

plt.plot(acc, label='Leukemia NN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
  
