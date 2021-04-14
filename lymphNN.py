print('numpy'); import numpy as np
print('os'); import os
print('cv2'); import cv2
print('matplotlib.pyplot'); import matplotlib.pyplot as plt
print('pickle'); import pickle
print('Image'); from PIL import Image
print('tensorflow'); import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

print('Imports Sucessfull')

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\'
labelsFile = open(f'{path}train_labels.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

print('Hyperparameters Set')

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

print('Model Created')

def grabRGB(uuid):
    return np.array(Image.open(f'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\train\\{uuid}.tif'))
print(labels)
trainingData, size = [], len(labels)
for row in range(size):
    try:
        trainingData.append([grabRGB(labels[row][0]), int(labels[row][1])])
    except:
        pass
    
    if (row+1)%100 == 0:
        X, Y = [], []
        for features, label in trainingData:
            X.append(features)
            Y.append(label)

        model.fit(np.array(X).reshape(-1, 96, 96, 3)/255.0, np.array(Y), batch_size=1)
        model.save('HistoCNN2.model')
        
        trainingData = []
    
print('Model Trained')

def testRGB(uuid):
    image = Image.open(f'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\YRSTF2021\\Data\\test\\{uuid[0]}.tif')
    return np.array(image).reshape(-1, 96, 96, 3)/255.0, image

model = tf.keras.models.load_model('HistoCNN.model')

labelsFile = open(f'{path}sample_submission.csv')
labels = [row.split(',') for row in labelsFile.read().split('\n')][1:]
labelsFile.close()

print('Testing Ready')

from random import randrange
from IPython.display import clear_output

categories = ['Benign', 'Malignant']

curr = 0
while curr != 100:
    loc = randrange(0, len(labels))
    inputs, image = testRGB(labels[loc])

    pred = model.predict(inputs)

    print('   PREDICTION :', categories[round(pred[0][0])], '-', format(pred[0][0], '.2f'))

    plt.imshow(image)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    clear_output(wait = True)
    
    curr += 1

