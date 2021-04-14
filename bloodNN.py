print('numpy'); import numpy as np
print('os'); import os
print('cv2'); import cv2
print('random'); import random
print('pyplot'); import matplotlib.pyplot as plt
print('pickle'); import pickle
print('Image'); from PIL import Image
print('tensorflow'); import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

print('\nImports Sucessfull')

def grabRGB(path):
    return np.array(Image.open(path).resize((96, 96)))/255.0

print('\nFunctions Read')

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\CWSF\\BloodData\\dataset2-master\\dataset2-master\\images\\TRAIN\\'

eosI, lymI, monI, neuI = [], [], [], []
eosO, lymO, monO, neuO = [], [], [], []

print('EOSINOPHIL')
for file in os.listdir(f'{path}EOSINOPHIL'):
    eosI.append(grabRGB(f'{path}EOSINOPHIL\\{file}'))
    eosO.append([1,0,0,0])

print('LYMPHOCYTE')
for file in os.listdir(f'{path}LYMPHOCYTE'):
    lymI.append(grabRGB(f'{path}LYMPHOCYTE\\{file}'))
    lymO.append([0,1,0,0])

print('MONOCYTE')
for file in os.listdir(f'{path}MONOCYTE'):
    monI.append(grabRGB(f'{path}MONOCYTE\\{file}'))
    monO.append([0,0,1,0])

print('NEUTROPHIL')
for file in os.listdir(f'{path}NEUTROPHIL'):
    neuI.append(grabRGB(f'{path}NEUTROPHIL\\{file}'))
    neuO.append([0,0,0,1])

X = np.array(eosI + lymI + monI + neuI)
Y = np.array(eosO + lymO + monO + neuO)

zipped = list(zip(X, Y))
random.shuffle(zipped)
X, Y = zip(*zipped)

print('\nData Formatted')

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
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\nModel Created')

history = model.fit(np.array(X), np.array(Y), epochs=10)
model.save('BloodNN.model')

print('\nModel Trained')

plt.plot(history.history['accuracy'], label='Blood NN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\CWSF\\BloodData\\dataset2-master\\dataset2-master\\images\\TEST\\'

eosI, lymI, monI, neuI = [], [], [], []
eosO, lymO, monO, neuO = [], [], [], []

print('EOSINOPHIL')
for file in os.listdir(f'{path}EOSINOPHIL'):
    eosI.append(grabRGB(f'{path}EOSINOPHIL\\{file}'))
    eosO.append([1,0,0,0])

print('LYMPHOCYTE')
for file in os.listdir(f'{path}LYMPHOCYTE'):
    lymI.append(grabRGB(f'{path}LYMPHOCYTE\\{file}'))
    lymO.append([0,1,0,0])

print('MONOCYTE')
for file in os.listdir(f'{path}MONOCYTE'):
    monI.append(grabRGB(f'{path}MONOCYTE\\{file}'))
    monO.append([0,0,1,0])

print('NEUTROPHIL')
for file in os.listdir(f'{path}NEUTROPHIL'):
    neuI.append(grabRGB(f'{path}NEUTROPHIL\\{file}'))
    neuO.append([0,0,0,1])

X = np.array(eosI + lymI + monI + neuI)
Y = np.array(eosO + lymO + monO + neuO)

zipped = list(zip(X, Y))
random.shuffle(zipped)
X, Y = zip(*zipped)

print('\nTesting Ready')

from random import randrange
from time import sleep
from IPython.display import clear_output

def show():
    categories = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    model = tf.keras.models.load_model('BloodNN.model')

    curr, score = 1, 0
    while curr != 100:
        loc = randrange(0, len(Y))
        inputs, outputs = X[loc], Y[loc]

        pred = model.predict(np.array([inputs]))

        guess = categories[np.argmax(pred)]
        real = categories[np.argmax(outputs)]

        print('   PREDICTION :', guess)
        print('   REAL       :', real)
        print('\n   EOSINOPHIL :', f'{int(pred[0][0]*100)}%')
        print('   LYMPHOCYTE :', f'{int(pred[0][1]*100)}%')
        print('   MONOCYTE   :', f'{int(pred[0][2]*100)}%')
        print('   NEUTROPHIL :', f'{int(pred[0][3]*100)}%')

        score += guess==real
        print('\n   ACCURACY   :', f'{int(score/curr*100)}%')

        plt.imshow(inputs)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        clear_output(wait = True)

        curr += 1

    print('Final Accuracy:', f'{int(score/curr*100)}%')
    print('\nTesting Complete')

show()
