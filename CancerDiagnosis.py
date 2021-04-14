import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from IPython.display import clear_output

def grabRGB(dataPath):
	return np.array(Image.open(dataPath).resize((96, 96)))/255.0

def test(typ, dataPath):
    if typ == 'l':
        pred = 0
        model = tf.keras.models.load_model('HistoCNN.model')

        loc = os.listdir(f'{dataPath}')
        it, full = 1, len(loc)
        for file in loc:
            inputs = grabRGB(f'{dataPath}\\{file}')
            outputs = model.predict(np.array([inputs]))

            pred += outputs

            print(f'File {it}/{full} - {int(it/full*100)}%')
            clear_output(wait = True)

            it += 1

    elif typ == 'b':
        pred = np.array([0,0,0,0])
        model = tf.keras.models.load_model('BloodNN.model')

        loc = os.listdir(f'{dataPath}')
        it, full = 1, len(loc)
        for file in loc:
            inputs = grabRGB(f'{dataPath}\\{file}')
            outputs = model.predict(np.array([inputs]))

            pred = np.add(pred, outputs)

            print(f'File {it}/{full} - {int(it/full*100)}%')
            clear_output(wait = True)

            it += 1
        
    else:
        print('Invalid Entry')
        return 0, 0
    
    return pred, full

def diagnosis():
    path = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\CWSF\\TestData'

    typ = input('Lymph(l) | Blood(b): ')
    loc = input('Dataset Name: ')
    dataPath = f'{path}\\{typ}\\{loc}'

    pred, full = test(typ, dataPath)

    print('<- DETECTIONS ->\n')

    if typ == 'l':
        print('BENIGN     :', f'{int(100 - pred[0][0]/full*100)}%')
        print('MALIGNANT  :', f'{int(pred[0][0]/full*100)}%')

    else:
        print('EOSINOPHIL :', f'{int(pred[0][0]/full*100)}%')
        print('LYMPHOCYTE :', f'{int(pred[0][1]/full*100)}%')
        print('MONOCYTE   :', f'{int(pred[0][2]/full*100)}%')
        print('NEUTROPHIL :', f'{int(pred[0][3]/full*100)}%')

diagnosis()
