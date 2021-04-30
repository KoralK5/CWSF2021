from PIL import Image
import time
import _pickle
import numpy as np
import cupy as cp
import xdNNrewrite as nn
import time

f = open('/HDD1/iantitor/Downloads/histopathologic-cancer-detection/train_labels.csv', 'r')
data = f.read().split('\n')[1:-1]
f.close()
for x, d in enumerate(data):
    a = d.split(',')
    data[x] = [a[0], np.array([int(a[1])])]
print('Dataset list initialized successfully')


def getWeights(path):
    return np.load(path, allow_pickle=True)


def grabRGB(uuid, path):
	f = Image.open(f'{path}{uuid}.tif')
	a = np.array(f)
	return np.ndarray.flatten(a)/255


def datasetFunc(index, **kwargs):
    labels, path = kwargs['labels'], kwargs['path']
    inputs, outputs = labels[index % len(labels)]
    inputs = grabRGB(inputs, path)
    return inputs, outputs

def testnn(l, w, nw):
    c, nc = 0, 0
    for x in range(l):
        i, o = datasetFunc(x, path = '/HDD1/iantitor/Downloads/histopathologic-cancer-detection/train/', labels = data)
        c += int(o == int(nn.neuralNetwork(i, w)[-1] * 2))
        nc += int(o == int(nn.neuralNetwork(i, nw)[-1] * 2))
        print(f'finished neural network #{x + 1} out of {l}\naccuracy of old neural network is {c} / {x + 1} or {c / (x + 1) * 100}%\naccuracy of new neural network is {nc} / {x + 1} or {nc / (x + 1) * 100}%')
    print(f'\nold neural network got {c} / {l} or {c / l * 100}% correct\nnew neural network got {nc} / {l} or {nc / l * 100}% correct\nnew neural network got {nc - c} more images correct')
        
weights = nn.generateWeights([27648, 64, 32, 16, 1], minimum = -1, maximum = 1)

    start = time.time()
newWeights = nn.train(datasetFunc, weights, iterLimit = len(data) * 7, path = '/HDD1/iantitor/Downloads/histopathologic-cancer-detection/train/', labels = data, costThreshold = -1, learningRate = 1, saveInterval = 44005, saveLocation = '/home/iantitor/Desktop/Weights/Weights \n')
print(time.time() - start)
