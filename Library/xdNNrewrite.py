import numpy as np
import cupy as cp
import _pickle as pkl
from numba import jit
from datetime import datetime


def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoidD(x):
	return sigmoid(x) * (1 - sigmoid(x))


def deepcopy(x):
    return pkl.loads(pkl.dumps(x))


def cost(t, v):
    return np.sum((t - v) ** 2, axis = -1)


def karges(kwargs, check, default):
    if check in kwargs:
        return kwargs[check]
    else:
        return default


def generateWeights(layerData, **kwargs):
    minimum, maximum = karges(kwargs, 'minimum', -1), karges(kwargs, 'maximum', 1)
    return [np.random.uniform(minimum, maximum, [layerData[1:][layer], layerData[layer] + 1]) for layer in range(len(layerData) - 1)]


def layer(inputs, weights):
    return np.tensordot(np.append(inputs, np.full(list(inputs.shape)[:-1] + [1], 1), axis = -1), weights, axes = [[-1], [-1]])


def neuralNetwork(inputs, weights, **kwargs):
    actFunc, finalActFunc = karges(kwargs, 'actFunc', sigmoid), karges(kwargs, 'finalActFunc', sigmoid)
    neuronOutputs = []
    layerInputs = deepcopy(inputs)
    for layerWeights in weights[:-1]:
        layerInputs = actFunc(layer(layerInputs, layerWeights))
        neuronOutputs.append(deepcopy(layerInputs))
    neuronOutputs.append(finalActFunc(layer(layerInputs, weights[-1])))
    return neuronOutputs


def optimizer(inputs, weights, outputs, **kwargs):
    learningRate, actFunc, actFuncD = karges(kwargs, 'learningRate', 0.1), karges(kwargs, 'actFunc', sigmoid), karges(kwargs, 'actFuncD', sigmoidD)
    weightedSum = layer(inputs, weights)
    layerOutputs = actFunc(weightedSum)
    chainDerivCoef = np.sum(2 * (layerOutputs - outputs), axis = -1) * actFuncD(weightedSum)
    #chainDerivCoef = 2 * (layerOutputs - outputs) * actFuncD(weightedSum)
    inputsGrad = np.sum(weights.transpose()[:-1] * chainDerivCoef, axis = -1)
    weightsGrad = np.outer(np.append(inputs, 1), chainDerivCoef).transpose()
    newInputs = inputs - inputsGrad * learningRate
    newWeights = weights - weightsGrad * learningRate
    return newInputs, newWeights


def backProp(inputs, weights, outputs, **kwargs):
    layerInputs = neuralNetwork(inputs, weights, **kwargs)[:-1][::-1] + [inputs]
    newWeights = deepcopy(weights)[::-1]
    targetOutputs = outputs
    for layerIndex, layerWeights in enumerate(newWeights):
        targetOutputs, newWeights[layerIndex] = optimizer(layerInputs[layerIndex], layerWeights, targetOutputs, **kwargs)
    return newWeights[::-1]


def train(datasetFunc, weights, **kwargs):
    costThreshold, iterLimit, saveInterval, saveLocation = karges(kwargs, 'costThreshold', 0.1), karges(kwargs, 'iterLimit', 1000), karges(kwargs, 'saveInterval', -1), karges(kwargs, 'saveLocation', 'weights.npy')
    iterCost = 1
    newWeights = deepcopy(weights)
    for iteration in range(iterLimit):
        if iterCost <= costThreshold:
            break
        inputs, outputs = datasetFunc(iteration, **kwargs)
        newWeights = backProp(inputs, newWeights, outputs, **kwargs)
        prediction = neuralNetwork(inputs, weights, **kwargs)[-1]
        iterCost = cost(outputs, prediction)
        print(f'\n\n\nStatistics of iteration #{iteration + 1} of {iterLimit}:\n\nPrediction: {prediction}\n\nDataset Output: {outputs}\n\nCost: {iterCost}')
        if (iteration + 1) % saveInterval == 0:
            np.save(saveLocation.replace('\t', datetime.now().strftime('%Y-%m-%d-%H-%M-%S')).replace('\n', f'{(iteration + 1) - saveInterval}-{iteration + 1}'), np.array(newWeights))
            print(f'Weights for iteration #{iteration + 1} saved')
    return newWeights


if __name__ == '__main__':
    i = np.arange(2)
    o = np.array([1, 0, 1])
    o1 = np.array([1, 0, 0, 1, 1])
    w = generateWeights([2, 5, 3])
    w1 = deepcopy(w[0])
    nw = deepcopy(w1)