import _pickle as pkl
from numba import jit
from datetime import datetime


def lol(x):
    return x


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


def karges(kwargs, defaults):
    vars = deepcopy(defaults)
    vars.update(kwargs)
    return [vars[x] for x in list(defaults)]


def imageSplit(image, kernelShape):
    imageShape = list(image.shape)
    return np.moveaxis(np.array(np.split(np.moveaxis(np.array(np.split(image, imageShape[-1] / kernelShape[-1], axis = -1)), 0, -3), imageShape[-2] / kernelShape[-2], axis = -2)), 0, -4)


def imageFormat(image, kernelShape):
    imageShape = list(image.shape)
    imageDim = len(imageShape)
    formattedImage = np.full(imageShape[: -2] + [imageShape[-2] - kernelShape[-2] + 1, imageShape[-1] - kernelShape[-1] + 1] + kernelShape, 0)
    for y in range(kernelShape[-2]):
        for x in range(kernelShape[-1]):
            writeSlices = tuple([slice(None)] * (imageDim - 2) + [slice(y, None, kernelShape[-2]), slice(x, None, kernelShape[-1])])
            readSlices = tuple([slice(None)] * (imageDim - 2) + [slice(y, imageShape[-2] - (imageShape[-2] - y) % kernelShape[-2]), slice(x, imageShape[-1] - (imageShape[-1] - x) % kernelShape[-1])])
            formattedImage[writeSlices] = imageSplit(image[readSlices], kernelShape)
    return formattedImage.reshape(imageShape[: -2] + [(imageShape[-2] - kernelShape[-2] + 1), (imageShape[-1] - kernelShape[-1] + 1), kernelShape[-2] * kernelShape[-1]])


def generateWeights(layerData, **kwargs):
    minimum, maximum = karges(kwargs, {'minimum': -1, 'maximum': 1})
    return [np.random.uniform(minimum, maximum, [layerData[1:][layer], layerData[layer] + 1]) for layer in range(len(layerData) - 1)]


def generateKernels(layerData, kernelShape, **kwargs):
    minimum, maximum = karges(kwargs, {'minimum': -1, 'maximum': 1})
    inputSize = kernelShape[0] * kernelShape[1] + 1
    return [np.random.uniform(minimum, maximum, [layer, inputSize]) for layer in layerData]


def fullyConnectedLayer(inputs, weights):
    return np.tensordot(np.append(inputs, np.full(list(inputs.shape)[:-1] + [1], 1), axis = -1), weights, axes = [[-1], [-1]])


def convolutionLayer(inputs, weights, kernelShape):
    formattedInputs = imageFormat(inputs, kernelShape)
    return np.moveaxis(np.tensordot(np.append(formattedInputs, np.full(list(formattedInputs.shape)[:-1] + [1], 1), axis = -1), weights, axes = [[-1], [-1]]), -1, 0).reshape([len(formattedInputs) * len(weights), len(formattedInputs[0]), len(formattedInputs[0, 0])])


def neuralNetwork(inputs, weights, **kwargs):
    actFunc, finalActFunc, layerFuncs = karges(kwargs, {'actFunc': sigmoid, 'finalActFunc': sigmoid, 'layerTypes': [fullyConnectedLayer] * len(weights)})
    neuronOutputs = []
    layerInputs = deepcopy(inputs)
    for layerIndex, layerWeights in enumerate(weights[:-1]):
        layerInputs = actFunc(layerFuncs[layerIndex](layerInputs, layerWeights))
        neuronOutputs.append(deepcopy(layerInputs))
    neuronOutputs.append(finalActFunc(layerFuncs[layerIndex](layerInputs, weights[-1])))
    return neuronOutputs

# make fully connected and convolutional gradients
def fullyConnectedGradient(inputs, weights, outputs, **kwargs):
    actFunc, actFuncD, layerFunc = karges(kwargs, {'actFunc': sigmoid, 'actFuncD': sigmoidD, 'layerFunc': fullyConnectedLayer})
    weightedSum = layerFunc(inputs, weights)
    layerOutputs = actFunc(weightedSum)
    chainDerivCoef = np.sum(2 * (layerOutputs - outputs), axis = -1) * actFuncD(weightedSum)
    #chainDerivCoef = 2 * (layerOutputs - outputs) * actFuncD(weightedSum)
    inputsGrad = np.sum(weights.transpose()[:-1] * chainDerivCoef, axis = -1)
    weightsGrad = np.outer(np.append(inputs, 1), chainDerivCoef).transpose()
    return inputsGrad, weightsGrad


def convolutionalGradient(inputs, weights, outputs, **kwargs):
    actFunc, actFuncD, layerFunc = karges(kwargs, {'actFunc': sigmoid, 'actFuncD': sigmoidD, 'layerFunc': fullyConnectedLayer})
    weightedSum = layerFunc(inputs, weights)
    layerOutputs = actFunc(weightedSum)
    chainDerivCoef = np.sum(2 * (layerOutputs - outputs), axis = -1) * actFuncD(weightedSum)
    #chainDerivCoef = 2 * (layerOutputs - outputs) * actFuncD(weightedSum)
    inputsGrad = weights.transpose()[:-1] * chainDerivCoef
    weightsGrad = np.sum(np.outer(np.append(inputs, 1), chainDerivCoef).transpose(), axis = -1)
    return inputsGrad, weightsGrad


def gdOptimize(inputs, weights, outputs, **kwargs):
    learningRate, actFunc, actFuncD, layerFunc = karges(kwargs, {'learningRate': 0.1, 'actFunc': sigmoid, 'actFuncD': sigmoidD, 'layerFunc': fullyConnectedLayer})
    weightedSum = layerFunc(inputs, weights)
    layerOutputs = actFunc(weightedSum)
    chainDerivCoef = np.sum(2 * (layerOutputs - outputs), axis = -1) * actFuncD(weightedSum)
    #chainDerivCoef = 2 * (layerOutputs - outputs) * actFuncD(weightedSum)
    inputsGrad = np.sum(weights.transpose()[:-1] * chainDerivCoef, axis = -1)
    weightsGrad = np.outer(np.append(inputs, 1), chainDerivCoef).transpose()
    newInputs = inputs - inputsGrad * learningRate
    newWeights = weights - weightsGrad * learningRate
    return newInputs, newWeights

def backProp(inputs, weights, outputs, **kwargs):
    optimizer, layerFuncs = karges(kwargs, {'optimizer': gdOptimize, 'layerTypes': [fullyConnectedLayer] * len(weights)})
    layerInputs = neuralNetwork(inputs, weights, **kwargs)[:-1][::-1] + [inputs]
    newWeights = deepcopy(weights)[::-1]
    targetOutputs = outputs
    layerFuncs = deepcopy(layerFuncs)[::-1]
    for layerIndex, layerWeights in enumerate(newWeights):
        targetOutputs, newWeights[layerIndex] = optimizer(layerInputs[layerIndex], layerWeights, targetOutputs, layerFunc = layerFuncs[layerIndex], **kwargs)
    return newWeights[::-1]
    

def train(datasetFunc, weights, **kwargs):
    costThreshold, iterLimit, saveInterval, saveLocation = karges(kwargs, {'costThreshold': 0.1, 'iterLimit': 1000, 'saveInterval': -1, 'saveLocation': 'weights.npy'})
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
