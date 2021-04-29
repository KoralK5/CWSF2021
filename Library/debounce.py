import NeuralNetwork as nn
from numpy import tanh
import _pickle as cPickle

class Layer:
	def __init__(self, inputs, weights, outputs, dx):
		self.inputs = inputs
		self.weights = weights
		self.outputs = outputs
		self.dx = dx

	def adjustWeight(self, neuron, weight):
		newWeights = cPickle.loads(cPickle.dumps(self.weights, -1))
		newWeights[neuron][weight] += self.dx
		return (nn.layerCost(self.inputs, newWeights, self.outputs) - nn.layerCost(self.inputs, self.weights, self.outputs)) / self.dx

	def adjustNeuron(self, neuron):
		newInputs = cPickle.loads(cPickle.dumps(self.inputs, -1))
		newInputs[neuron] += self.dx
		return (nn.layerCost(newInputs, self.weights, self.outputs) - nn.layerCost(self.inputs, self.weights, self.outputs)) / self.dx
		
	def adjustLayer(self, rate, beta, scale, velW, velI, preVelW, preVelI):
		newWeights = cPickle.loads(cPickle.dumps(self.weights, -1))
		newInputs = cPickle.loads(cPickle.dumps(self.inputs, -1))
		for neuron in range(len(self.weights)):
			for weight in range(len(self.weights[neuron])):
				gradient = self.adjustWeight(neuron, weight)
				velW = (beta - scale * tanh(velW - preVelW)) * velW + gradient * rate

				newWeights[neuron][weight] -= (beta - scale * tanh(velW - preVelW)) * velW + rate * gradient

				preVelW = velW

		for neuron in range(len(self.inputs)):
			gradient = self.adjustNeuron(neuron)
			velI = (beta - scale * tanh(velI - preVelI)) * velI + gradient * rate

			newInputs[neuron] -= (beta - scale * tanh(velI - preVelI)) * velI + rate * gradient

			preVelI = velI
		return newWeights, newInputs

class Network:
	def __init__(self, inputs, weights, outputs, dx):
		self.inputs = inputs
		self.weights = weights
		self.outputs = outputs
		self.dx = dx

	def adjustWeight(self, layer, neuron, weight):
		newWeights = cPickle.loads(cPickle.dumps(self.weights, -1))
		newWeights[layer][neuron][weight] += self.dx
		return (nn.neuralNetworkCost(self.inputs, newWeights, self.outputs) - nn.neuralNetworkCost(self.inputs, self.weights, self.outputs)) / self.dx

	def adjustNeuron(self, layer, neuron):
		newInputs = cPickle.loads(cPickle.dumps(self.inputs, -1))
		newInputs[layer][neuron] += self.dx
		return (nn.neuralNetworkCost(newInputs, self.weights, self.outputs) - nn.neuralNetworkCost(self.inputs, self.weights, self.outputs)) / self.dx
		
	def adjustNetwork(self, rate):
		newWeights = cPickle.loads(cPickle.dumps(self.weights, -1))
		for layer in range(len(self.weights)):
			for neuron in range(len(self.weights[layer])):
				for weight in range(len(self.weights[layer][neuron])):
					newWeights[layer][neuron][weight] -= self.adjustWeight(layer, neuron, weight) * rate
		return newWeights

def backPropagation(inputs, weights, outputs, dx, rate, beta, scale):
	newWeights = cPickle.loads(cPickle.dumps(weights[::-1], -1))
	newOutputs = cPickle.loads(cPickle.dumps(outputs, -1))
	
	inps = nn.neuralNetwork(inputs, weights)

	networkInputs = inps[-2::-1] + [cPickle.loads(cPickle.dumps(inputs, -1))]

	velW, velI = 0, 0
	preVelW, preVelI = 0, 0
	for currentLayer in range(len(networkInputs)):
		layer = Layer(networkInputs[currentLayer], newWeights[currentLayer], newOutputs, dx)
		newWeights[currentLayer], newOutputs = layer.adjustLayer(rate, beta, scale, velW, velI, preVelW, preVelI)

	return newWeights[::-1], inps[-1]
