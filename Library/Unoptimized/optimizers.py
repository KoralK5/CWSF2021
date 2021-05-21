import numpy as np

def gradient_descent(param, gradient, time=1, rate=0.001, beta=0.9, scale=1, momentum=0, velocity=0):
	return param - gradient * rate, 0, 0

def momentum(param, gradient, time=1, rate=0.001, beta=0.9, scale=1, momentum=0, velocity=0):
	momentum = beta * momentum + rate * gradient
	return param - momentum, momentum, 0

def debounce(param, gradient, time=1, rate=0.001, beta=0.9, scale=1, momentum=0, velocity=0):
	return param - scale * velocity * np.tanh(gradient), 0, velocity

def nadam(param, gradient, time=1, rate=0.001, beta=0.9, scale=1, momentum=0, velocity=0):
	momentum = beta * momentum + (1 - beta) * gradient
	velocity = beta * velocity + (1 - beta) * gradient**2
	momentum_hat = momentum / (1 - beta**time) + (1 - beta) * gradient / (1 - beta**time)
	velocity_hat = velocity / (1 - beta**time)
	return param - rate * momentum_hat / (velocity_hat**(1/2) + 1e-12), momentum, velocity
