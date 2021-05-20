import numpy as np
from CNN import *

def cross_entropy(p, q, epsilon=1e-12):
	return -sum([p[i]*np.log2(q[i]+epsilon) for i in range(len(p))])

def accuracy_eval(p, q):
	return np.argmax(p) == np.argmax(q)

def test(image, label, conv, maxp, acti):
	image = conv.forward_prop(image)
	image = maxp.forward_prop(image)
	image = acti.forward_prop(image)

	loss = cross_entropy(image, label)
	accuracy = accuracy_eval(image, label)

	return image, loss, accuracy

def train(image, label, rate, conv, maxp, acti):
	out, loss, acc = test(image, label, conv, maxp, acti)

	gradient = np.zeros(label.shape)
	gradient[np.argmax(label)] = -1 / out[np.argmax(label)]

	back = acti.back_prop(gradient, rate)
	back = maxp.back_prop(back)
	back = conv.back_prop(back, rate)

	return out, loss, acc
