import numpy as np
from CNN import *

def cross_entropy(p, q, epsilon=1e-12):
	return -sum([p[i]*np.log2(q[i]+epsilon) for i in range(len(p))])

def accuracy_eval(p, q):
	return np.argmax(p) == np.argmax(q)

def test(image, label, conv, maxp, acti):
	out = conv.forward_prop(image)
	out = maxp.forward_prop(out)
	out = acti.forward_prop(out)

	loss = cross_entropy(out, label)
	accuracy = accuracy_eval(out, label)

	return out, loss, accuracy

def train(image, label, conv, maxp, acti, rate=0.0005):
	out, loss, acc = test(image, label, conv, maxp, acti)

	gradient = np.zeros(label.shape)
	gradient[np.argmax(label)] = -1 / out[np.argmax(label)]

	back = acti.back_prop(gradient, rate)
	back = maxp.back_prop(back)
	back = conv.back_prop(back, rate)

	return out, loss, acc
