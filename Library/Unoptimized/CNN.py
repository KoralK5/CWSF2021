import numpy as np
from time import time
from CNN import *
np.random.seed(1)

def cross_entropy(p, q, epsilon=1e-12):
	return -sum([p[i]*np.log2(q[i]+epsilon) for i in range(len(p))])


def accuracy_eval(p, q):
	return np.argmax(p) == np.argmax(q)


def test(image, label, conv, maxp):
	out = conv.forward_prop(image)
	out = maxp.forward_prop(out)
	out = Softmax(np.prod(out.shape), label.size).forward_prop(out)

	loss = cross_entropy(out, label)
	accuracy = accuracy_eval(out, label)

	return out, loss, accuracy


def train(image, label, conv, maxp, rate=0.1):
	out, loss, acc = test(image, label, conv, maxp)

	gradient = np.zeros(2)
	gradient[np.argmax(label)] = -1 / out[np.argmax(label)]

	back = acti.back_prop(gradient, rate)
	back = maxp.back_prop(back)
	back = conv.back_prop(back, rate)

	return out, loss, acc


conv = Conv(18,7)
maxp = Maxpool(4)
inputs = np.random.rand(100,96,96*3)
outputs = np.zeros((100,2))
for row in range(len(outputs)):
	outputs[row][np.random.choice(len(outputs[row]))] = 1


start = time()
l, a, size = 0, 0, len(outputs)
for row in range(size):
	output, loss, accuracy = train(inputs[row], outputs[row], conv, maxp)
	l += loss; a += accuracy

	print('Output   ➤ ', output)
	print('Real     ➤ ', outputs[row])
	print('Loss     ➤ ', loss)
	print('Accuracy ➤ ', accuracy)
	print('Time     ➤ ', f'{int(time() - start)}s\n')
end = time()


print('\nLoss     ➤ ', l/size)
print('Accuracy ➤ ', a/size)
print('Time     ➤ ', '{0:.4f}'.format(end - start), 'seconds')
