import numpy as np
from time import time
from CNN import *
np.random.seed(1)

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


def train(image, label, conv, maxp, acti, rate=0.1):
	out, loss, acc = test(image, label, conv, maxp, acti)

	gradient = np.zeros(label.shape)
	gradient[np.argmax(label)] = -1 / out[np.argmax(label)]

	back = acti.back_prop(gradient, rate)
	back = maxp.back_prop(back)
	back = conv.back_prop(back, rate)

	return out, loss, acc

inputs = np.random.rand(100,96,96*3)
outputs = np.zeros((100,2))

conv = Conv(18,7)
maxp = Maxpool(4)
acti = Softmax(22*70*18, outputs[0].size)

for row in range(len(outputs)):
	outputs[row][np.random.choice(len(outputs[row]))] = 1

start = time()
l, a = [], []
for row in range(len(outputs)):
	output, loss, accuracy = train(inputs[row], outputs[row], conv, maxp, acti)
	l.append(loss); a.append(accuracy)

	print('Output   ➤ ', output)
	print('Real     ➤ ', outputs[row])
	print('Loss     ➤ ', loss)
	print('Accuracy ➤ ', accuracy)
	print('Time     ➤ ', f'{int(time() - start)}s\n')

print('\nTRAINING REPORT')
print('Loss     ➤ ', sum(l)/len(l))
print('Accuracy ➤ ', sum(a)/len(a))
print('Duration ➤ ', '{0:.4f}'.format(time() - start), 'seconds')
