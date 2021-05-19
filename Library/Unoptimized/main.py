import numpy as np
from time import time
from CNN import *


def cross_entropy(p, q, epsilon=1e-12):
	return -sum([p[i]*np.log2(q[i]+epsilon) for i in range(len(p))])


def accuracy_eval(p, q):
	return np.argmax(p) == np.argmax(q)


def network(image, label, conv, maxp):
	out1 = conv.forward_prop(image)
	out2 = maxp.forward_prop(out1)
	out3 = Softmax(np.prod(out2.shape), label.size).forward_prop(out2)

	loss = cross_entropy(out3, label)
	accuracy = accuracy_eval(out3, label)

	return out3, loss, accuracy


def train(image, label, rate=0.1):
	out, loss, acc = cnn_forward_prop(image, label)
	
	gradient = np.zeros(1)
	gradient[label] = -1/out[label]
	
	back = acti.back_prop(gradient, rate)
	back = maxp.back_prop(back)
	back = conv.back_prop(back, rate)

	return loss, acc


def fullTrain():
	for epoch in range(1):
		print(f'Epoch {epoch+1}')

		shuffled_data = np.random.permutation(len(train_images))
		train_images = train_images[shuffled_data]
		train_labels = train_labels[shuffled_data]

		loss, num_correct = 0, 0
		for i, (im, label) in enumerate(zip(train_images, train_labels)):
			if i%100 == 0:
				print('Iteration:', i+1)
				print('Loss:     ', float("{0:.4f}".format(loss/100)))
				print('Accuracy: ', num_correct)
				loss, num_correct = 0, 0

			l, accu = training_cnn(im, label)
			loss += l
			num_correct += accu


inputs = np.random.rand(100,96,96*3)
outputs = np.zeros((100,2))
for row in range(len(outputs)):
	outputs[row][np.random.choice(len(outputs[row]))] = 1

conv = Conv(18,7)
maxp = Maxpool(4)

start = time()
l, a, size = 0, 0, len(outputs)
for row in range(size):
	output, loss, accuracy = network(inputs[row], outputs[row], conv, maxp)
	l += loss

	print('Output   ➤ ', output)
	print('Real     ➤ ', outputs[row])
	print('Loss     ➤ ', loss)
	print('Accuracy ➤ ', accuracy)
	print('Time     ➤ ', f'{int(time() - start)}s\n')
end = time()

print('\nLoss     ➤ ', l/size)
print('Time     ➤ ', '{0:.4f}'.format(end - start), 'seconds')
