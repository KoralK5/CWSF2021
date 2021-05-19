import numpy as np
from time import time
from CNN import *


def network(image, label, conv, maxp):
	out1 = conv.forward_prop(image)
	out2 = maxp.forward_prop(out1)
	out3 = Softmax(np.prod(out2.shape), label.size).forward_prop(out2)

	cross_ent_loss = -np.log(out3[label])
	accuracy_eval = np.argmax(out3) == label
		
	return out3, cross_ent_loss, accuracy_eval


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


dims = (96,96,3)
inp = np.arange(np.prod(dims)).reshape(dims[0],dims[1]*dims[2])
out = np.arange(2)

conv = Conv(18,7)
maxp = Maxpool(4)

start = time()
output, loss, accuracy = network(inp, out, conv, maxp)
end = time()

print('Output   ➤ ', output)
print('Real     ➤ ', out)
print('Loss     ➤ ', loss)
print('Accuracy ➤ ', accuracy)
print('Time     ➤ ', '{0:.4f}'.format(end - start), 'seconds')
