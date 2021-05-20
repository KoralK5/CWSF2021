print('\nImporting Modules')

import os
import numpy as np
from time import time
from matplotlib import pyplot as plt
from cv2 import imread, resize
from CNN import *
from train import *
np.random.seed(1)

def fix(img):
    return np.array(resize(img, (96,96)).reshape(96,96*3))/255.0

def grab(path):
	data = []
	categories = ['hem','all']
	folders = ['fold_0','fold_1','fold_2']
	for folder in folders:
		for category in categories:
			pt = f'{path}\\{folder}'
			p = f'{pt}\\{category}'
			output = int(category=='hem')
			for img in os.listdir(p):
				data.append([fix(imread(os.path.join(p,img))), np.array([output, not(output)])])
	np.random.shuffle(data)
	return data

def plot(acc, loss, title):
	fig, axs = plt.subplots(2)
	fig.suptitle(title)
	axs[0].plot(range(len(acc)), acc, 'tab:blue')
	axs[0].set(ylabel='Accuracy')

	axs[1].plot(range(len(loss)), loss, 'tab:red')
	axs[1].set(xlabel='Epoch', ylabel='Loss')

	plt.show()

def run(data, rate, epochs, model, path):
	start = time()
	l, a, size = [], [], len(data)
	for epoch in range(epochs):
		for row in range(size):
			output, loss, accuracy = train(data[row][0], data[row][1], rate, *model)
			l.append(loss); a.append(accuracy)

			np.save(f'{path}model\\weights.npy', np.array([model[2].weight, model[2].bias], dtype=object))
			f = open(f'{path}model\\scores.txt', 'a'); f.write(f'\n{loss}'); f.close()

			print(f'\nITERATION {row+1}/{size} OF EPOCH {epoch+1}/{epochs}:\n')
			print('Output    ➤ ', output)
			print('Real      ➤ ', data[row][1])
			print('Loss      ➤ ', '{0:.4f}'.format(loss))
			print('Accuracy  ➤ ', accuracy)
			print('Time      ➤ ', f'{int(time() - start)}s\n')

	print('\n\nTRAINING REPORT\n')
	print('Loss     ➤ ', sum(l)/len(l))
	print('Accuracy ➤ ', sum(a)/len(a))
	print('Duration ➤ ', '{0:.4f}'.format(time() - start), 'seconds')

	return a, l

print('\nReading Data')

path = 'Unoptimized\\'
dataPath = 'LeukemiaData\\C-NMC_Leukemia\\training_data'

data = grab(dataPath)
rate = 0.005
epochs = 3
model = (Conv(18,7), Maxpool(4), Softmax(22*70*18, data[0][1].size))

print('\nTraining Model')

accuracy, loss = run(data, rate, epochs, model, path)

epoch_accuracy = np.sum(np.array_split(accuracy, epochs), axis=1) / len(outputs)
epoch_loss = np.sum(np.array_split(loss, epochs), axis=1) / len(outputs)

print('\n\nTRAINING PROGRESS\n')
print('Accuracy ➤ ', epoch_accuracy)
print('Loss     ➤ ', epoch_loss)

plot(epoch_accuracy, epoch_loss, 'Epochs')
plot(accuracy, loss, 'Iterations')
