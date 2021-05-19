import numpy as np
from time import time
from CNN import *
from train import *
np.random.seed(1)

inputs = np.random.rand(100,96,96*3)
outputs = np.zeros((100,2))

conv = Conv(18,7)
maxp = Maxpool(4)
acti = Softmax(22*70*18, outputs[0].size)

for row in range(len(outputs)):
	outputs[row][np.random.choice(len(outputs[row]))] = 1

start = time()
l, a = [], []
for epoch in range(3):
	for row in range(len(outputs)):
		output, loss, accuracy = train(inputs[row], outputs[row], conv, maxp, acti)
		l.append(loss); a.append(accuracy)

		print(f'\nITERATION {row+1} OF EPOCH {epoch+1}:\n')
		print('Output    ➤ ', output)
		print('Real      ➤ ', outputs[row])
		print('Loss      ➤ ', loss)
		print('Accuracy  ➤ ', accuracy)
		print('Time      ➤ ', f'{int(time() - start)}s\n')

print('\n\nTRAINING REPORT')
print('Loss     ➤ ', sum(l)/len(l))
print('Accuracy ➤ ', sum(a)/len(a))
print('Duration ➤ ', '{0:.4f}'.format(time() - start), 'seconds')
