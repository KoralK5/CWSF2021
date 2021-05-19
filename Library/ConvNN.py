import numpy as np
import matplotlib.pyplot as plt


class Conv:
	def __init__(self, num_filters, filter_size):
		self.num_filters = num_filters
		self.filter_size = filter_size
		self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/filter_size**2
		
	def image_region(self, image):
		height, width = image.shape
		self.image = image
		for x in range(height-self.filter_size+1):
			for y in range(width-self.filter_size+1):
				image_patch = image[x:x+self.filter_size, y:y+self.filter_size]
				yield image_patch, x, y
		
	def forward_prop(self, image):
		height, width = image.shape
		conv_out = np.zeros((height-self.filter_size+1, width-self.filter_size+1, self.num_filters))
		for image_patch, x, y in self.image_region(image):
			conv_out[x, y] = np.sum(image_patch*self.conv_filter, axis=(1,2))
		return conv_out
		
	def back_prop(self, dL_dout, learning_rate):
		dl_dF_params = np.zeros(self.conv_filter.shape)
		for image_patch, x, y, in self.image_region(self.image):
			for z in range(self.num_filters):
				dL_dF_params[k] += image_patch * dL_dout[x,y,z]
		self.conv_filter -= learning_rate * dL_dF_params
		return dL_dF_params


class Maxpool:
	def __init__(self, filter_size):
		self.filter_size = filter_size
		
	def image_region(self, image):
		new_height = image.shape[0] // self.filter_size
		new_width = image.shape[1] // self.filter_size
		self.image = image
		for x in range(new_height):
			for y in range(new_width):
				image_patch = image[x*self.filter_size:x*self.filter_size+self.filter_size, y*self.filter_size:y*self.filter_size+self.filter_size]
				yield image_patch, x, y
		
	def forward_prop(self, image):
		height, width, num_filters = image.shape
		output = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))
		for image_patch, x, y in self.image_region(image):
			output[x, y] = np.amax(image_patch, axis=(0,1))
		return output
		
	def back_prop(self, dL_dout):
		dL_dmax_pool = np.zeros(self.image.shape)
		for image_patch, x, y in self.image_region(self.image):
			height, width, num_filters = image_patch.shape
			maximum_val = np.amax(image_patch, axis=(0,1))
			
			for i in range(height):
				for j in range(width):
					for k in range(num_filters):
						if image_patch[i,j,k] == maximum_val[i,j,k]:
							dL_dmax_pool[x*self.silter_size+i, y*self.filter_size+j+k] - dL_dout[x,y,k]

		return dL_dmax_pool


class Softmax:
	def __init__(self, input_node, softmax_node):
		self.weight = np.random.randn(input_node, softmax_node)/input_node
		self.bias = np.zeros(softmax_node)
		
	def forward_prop(self, image):
		self.orig_im_shape = image.shape
		image_modified = image.flatten()
		self.modified_input = image_modified
		output_val = np.dot(image_modified, self.weight) + self.bias
		self.out = output_val
		exp_out = np.exp(output_val)
		return exp_out/np.sum(exp_out, axis=0)
		
	def back_prop(self, dL_dout, learning_rate):
		for i, grad in enumerate(dL_dout):
			if grad == 0:
				continue
			
			transformation_eq = np.exp(self.out)
			S_total = np.sum(transformation_eq)
			
			dy_dz = -transformation_eq * transformation_eq / S_total**2
			dy_dz[i] = transformation_eq[i]*(S_total-transformation_eq[i]) / S_total**2
			
			dz_dw = self.modified_input
			dz_db = 1
			dz_d_inp = self.weight
			
			dL_dz = grad * dy_dz
			
			dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
			dL_db = dL_dw * dz_db
			dL_d_inp = dz_d_inp @ dL_dz
			
			self.weight -= learning_rate * dL_dw
			self.bias -= learning_rate * dL_db
			
			return dL_d_inp.reshape(self.orig_im_shape)


def network(image, label):
	out1 = conv.forward_prop(img)
	out2 = maxp.forward_prop(out1)
	out3 = acti.forward_prop(out2)
		
	cross_ent_loss = -np.log(out_p[label])
	accuracy_eval = np.argmax(out_p) == label
		
	return out_p, cross_ent_loss, accuracy_eval


def train(image, label, rate=0.1):
	out, loss, acc = cnn_forward_prop(image, label)
		
	gradient = np.zeros(1)
	gradient[label] = -1/out[label]
		
	back = Softmax.back_prop(gradient, rate)
	back = Maxpool.back_prop(back)
	back = Conv.back_prop(back, rate)
		
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