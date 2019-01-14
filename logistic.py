import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

# class of logistic regression
class logistic:
	# constructor arguments:
		# dimension of input data
		# dimension of output data
		# learning rate of the classifier
		# size of one training epoch
		# size of one training batch
	def __init__(self,input_dim,output_dim,rate,epoch,batch):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.theta = np.random.normal(0,0.1,[self.output_dim,self.input_dim])
		self.bias = np.random.normal(0,0.1,self.output_dim)
		self.delta = np.zeros(self.output_dim,dtype='float')
		self.rate = rate
		self.batch = batch
		self.epoch = epoch
		self.num_epoch = 0

	# softmax function
	def softmax(self,x):
		hypo = np.add(np.dot(self.theta,x),self.bias)
		hypo = np.exp(hypo)
		total = np.sum(hypo)
		hypo = np.divide(hypo,total)
		return hypo

	# cost function given hypothesis
	def cross_entropy(self,hypo,y):
		cost = -np.dot(np.log(hypo),y)
		return cost

	# calculate gradient of cost function with respect to parameters
 	def gradient(self,hypo,x,y):
		self.delta = np.subtract(hypo,y)
		d_theta = np.outer(self.delta,x)
		d_bias = self.delta
		return d_theta,d_bias

	# train logistic regression
	def train(self,train_x,train_y,num_epoch):
		self.num_epoch = num_epoch
		for ne in range(num_epoch):
			num_batch = int(float(self.epoch)/float(self.batch))
			# iterate through one epoch of training data
			for nb in range(num_batch):
				base = ne * self.epoch + nb * self.batch
				# initialize the gradient for one batch 
				d_theta = np.zeros([self.output_dim,self.input_dim],dtype='float')
				d_bias = np.zeros(self.output_dim,dtype='float')
				# total gradient in a batch
				for ni in range(self.batch):
					dt,db = self.gradient(self.softmax(train_x[base+ni]),train_x[base+ni],train_y[base+ni])
					d_theta = np.add(d_theta,dt)
					d_bias = np.add(d_bias,db)
			# update parameters with mini-batch gradient
				self.theta = np.subtract(self.theta,np.multiply(self.rate,np.divide(d_theta,self.batch)))
				self.bias = np.subtract(self.bias,np.multiply(self.rate,np.divide(d_bias,self.batch)))
 			# total cost in current epoch
			error_total = 0
			base = ne * self.epoch
			for ni in range(self.epoch):
				error_total = error_total + self.cross_entropy(self.softmax(train_x[base+ni]),train_y[base+ni])	
			error_total = error_total/self.epoch
			print('epoch:'+str(ne)+'      '+'error:'+str(error_total))

	# test performance of logistic regression
	def verify(self,test_x,test_y):
		error = 0.
		for row_i in range(len(test_x)):
			h = self.softmax(test_x[row_i])
			digit_est = np.argmax(h)
			# compare estimated output and actual output
			if(digit_est!=test_y[row_i]):
				error = error + 1
		return 1-error/len(test_x)

	# visualize filter
	def visualize_filter(self,dim_row,dim_col):
		plt.figure()
		row = -1;
		col = 0
		num_col = 3 # number of filters in a row
		for ni in range(self.output_dim):
			img = self.theta[ni].reshape(dim_row,dim_col)
			col = ni%num_col
			if col == 0:
				row += 1
			plt.imshow(img,cmap = cm.Greys_r,extent = np.array([col*dim_col,(col+1)*dim_col,row*dim_row,(row+1)*dim_row]))	
		plt.xlim(-5,dim_col * num_col + 5)
		plt.ylim(-5,dim_row * (row+1) + 5)
	
	# display classifier information
	def show(self):
		print('logistic classifier structure')
		print(str(self.input_dim)+' ---- '+str(self.output_dim))
		print('learning rate:    '+str(self.rate))
		print('--------------------------------')
		print('training information')
		print('mini batch size:  '+str(self.batch))
		print('epoch info:       '+str(self.epoch)+' * '+str(self.num_epoch))
		print('--------------------------------')
				
			
			
