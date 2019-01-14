import cPickle,gzip
import numpy as np

class MNIST(object):
	def __init__(self,path):
		f = gzip.open(path,'rb')
		self.train_set,self.valid_set,self.test_set = cPickle.load(f)
		f.close()
		
		self.train_set_img,self.train_set_lab = self.process_data(self.train_set)
		self.valid_set_img,self.valid_set_lab = self.process_data(self.valid_set)
		self.test_set_img,self.test_set_lab = self.process_data(self.test_set)

		# reshape img
		self.train_set_input = np.zeros([len(self.train_set_img),1,28,28],dtype=np.float32)
		self.valid_set_input = np.zeros([len(self.valid_set_img),1,28,28],dtype=np.float32)
		self.test_set_input = np.zeros([len(self.test_set_img),1,28,28],dtype=np.float32)
		for index in range(len(self.train_set_img)):
			self.train_set_input[index][0] += np.reshape(self.train_set_img[index],[28,28])
		for index in range(len(self.valid_set_img)):
			self.valid_set_input[index][0] += np.reshape(self.valid_set_img[index],[28,28])
		for index in range(len(self.test_set_img)):
			self.test_set_input[index][0] += np.reshape(self.test_set_img[index],[28,28])

		# cast one hot output
		self.train_set_output = self.cast_output(10,self.train_set_lab)
		self.valid_set_output = self.cast_output(10,self.valid_set_lab)
		self.test_set_output = self.cast_output(10,self.test_set_lab)

		

	# split training data into input and label set
	def process_data(self,data_set):
		data_x,data_y = data_set
		return data_x,data_y

	# create one-hot label from raw label data
	def cast_output(self,output_dim,set_lab):
		output = np.zeros([len(set_lab),output_dim],dtype=np.float32)
		for row_i in range(len(set_lab)):
			output[row_i][set_lab[row_i]] = 1
		return output

# load training, validation and test data
# f = gzip.open('../data/mnist.pkl.gz','rb')
# train_set,valid_set,test_set = cPickle.load(f)
# f.close()

