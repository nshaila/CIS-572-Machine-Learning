'''
	Logistic Regression Classifier Implementation 
	Author: Nashid Shaila
	UofO, Winter 2015
	Machine Learning CIS 572
	Homework 3
	
	to run: python logistic.py <train_file> <test_file> <learning_rate> <standard_deviation> <model_file>
'''


import sys
import numpy as np
import h5py


class LogisticRegression():

	'''
		This class is responsible for all logistic regression classifier operations as in
		building a model from training data, class prediction from testing data, and printing the model.
	'''
	
	def __init__(self, eta, sigma, maxiter):
	
		self.eta = float(eta) # learning rate for gradient ascent
		self.sigma = float(sigma) # standerd deviation 
		self.epsilon = 0.00001 # convergence measure
		self.maxiter = int(maxiter) # maximum number of iterations through the data before stopping
		self.threshold = 0.5 # class prediction threshold

	def fit(self, X_, y):
	
		'''
			This function optimizes the parameters for the logistic regression classification model from training 
			data using learning rate eta and regularization constant sigma
			@post: parameter(theta) optimized by gradient descent
		'''
		X = self.add_ones(X_) # prepend ones of training set for theta_0 calculations
		
		# initialize optimization arrays
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.probability = np.zeros(self.m, dtype='float') # stores probabilities generated by the logistic function
		self.theta = np.zeros(self.n, dtype='float') # stores the model theta

		# iterate through the data at most maxiter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		#print "iter | magnitude of the gradient"
		for iteration in xrange(self.maxiter):
			# calc probabilities
			self.probability = self.get_probability(X)

			# calculate the gradient and update theta
			gw = (1.0/self.m) * np.dot(X.T, (self.probability - y))
			g0 = gw[0] # save the theta_0 gradient calc before regularization
			gw += ((self.sigma * self.theta) / self.m) # regularize using the sigma term
			gw[0] = g0 # restore regularization independent theta_0 gradient calc
			self.theta -= self.eta * gw # update parameters
			
			# calculating the magnitude of the gradient and check for convergence
			loss = np.linalg.norm(gw)
			if self.epsilon > loss:
				break
			
			#print iteration, ":", loss

	def get_probability(self, X):
		return 1.0 / (1 + np.exp(- np.dot(X, self.theta)))

	def predict_probability(self, X):
	
		'''
			Returns the set of classification probabilities based on the model theta.
		'''
		
		X_ = self.add_ones(X)
		return self.get_probability(X_)
		

	def predict(self, X):
	
		'''
			Classifies a set of data instances X based on the set of trained feature theta.
		'''
		y_pred = [proba > self.threshold for proba in self.predict_probability(X)]
		return np.array(y_pred)


	def add_ones(self, X):
		# prepend a column of 1's to dataset X to enable theta_0 calculations
		return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))
		

	def print_model(self, features, model_file):
	
		for i in xrange(self.n):
			model_file.write('%f\n' % (self.theta[i]))
			#else:
			#model_file.write('%s %f\n' % (features[i-1], self.theta[i]))


def load_csv(data, shuffle=False):

	'''
		Loads the csv files into numpy arrays.
		data: The data file in csv format to be loaded
		shuffle: True => shuffle data instances to randomize
	'''
	
	print "Loading data from", data
	dset = np.loadtxt(data, delimiter=",", dtype='float')
	return shuffle_split(dset, shuffle)
	

def saveh(dset, path):

	'''
		Stores the numpy data in h5 format.
		dset: Dataset to store
		path: The path to file (including the file name) to save h5 file to
	'''
	
	f = h5py.File(path, 'w')
	f['dset'] = dset
	f.close()
	

def loadh(path, shuffle=False):

	'''
		Loads the h5 data into a numpy array.
		path: path to file (including the file name) to load data from
		shuffle: True => shuffle data instances to randomize
	'''
	
	f = h5py.File(path,'r') 
	data = f.get('dset') 
	dset = np.array(data)
	return shuffle_split(dset, shuffle)
	

def shuffle_split(dset, shuffle):

	# randomize data
	if shuffle:
		dset = shuffle_data(dset)

	# split instances and labels
	y = dset[:,-1:] # get only the labels
	X = dset[:,:-1] # remove the labels column from the data array

	return X, y
	

def shuffle_data(X):

	# get a random list of indices
	rand_indices = np.arange(X.shape[0])
	np.random.shuffle(rand_indices)

	# build shuffled array
	X_ = np.zeros(X.shape)
	for i, index in enumerate(rand_indices):
		X_[i] = X[index]
	return X_
	

def scale_features(X, new_min, new_max):

	# scales all features in dataset X to values between new_min and new_max
	X_min, X_max = X.min(0), X.max(0)
	return (((X - X_min) / (X_max - X_min)) * (new_max - new_min + 0.000001)) + new_min
	

def accuracy(y_test, y_pred):

	correct = 0
	for i, pred in enumerate(y_pred):
		if int(pred) == y_test[i]:
			correct += 1
	return float(correct) / y_test.size


def main(train_file, test_file,  model_file, eta=0.01, sigma=0, maxiter=600000):

	'''
		eta - the learning rate for gradient descent
		sigma - the standerd deviation
		model_file - the name of the file to store the final classification model
	'''
	
	# open and load csv files
	X_train, y_train = load_csv(train_file, True)
	X_test, y_test = load_csv(test_file, True)
	y_train = y_train.flatten() 
	y_test = y_test.flatten()

	# scale features to encourage gradient descent convergence
	X_train = scale_features(X_train, 0.0, 1.0)
	X_test = scale_features(X_test, 0.0, 1.0)


	# create the logistic regression classifier using training data
	classifier = LogisticRegression(eta, sigma, maxiter)
	
	# fit the model to the training data
	print "\nFitting the training data...\n"
	classifier.fit(X_train, y_train)

	# predict the results for the test data
	print "Generating probability prediction for the test data...\n"
	y_pred = classifier.predict(X_test)

	#opens the model file for writing down outputs
	model = open(model_file, "w+")
	classifier.print_model(X_test, model)
	
	# print the classification results
	print "The probabilities for each instance in the test set are:\n"
	for p in classifier.predict_probability(X_test):
		print p
			
	# print simple precision metric to the console
	print('\n\nAccuracy:  ' + str(accuracy(y_test, y_pred)))
	
	model.close()


if __name__ == '__main__':

	'''
		The main function is called when logistic.py is run from the command line with arguments.
	'''
	args = sys.argv[1:] # get arguments from the command line
	main( *args )