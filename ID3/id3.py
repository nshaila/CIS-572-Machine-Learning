'''
	ID3 Decision Tree Classifier Implementation 
	Author: Nashid Shaila
	UofO, Winter 2015
	Machine Learning CIS 572
	
'''

import csv
import numpy as np
import copy
import sys
import math


class Node: # class for building the node of the decision tree
	
	def __init__(self, feat):
		self.feat = feat
		self.child = [] # list of children 


class Leaf: # class for building the leaf of the decision tree
	
	def __init__(self, value):
		self.value = value


class DecisionTree: # class for doing all decision tree operation
	
	def __init__(self, criteria='entropy', chi=6.635):
	
		self.chi_min = chi   # chi-squared test parameter
		self.criteria = criteria

	def __str__(self):
	
		return "using ID3 algorithm with criterion = '" + self.criteria + "'and chi squared critical value = " + str(self.chi_min) + "\n"

	def fitting(self, features, X, y): # builds decision tree using training data labeling
		 
		self.node = 0 # stores the total number of nodes 
		self.leaf = 0 # stores the total number of leaves
		self.features = features
		self.index = {} 
		self.cardinality = [] 

		for i, col in enumerate(X.transpose()):
			p = 0
			r = []
			for val in col:
				if val not in r:
					p = p + 1
					r.append(val)
			self.cardinality.append(len(r))
			self.index[features[i]] = i

		self.root = self.expand(X, y) # recursive call to grow the tree
		

	def expand(self, X, y): # X is input, y is predicted class
		
		if np.sum(y==1) == 0: # basecase for recursion
			self.leaf = self.leaf + 1
			return Leaf(0)
			
		elif np.sum(y==0) == 0:
			self.leaf = self.leaf + 1
			return Leaf(1)
			
		else:
			
			index = self.best_feat(X, y) # choose the attribute value
			if index is None:
				y0, y1 = np.sum(y==0), np.sum(y==1)
				return Leaf(0) if y0 > y1 else Leaf(1)

			splits = self.split(X, y, index) # partition data by the chosen value

			current_node = Node(self.features[index])  # recursive call to grow the tree
			
			for X_split, y_split in splits:
				current_node.child.append(self.expand(X_split, y_split))
			
			self.node += 1 
			return current_node  # returns root of the tree
			

	def split(self, X, y, i): #this function partitions the data. i is the index of the attribute for partition
		
		X_set = []
		y_set = []
		c = xrange(self.cardinality[i]) # an iterator 
		col = X.transpose()[i] # values of the split column 
		
		for val in c:
			count = np.sum(col==val)
			X_set.append(np.zeros(shape=(count, X.shape[1]), dtype='int'))
			y_set.append(np.zeros(shape=count, dtype='int'))
		
		arr = {}
		for val in c:
			arr[val] = 0
			
		for index, instance in enumerate(X):
			val = instance[i]
			X_set[val][arr[val]] = instance
			y_set[val][arr[val]] = y[index]
			arr[val] += 1

		return zip(X_set, y_set) # returns data and label lists, each split into two parts
		

	def best_feat(self, X, y): # feature with highest info gain is selected in this function
		
	  	if X.shape[1] == 1:
	  		return None
	  		
  		if self.criteria == 'entropy':
  			info_y = self.info(float(np.sum(y==1)), float(np.sum(y==0)))
  			
		  	gains = [] # list of information gains
		  	
		  	for i, feat in enumerate(self.cardinality):  # gets entropy, filters with chi-squared test
		  		c = xrange(self.cardinality[i])
	  			e, chi = self.get_entropy(c, X.transpose()[i], y)
	  			gain = info_y - e if chi else 0.0
	  			gains.append(gain)

	  		if gains.count(0.0) == len(self.cardinality): # to check if chi-squared test stops
				return None

	  		return gains.index(max(gains)) # returns the index of the max info gain

  	
	def split_probability(self, c, B, y): # B is the split attribute
		
		y_split = []
		hash = {}
		
		for val in c:
			y_split.append(np.zeros(shape=np.sum(B==val), dtype='int'))
			hash[val] = 0
			
		for index, val in enumerate(B):
			#print count(B)
			try:
				y_split[val][hash[val]] = y[index]
				hash[val] = hash[val] + 1
			#print y_split[val][hash[val]]
			except IndexError:
				#print "Exit"
				#hash[val] = hash[val] + 1
				pass

		pn_count = []
		for l in c:
			split = []
			for target in [0,1]:
				split.append(float(np.sum(y_split[l]==target)))
			pn_count.append(copy.deepcopy(split))

		probability = []
		for val in c:
			probability.append(float(y_split[val].shape[0])/(y.size)) 

		return probability, pn_count # returns probabilities for a split attribute B and the counts of labels for each split
		

	def chi_squared_helper(self, p, n, c, probability, pn_count): # chi_squared test calculation used by split criteria
		
		result = 0
		for l in c:
			P = p * probability[l]
			N = n * probability[l]
			result = result + (((pn_count[l][1] - P)**2)/P + ((pn_count[l][0] - N)**2)/N) if probability[l] != 0.0 else 0.0

		return result > self.chi_min


	def info(self, p, n): # this function calculates probabilities to avoid multiple calls for split gain calculation
	
		if (p + n) == 0.0:
			return 0.0

		P = (p)/(p + n)
		N = (n)/(p + n)
		
		if P == 0.0 and N == 0.0:
			return 0.0
		elif P == 0.0:
			return (-(N) * math.log(N, 2))
		elif N == 0.0:
			return (-(P) * math.log(P, 2))
		else:
			return (-(P) * math.log(P, 2)) - ((N) * math.log(N, 2))
			

	def get_entropy(self, cardinality, A, y): # this function calculates entropy
		
		def entropy(c, B, y, p, n):
		
			probability, pn_count = self.split_probability(c, B, y)
			chi_result =  self.chi_squared_helper(p, n, c, probability, pn_count)

			gain = 0
			for val in c:
				n, p = pn_count[val]
				gain = gain + probability[val] * self.info(p, n)

			return gain, chi_result  # returns the entropy and the chi_squared test value for split

		p = float(np.sum(y==1))
		n = float(np.sum(y==0))
		return entropy(cardinality, A, y, p, n)  # returns a tuple of entropy and chi squared test


	def prediction(self, X):
		
		class_prediction = [self.classify(self.root, instance) for instance in X]
		return class_prediction  # returns the predicted classes
		

	def classify(self, current_node, k):
		
		if isinstance(current_node, Leaf): # recursion basecase. it returns predicted value
			return current_node.value
		
		index = self.index[current_node.feat]
		val = k[index]
		return self.classify(current_node.child[val], k) # recursion on the tree 
		

	def output_model(self, model_file): # writes into the output model file
	
		print('Writing the ID3 Decision Tree with ' + str(self.node) + ' nodes and ' +
			  str(self.leaf) + ' leaves into ' + model_file)
		
		self.model = open(model_file, "w+")
		self.output_model_recursive(self.root, 0, True) # recursive tree traversing
		self.model.close()
		print '\n'
		print('Write Complete...\n')
		

	def output_model_recursive(self, current_node, depth, begin=False):
	
		if isinstance(current_node, Leaf): # basecase 
			self.model.write(str(current_node.value))
			return
			
		for val, child in enumerate(current_node.child): # iterates the children 
		
			self.depth_helper(depth, begin)
			begin = False
			self.model.write(str(current_node.feat) + ' = ' + str(val) + ' : ')
			self.output_model_recursive(child, depth+1)

	def depth_helper(self, depth, begin):
	
		if not begin:
			self.model.write('\n')
		if depth > 0:
			for i in range(0,depth):
				self.model.write('| ')


def csv_load(data_set_train, data_set_test=None): # loads the csv file and puts the data into numpy arrays
	
	def load_helper(data):
	
		print "\nLoading Data from", data
		
		with open(data, 'r+') as csv_file: # creates attribute tuple and labels array
			reader = csv.reader(csv_file)
			features = list(reader.next())
			#print features
			features.pop() # removes 'class' feature name

		# load the csv data into a numpy array
		X = np.loadtxt(data, delimiter=",", dtype='int', skiprows=1)
		#print "X",X
		y = X[:,-1:] # get only the labels
		#print "Y", y
		y = y.flatten() # make the single column 1 dimensional
		#print "Y", y
		X = X[:,:-1] # remove the labels column
		#print "X",X
		return features, X, y

	features, X_train, y_train = load_helper(data_set_train)
	if data_set_test is not None:
		dummy, X_test, y_test = load_helper(data_set_test)
	else:
		X_test = None
		y_test = None

	return features, X_train, y_train, X_test, y_test # returns test and training data split into separated numpy arrays with their repective labels


def accuracy(y_test, class_prediction):
	
	accurate = 0
	i = 0
	for p in class_prediction:
		if p == y_test[i]:
			accurate = accurate + 1
		i = i + 1
		
	return (float(accurate) / len(y_test)) * 100   # returns the accuracy of the predictions


def main(train_file, test_file, model_file, criteria='entropy', chi=6.635):
	
	features, X_train, y_train, X_test, y_test = csv_load(train_file, test_file) # opens and loads csv files

	id3_tree = DecisionTree(criteria, float(chi)) # creates the decision tree with training data
	print "\n\nCreated a Decision Tree Classifier", id3_tree

	id3_tree.fitting(features, X_train, y_train) # fits the model to the loaded training data

	class_prediction = id3_tree.prediction(X_test) # prediction for test data

	print('Accuracy: ' + str(accuracy(y_test, class_prediction)) + '%' '\n')
	
	id3_tree.output_model(model_file) # writes model to model file


if __name__ == '__main__':
	
	args = sys.argv[1:] # Get the filename parameters from the command line
	main( *args )
	
	
	