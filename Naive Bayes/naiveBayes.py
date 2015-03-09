'''
	Naive Bayes Classifier Implementation 
	Author: Nashid Shaila
	UofO, Winter 2015
	Machine Learning CIS 572
	Homework 3
	
	to run: python naiveBayes.py <train_file> <test_file> <beta> <model_file>
'''

import random;
import sys;
import csv;

class NaiveBayes:

	def __init__(self, n):

		self.data = [] # data
		self.n = n
		self.dim = n # dimension of the data
		self.values = {} # possible values
		self.totals = {} # number of times a value appears
		self.counts = [{} for i in range(n)] # number of feature value appearences by index and label value
		self.possible = [set() for i in range(n)] # range possible values for each feature
		

	def getData(self):
		return self.data

	def getDim(self):
		return self.dim

	'''
	Adds a list of data points to the classifier.
	data a list of tuples consisting of data point, label such that
			len(data point) = self.dim
	'''
	
	def addData(self, data):
		
		for d in data:
			if len(d[0]) != self.dim:
				return False

		self.data += data

		for d in data:

			if d[1] in self.values:
				self.values[d[1]] += 1

			else: self.values[d[1]] = 1

		return True

	def train(self):

		for d in self.data:

			e = d[0];  # example
			v = d[1]; # value
			
			# update totals for this value
			if v in self.totals:
				self.totals[v] += 1
			else: self.totals[v] = 1

			# go through the fields of the example
			for i in range(self.dim):
				
				if v not in self.counts[i]:
					self.counts[i][v] = {}

				f = e[i]
				self.possible[i].add(f)

				# update the count for this value of i with label v
				if f in self.counts[i][v]:
					self.counts[i][v][f] += 1
				else: self.counts[i][v][f] = 1
			

	def predict(self, obj, beta, model_file):
		
		if len(obj) != self.dim:
			raise ValueError('Unclassifiable object. Wrong feature space.')
			
		#print beta
		b = float(beta)
		#print b
		newData = {}
		ps = []
		#print len(self.values)
		#opens the model file for writing down outputs
		model = open(model_file, "w+")
		for i in range(self.dim):
			if obj[i] not in self.possible[i]:
				newData[i] = obj[i]
				model.write('%s\n' % newData[i])
				#print newData[i]
			
		for v in self.values:
			#p = float(self.values[v])/len(self.data); # probability of label v
			p = (float(self.values[v]) + b - 1)/ (len(self.data) + (2*b) - 2) # probability of label v, using beta prior
			i = 0
			while p > 0.0 and i < self.dim:
				if i not in newData: # can't predict based on things not trained on
					f = obj[i]
					if f in self.counts[i][v]: # f has been seen at position i with value v
						p *= (float(self.counts[i][v][f]) + b - 1)/(self.totals[v]+ (2*b) - 2)
					else:
						p = 0.0
				print p
				i += 1
			ps.append((p, v))
		model.close()
		return max(ps, key=lambda x: x[0]), newData
		

def main(train_file, test_file, beta, model_file):
	
	'''
		beta - value of beta prior
		model_file - the name of the file to store the final classification model
	'''
	
	print "Loading data from", train_file
	print "\nLoading data from", test_file
	
	# read training data
	examples = []
	training_file = open(train_file, 'r')
	sheetReader = csv.reader(training_file)
	for example in sheetReader:
		for i in range(len(example)):
			example[i] = example[i].strip()
			#print example
		v = example.pop()
		#print example[1]
		if example[0]:
			examples.append((example, v))
			#print example, v


	# read testing data
	tests = []
	test_file = open(test_file, 'r')
	sheetReader = csv.reader(test_file)
	for row in sheetReader:
		for i in range(len(row)):
			row[i] = row[i].strip()
		v = row.pop()
		if row[0]:
			tests.append((row, v))
			#print row, v


	nb = NaiveBayes(len(examples[0][0]))
	nb.addData(examples)
	nb.train()

	print "\nThe probabilities for each instance in the test set are:\n"
	
	d = len(tests) # number of test examples
	n = 0 # number correct predictions

	for e in tests:

		test = nb.predict(e[0], beta, model_file)
		if test[0][1] == e[1]: n += 1
		
	
	print '\nNumber of correct predictions:', n
	print 'Total tests:', d
	print '\nAccuracy:', float(n)/d
	
	
	
if __name__ == '__main__':

	'''
		The main function is called when logistic.py is run from the command line with arguments.
	'''
	
	args = sys.argv[1:] # get arguments from the command line
	main( *args )