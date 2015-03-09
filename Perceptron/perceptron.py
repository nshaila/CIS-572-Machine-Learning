'''''

	Perceptron Algorithm Implementation
	Author: Nashid Shaila
	UofO, winter2015
	Machine Learning
	
'''''


import csv
import numpy as np
import copy
import sys
import math

from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import Perceptron
from datetime import datetime


def sumFunction(values, weights, b):
	return sum(value * weights[index] for index, value in enumerate(values)) + b
	
def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))

'''
	The training loop for the perceptron passes through the training data 100
    times. it updates the weight vector for each label based on errors it makes.
'''
    
def trainPerceptron(eta, X_train,text_file):

	xSample, xLength = X_train.shape
	#print xSample, xLength
	#xLength = xLength - 1
	#print xLength
	weights = [0] * xLength
	#print weights
	#y_bin = y_train.copy()
	#print y_bin.shape
	learning_rate = float(eta)
	passes = 0
	updates = 0
	b = 0
	errorCount = 0.0
	#trainCount = 0.0
	
	#start = datetime.now()
	while True: #passes < 100:
		errorCount = 0
		i = 0
		for row in X_train:
			temp = row[0:xLength]
			#print inputVector
			inputVector = temp[:-1]
			#print inputVector
			desiredOutput = row[-1]
			#print desiredOutput
			#print sumFunction(inputVector, weights)
			result = 1 if sumFunction(inputVector, weights, b) > 0 else 0
			'''p = sumFunction(inputVector, weights, b)
			if(p > 0) :
				result = 1
			elif (p <=0) :
				result = -1'''
			#result = 1 if dot_product(inputVector, weights, b) > 0 else 0 
			error = desiredOutput - result
			if error != 0:
				errorCount += 1
				for index, value in enumerate(inputVector):
					weights[index] += (learning_rate * error * value)
					#print weights[index]
					b = b+(learning_rate * error)
				updates += 1
			#trainCount +=1

		#print sumFunction(inputVector, weights, b)
		text_file.write("%.7g \n" % sumFunction(inputVector, weights, b))
		if errorCount == 0 or passes > 100:
			break
		else:
			passes += 1
			
	print "\n\nNumber of passes: " , passes
	print "Number of updates: " , updates
	#print trainCount
	#print "Time: ", (datetime.now()-start)
	
	print "\nDuring Training....."
	print "\nTotal no. of Error: ", errorCount
	print "Error Percentage: ", (errorCount/float(updates))*100
	#print "Error Percentage: ", (errorCount/updates)*100	
	print "Efficiency: ", ((float(updates) - errorCount)/float(updates))*100

	return weights, b
	
	
'''
    The testing loop classifies each datum as the label that most closely matches the training vector
    for that label.  
     
'''

def testPerceptron(eta,X_test, weights,text_file):

	xSample, xLength = X_test.shape
	errorCount = 0.0
	testCount = 0.0
	learning_rate = float(eta)
	b = 0
	#start = datetime.now()
	
	for row in X_test:
		temp = row[0:xLength]
		inputVector = temp[:-1]
		#print inputVector 
		desiredOutput = row[-1]
		#print desiredOutput
		result = 1 if sumFunction(inputVector, weights, b) > 0 else 0
		#result = 1 if dot_product(inputVector, weights) > 0 else 0 
		text_file.write("%f \n" % sumFunction(inputVector, weights, b))
		#print sumFunction(inputVector, weights, b)
		error = desiredOutput - result
		if error != 0:
			errorCount += 1
			for index, value in enumerate(inputVector):
					weights[index] += (learning_rate * error * value)
					b = b + (learning_rate * error)
		
		testCount += 1
		
	print "\nDuring Test....."
	print "\nTotal no. of Error: ", errorCount
	print "Total # of Test: ", testCount	
	#print "Done testing in ", (datetime.now()-start)
	print "Error Percentage: ", (errorCount/testCount)*100	
	print "Efficiency: ", ((testCount - errorCount)/testCount)*100
	print "\n"
	


def csv_load(data_set_train, data_set_test=None): # loads the csv file and puts the data into numpy arrays
	
	def load_helper(data):
	
		print "Loading Data from", data
		
		with open(data, 'r+') as csv_file: # creates attribute tuple and labels array
			reader = csv.reader(csv_file)
			features = list(reader.next())
			features.pop() # removes 'class' feature name
			#print "features: ", features

		# load the csv data into a numpy array
		X = np.loadtxt(data, delimiter=",", dtype='int', skiprows=1)
		#print "X: ",X
		#y = X[:,-1:] # get only the labels
		#print "Y: ", y
		#y = y.flatten() # make the single column 1 dimensional
		#print "Y: ", y
		#X = X[:,:-1] # remove the labels column
		#print "X: ",X
		#return features, X, y
		return features, X


	#features, X_train, y_train = load_helper(data_set_train)
	features, X_train = load_helper(data_set_train)
	if data_set_test is not None:
		#dummy, X_test, y_test = load_helper(data_set_test)
		dummy, X_test = load_helper(data_set_test)
	else:
		X_test = None
		#y_test = None

	#return features, X_train, y_train, X_test, y_test # returns test and training data split into separated numpy arrays with their repective labels
	return features, X_train, X_test # returns test and training data split into separated numpy arrays with their repective labels


def main(train_file, test_file, eta, model_file):
	
	text_file = open("testing_activation.txt", "w")
	text_file1 = open("training_activation.txt", "w")
	model = open(model_file, "w+")
	#features, X_train, y_train, X_test, y_test = csv_load(train_file, test_file) # opens and loads csv files
	features, X_train, X_test = csv_load(train_file, test_file) # opens and loads csv files
	#print X_test.shape
	#print X_train, y_train
	#perceptron = MyPerceptron(eta, 1)
	#perceptron.fit(features, X_train, y_train)
	#test_perceptron_correctness(features, X_test, y_test, eta)
	#print eta
	#trainPerceptron(eta, X_train, y_train)
	w, bias =trainPerceptron(eta, X_train,text_file1)
	model.write("%f \n" % bias)
	for index, value in enumerate(features):
		#print features[index], w[index]
		model.write("%s %f \n" % (features[index], w[index]))
	#print w
	testPerceptron(eta,X_test, w,text_file)
	model.close()


if __name__ == '__main__':
	
	args = sys.argv[1:] # Get the filename parameters from the command line
	main( *args )
	
