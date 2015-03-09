to run:

type in your terminal:

$ python perceptron.py training_set.csv test_set.csv eta output.txt

The third argument is eta which takes a float. It creates an output file with the name passed as the fourth argument. It has the bias
and weight of the training data. Two other files training_activation.txt and test_activation. txt is created with the values of training 
and test activation respectively.

The std output has the following results shown for eta = 0.1
 
Number of passes:  101
Number of updates:  39632

During Training.....

Total no. of Error:  392
Error Percentage:  0.9890997174
Efficiency:  99.0109002826

During Test.....

Total no. of Error:  103.0
Total # of Test:  1000.0
Error Percentage:  10.3
Efficiency:  89.7
