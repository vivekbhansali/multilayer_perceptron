INSTRUCTIONS for trainMLP.py:

Usage: python trainMLP.py <datafile.csv>

Train a 2-5-4 multi-layer perceptron on the dataset,
plot the learning curve and save the weights after iterations 
0, 10, 100, 1000 and 10000 to the files wt_datafile_i, where 
datafile is the name of the input file provided and i  is the 
number of iterations.

Datafile format:
x1,y1,c1
x2,y2,c2
...


INSTRUCTIONS for executeMLP.py:

Usage: python executeMLP.py <weightfile> <datafile>

Load the weights from given weightfile to recreate the MLP, 
classify the dataset, reporting number of correct and incorrect
classifications, recognition rate and the confusion matrix.
It then plots the given dataset and also plots the classification 
regions.

Note: the weightfiles are auto-generated using pickle module