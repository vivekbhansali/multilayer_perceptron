# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

# This file contains some commonly used functions for
# trainMLP and execMLP

import math

# pts at which weights need to be saved
save_at = [0, 10, 100, 1000, 10000]

# sigmoid function
def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-1.0 * x) )

# derivative of sigmoid function
def dSigmoid(x):
	return x * (1.0 - x)

# read the dataset and form input and output lists
def readDataset(filename):
    try:
        with open(filename, 'rb') as f:
            inputs  = []
            outputs = []

            # Read the file, strip any whitespace.
            filedata = f.read().replace(" ", "")

            # Read each line, parse and cast.
            for line in filedata.split():
            	# TO-DO: modularize this:
                (sym, ecc, cls) = line.split(',')
                inputs.append([float(sym), float(ecc)])
                # 4 classes of outputs
                temp_out = [0] * 4
                temp_out[int(cls)-1] = 1
                outputs.append(temp_out)
            
            return (inputs, outputs)

    except:
        print "Error: Unable to read file ", filename
        exit(-1)