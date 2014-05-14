# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

import sys
import mlp
import commonMLP
import matplotlib.pylab as plt

# plot the graph of sum of squared errors vs epoch
def plotSqErr(epochSqErr):
	epoch = epochSqErr.keys()
	sqerr = epochSqErr.values()

	plt.plot(epoch, sqerr, 'b-')
	plt.xlabel('Epochs')
	plt.ylabel('Sum of Squared Error') 
	plt.show()

# main
if __name__ == "__main__":
	
	if(len(sys.argv) != 2):
	    print "Usage: python trainMLP.py <datafile>"
	    exit(-1)

	datafile = sys.argv[1]
	# read dataset
	(ins, outs) = commonMLP.readDataset(datafile)
	# create MLP object
	myMlp = mlp.MLP(2, 5, 4, commonMLP.sigmoid, commonMLP.dSigmoid)

	# train MLP
	epochSqErr = myMlp.train(ins, outs, datafile)
	# plot
	plotSqErr(epochSqErr)