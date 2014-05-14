# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

import sys
import mlp
import commonMLP
import matplotlib.pylab as plt
import numpy as np

# profit function
def profitFunction(assigned, correct):
	profit = [[ 0.20, -0.07, -0.07, -0.07],
			  [-0.07,  0.15, -0.07, -0.07],
			  [-0.07, -0.07,  0.05, -0.07],
			  [-0.03, -0.03, -0.03, -0.03]]
	return profit[assigned][correct]

# prints formatted confusion matrix
def printConfusionMatrix(confMatrix):
	print '%20s %30s' % ('', 'Actual class')
	print '%20s %10s %10s %10s %10s' % ('Assigned Class', 'bolt', 'nut', 'ring', 'scrap')
	print '%20s %10d %10d %10d %10d' % ( 'bolt', confMatrix[0][0], confMatrix[0][1], confMatrix[0][2], confMatrix[0][3] )
	print '%20s %10d %10d %10d %10d' % (  'nut', confMatrix[1][0], confMatrix[1][1], confMatrix[1][2], confMatrix[1][3] )
	print '%20s %10d %10d %10d %10d' % ( 'ring', confMatrix[2][0], confMatrix[2][1], confMatrix[2][2], confMatrix[2][3] )
	print '%20s %10d %10d %10d %10d' % ('scrap', confMatrix[3][0], confMatrix[3][1], confMatrix[3][2], confMatrix[3][3] )

# calculates number of correctly and incorrectly 
# classified samples and recognition rate and 
# total profit and the confusion matrix
def calcStatistics(emlp, ins, outs):
	noRight = 0
	noWrong = 0
	profit  = 0

	confusionMatrix = [[0, 0, 0, 0],
			           [0, 0, 0, 0],
			           [0, 0, 0, 0],
			           [0, 0, 0, 0]]

	for i in range(0, len(ins)):
		# calculate ouput
		vOut = emlp.output(ins[i])[2]
		# check if correct
		if(equal(vOut, outs[i])):
			noRight += 1
		else:
			noWrong += 1

		# build confusion matrix
		assigned = vOut.index(max(vOut))
		correct  = outs[i].index(1)
		confusionMatrix[assigned][correct] += 1  
		# calculate profit
		profit += profitFunction(assigned, correct)

	return (noRight, noWrong, profit, confusionMatrix)

# returns true if predicted output is true
# note both arguments are vectors (lists)
# right argument should have a single 1
def equal(assignedOut, actualOut):
	# find max and match its index
	assigned = assignedOut.index(max(assignedOut))
	actual   = actualOut.index(1)

	if (assigned == actual):
		return True
	else:
		return False

# plot the MLP's clasifier
def plotClassifier(emlp):
	boltX = []
	boltY = []
	boltColor = 'bo'
	boltOutput= [1, 0, 0, 0]
	boltLabel = 'Bolt'

	nutX = []
	nutY = []
	nutColor = 'yo'
	nutOutput= [0, 1, 0, 0]
	nutLabel = 'Nut'

	ringX = []
	ringY = []
	ringColor = 'go'
	ringOutput= [0, 0, 1, 0]
	ringLabel = 'Ring'

	scrapX = []
	scrapY = []
	scrapColor = 'ro'
	scrapOutput= [0, 0, 0, 1]
	scrapLabel = 'Scrap'

	for x in np.arange(0.0, 1.0, 0.01):
		for y in np.arange(0.0, 1.0, 0.01):
			tout = emlp.output([x,y])[2]

			if (equal(tout, boltOutput)):
				boltX.append(x)
				boltY.append(y)

			elif (equal(tout, nutOutput)):
				nutX.append(x)
				nutY.append(y)

			elif (equal(tout, ringOutput)):
				ringX.append(x)
				ringY.append(y)

			elif (equal(tout, scrapOutput)):
				scrapX.append(x)
				scrapY.append(y)

	# plot all four
	fig, ax = plt.subplots()
	ax.plot(boltX, boltY, boltColor, label=boltLabel)
	ax.plot(nutX, nutY, nutColor, label=nutLabel)
	ax.plot(ringX, ringY, ringColor, label=ringLabel)
	ax.plot(scrapX, scrapY, scrapColor, label=scrapLabel)


	# Shink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

	# Put a legend below current axis
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=5)

	plt.xlabel('X Axis - Six-fold rotational symmetry')
	plt.ylabel('Y Axis - Eccentricity') 
	plt.title('MLP Classification')
	plt.show()

# plot initial dataset
def plotDataset(ins, outs):
	boltX = []
	boltY = []
	boltColor = 'bo'
	boltLabel = 'Bolt'

	nutX = []
	nutY = []
	nutColor = 'yo'
	nutLabel = 'Nut'

	ringX = []
	ringY = []
	ringColor = 'go'
	ringLabel = 'Ring'

	scrapX = []
	scrapY = []
	scrapColor = 'ro'
	scrapLabel = 'Scrap'

	for i in range(len(outs)):
		outClass = (outs[i]).index(1) 
		if (outClass == 0):
			boltX.append(ins[i][0])
			boltY.append(ins[i][1])

		elif (outClass == 1):
			nutX.append(ins[i][0])
			nutY.append(ins[i][1])

		elif (outClass == 2):
			ringX.append(ins[i][0])
			ringY.append(ins[i][1])

		elif (outClass == 3):
			scrapX.append(ins[i][0])
			scrapY.append(ins[i][1])

	# plot all four
	fig, ax = plt.subplots()
	ax.plot(boltX, boltY, boltColor, label=boltLabel)
	ax.plot(nutX, nutY, nutColor, label=nutLabel)
	ax.plot(ringX, ringY, ringColor, label=ringLabel)
	ax.plot(scrapX, scrapY, scrapColor, label=scrapLabel)


	# Shink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

	# Put a legend below current axis
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=5)

	plt.xlabel('X Axis - Six-fold rotational symmetry')
	plt.ylabel('Y Axis - Eccentricity') 
	plt.title('Distribution of input dataset')
	plt.show()

# main
if __name__ == "__main__":
	
	if(len(sys.argv) != 3):
	    print "Usage: python execMLP.py <weightfile> <datafile>"
	    exit(-1)

	wtfile = sys.argv[1]
	datafile = sys.argv[2]
	# testfile = sys.argv[2]

	(insData, outsData) = commonMLP.readDataset(datafile)
	# (insTest, outsTest) = commonMLP.readDataset(testfile)

	emlp = mlp.MLP(2, 5, 4, commonMLP.sigmoid, commonMLP.dSigmoid)

	# for i in range(0, len(commonMLP.save_at)):
	# for i in range(0, 1):
		# wtfile2 = wtfile + str(commonMLP.save_at[i])
	
	# load weights from file and assign to MLP
	emlp.assignWts(wtfile)

	# print (str(i+1) + ")"),
	# print "After", str(commonMLP.save_at[i]), "epochs:\n"

	# print statistics
	print "Statistics:\n"
	(nR, nW, profit, confMatrix) = calcStatistics(emlp, insData, outsData)
	print "No of correctly classified samples: ", nR
	print "No of incorrectly classified samples", nW
	print "Recognition rate: ", 
	p = (float(nR)/float(nW+nR))*100
	print("%.2f" % p), "%"
	print "Profit obtained:", profit
	print "Confusion Matrix:"
	printConfusionMatrix(confMatrix)
	print "________________________________________________________________\n"
	
	# plot initial dataset 
	plotDataset(insData, outsData)
	# plot classifier
	plotClassifier(emlp)


