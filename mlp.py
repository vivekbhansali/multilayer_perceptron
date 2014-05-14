# Foundation of Intelligent Systems
# Project 2 : Recycling Used Parts for Fun and Profit
# Team: Ronit Galani, Ruchin Shah, Vivek Bhansali

import pickle
import random
import commonMLP

# this class realizes a multi-layer perceptron
# with 2 input nodes, 5 hidden nodes (only 1 hidden layer)
# and 4 output nodes
class MLP:
	def __init__(self, noInputs, noHidden, noOutputs, logfn, dLogfn, alpha=0.1):
		# 1st node is bias
		self._noInputs = noInputs + 1
		# 1st node is bias
		self._noHidden = noHidden + 1
		self._noOutputs = noOutputs

		# logistic function
		self._logfn = logfn
		# derivation of logistic function
		self._dLogfn = dLogfn
		# learning rate
		self._alpha = alpha

		# weights between the nodes of output layer
		# and hidden layer
		self._outWts = []
		# weights between the nodes of hidden layer
		# and input layer
		self._hidWts = []

		# randomly assign weights
		for i in range(0, self._noOutputs):
			self._outWts.append([])
			for j in range(0, self._noHidden):
				# initial wt random
				(self._outWts[i]).append(random.uniform(-1.0, 1.0))

		for i in range(0, self._noHidden):
			self._hidWts.append([])
			for j in range(0, self._noInputs):
				# initial wt random
				(self._hidWts[i]).append(random.uniform(-1.0, 1.0))


	# assign weights from file
	def assignWts(self, wtfile):
		# try:	
		wt_file = open(wtfile, 'rb')
		wtdata = pickle.load(wt_file)
		self._hidWts = wtdata[0]
		self._outWts = wtdata[1]
		wt_file.close()
		# except:
			# print "Error: Unable to read file:", wtfile
	        # exit(-1)

	# update weights by back-propogating the errors
	def updateWts(self, ins, outs):
		(vIn, vHid, vOut, dHid, dOut) = self.backProp(ins, outs)

		# update weights between output layer and hidden layer
		for i in range(0, self._noOutputs):
			for j in range(0, self._noHidden):
				self._outWts[i][j] = self._outWts[i][j] - (self._alpha * dOut[i] * vHid[j])

		# update weights between hidden layer and input layer
		for i in range(1, self._noHidden):
			for j in range(0, self._noInputs):
				self._hidWts[i][j] = self._hidWts[i][j] - (self._alpha * dHid[i] * vIn[j])


	# calculate output for a particular input
	def output(self, ins):
		# 1st node is bias
		vIn = [1.0] + ins
		# 1st node is bias - always 1
		vHid = [1.0] + [0.0] * (self._noHidden-1)

		vOut = [0.0] * self._noOutputs

		# calculate output of nodes of hidden layer
		for i in range(1, len(vHid)):
			for j in range(0, len(vIn)):
				vHid[i] = vHid[i] + (self._hidWts[i][j] * vIn[j])
			vHid[i] = self._logfn(vHid[i])

		# calculate output of nodes of output layer
		for i in range(0, len(vOut)):
			for j in range(0, len(vHid)):
				vOut[i] += self._outWts[i][j] * vHid[j]
			vOut[i] = self._logfn(vOut[i])

		# return outputs of all three layers
		return (vIn, vHid, vOut)


	# calculate output and then error at ouput nodes
	# and then backprogate to find error at nodes of hidden
	# layer
	def backProp(self, ins, outs):
		(vIn, vHid, vOut) = self.output(ins)

		dHid = [0.0] * self._noHidden
		dOut = [0.0] * self._noOutputs

		for i in range(0, len(dOut)):
			dOut[i] = (vOut[i] - outs[i]) * self._dLogfn(vOut[i])

		for i in range(1, len(dHid)):
			t = 0.0
			for j in range(0, len(dOut)):
				t = t + (dOut[j] * self._outWts[j][i])

			dHid[i] = t * self._dLogfn(vHid[i])
		
		# returns output as well as errors 
		return (vIn, vHid, vOut, dHid, dOut)


	# train the MLP using the given data for given no of iterations
	def train(self, ins, outs, infile, epochs=10000):
		outfile = "wt_" + infile.split('.')[0]  + "_"
		# outfile = "wt_"

		# dict to store epoch number to sum of squared error 
		epochSqErr = {}

		# save initial weights (i.e. epoch = 0) to file
		output = open((outfile + str(0)), 'wb')
		wts = [self._hidWts, self._outWts]
		pickle.dump(wts, output)
		output.close()

		for e in range(0, epochs):
			# iterate over dataset
			for i in range(0, len(ins)):
				# update weights after each sample
				self.updateWts(ins[i], outs[i])

			# calculate sum of squared errors
			sqErr = self.squaredError(ins, outs)
			epochSqErr[(e+1)] = sqErr

			# save weights after 10, 100, 1000, 10000 iterations
			try:
				idx = commonMLP.save_at.index(e+1)
				# save weights in a file
				# print commonMLP.save_at[idx]
				outfile1 = outfile + str(commonMLP.save_at[idx])
				output = open(outfile1, 'wb')
				wts = [self._hidWts, self._outWts]
				pickle.dump(wts, output)
				output.close()
			except:
				continue

		# return sum of squared errors for plotting
		return epochSqErr

	# calculates sum of squared errors for given input and output
	def squaredError(self, ins, outs):
		sqErr = 0

		for i in range(0, len(ins)):
			vOut = self.output(ins[i])[2]
			e2 = 0
			for j in range(0, len(outs[i])):
				e1 = (vOut[j] - outs[i][j]) ** 2
				e2 = e2 + e1

			sqErr = sqErr + e2

		return sqErr