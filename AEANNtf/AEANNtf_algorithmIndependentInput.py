"""AEANNtf_algorithmIndependentInput.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see AEANNtf_main.py

# Usage:
see ANNtf2.py

# Description:
AEANN algorithm independent input - define autoencoder generated artificial neural network

Greedy layer construction using autoencoders

(intermediate layer dimensionality reduction via autoencoder bottleneck requires redundancy in input data, e.g. image)

"""

import tensorflow as tf
import numpy as np
import copy
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import ANNtf2_algorithmLIANN_math	#required for supportDimensionalityReduction:useCorrelationMatrix
np.set_printoptions(suppress=True)

learningAlgorithmAEANN = False #AEANN backprop; default algorithm
learningAlgorithmLIANN = True	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmNone = False	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
#learningAlgorithmRUANNSUANN = False	#incomplete #perform stocastic update of weights (SUANN) based on RUANN (hypothetical AEANN hidden layer neuron activation/relaxation state modifications)

debugSingleLayerOnly = False
debugFastTrain = False	#not supported
debugSmallBatchSize = False	#not supported #small batch size for debugging matrix output

generateVeryLargeNetwork = False
if(learningAlgorithmAEANN):
	supportDimensionalityReduction = True	#optional	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN) - this dimensionality reduction method is designed to be used in combination with a large autoencoder hidden layer (> input/output layer), as opposed to a small (bottlenecked) autoencoder hidden layer
elif(learningAlgorithmLIANN):
	generateVeryLargeNetwork = True
	supportDimensionalityReduction = True	#mandatory	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN)
elif(learningAlgorithmNone):
	generateVeryLargeNetwork = True
	supportDimensionalityReduction = False
		
supportSkipLayers = True #fully connected skip layer network
if(supportDimensionalityReduction):
	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
	useCorrelationMatrix = True
	maxCorrelation = 0.95	#requires tuning

largeBatchSize = False	#not supported	#else train each layer using entire training set
generateLargeNetwork = True	#required #CHECKTHIS: autoencoder does not require bottleneck
generateNetworkStatic = False	#optional
generateDeepNetwork = True	#optional
if(generateDeepNetwork):
	generateNetworkStatic = True	#True: autoencoder requires significant number of neurons to retain performance?
	
#forward excitatory connections;
Wf = {}
Wb = {}
B = {}
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global batchSize
	
	if(debugSmallBatchSize):
		batchSize = 10
		learningRate = 0.001
	else:
		if(largeBatchSize):
			batchSize = 1000	#current implementation: batch size should contain all examples in training set
			learningRate = 0.05
		else:
			batchSize = 100	#3	#100
			learningRate = 0.005
	if(generateDeepNetwork):
		numEpochs = 100	#higher num epochs required for convergence
	else:
		numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
		
	if(generateVeryLargeNetwork):
		firstHiddenLayerNumberNeurons = num_input_neurons*10
	else:
		if(generateLargeNetwork):
			firstHiddenLayerNumberNeurons = num_input_neurons*3
		else:
			firstHiddenLayerNumberNeurons = num_input_neurons
	if(debugSingleLayerOnly):
		numberOfLayers = 1
	else:
		if(generateDeepNetwork):
			numberOfLayers = 3
		else:
			numberOfLayers = 2
			
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
			
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			#forward excitatory connections;
			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						WlayerF = randomNormal([n_h[l2], n_h[l1]]) 
						WlayerB = randomNormal([n_h[l1], n_h[l2]]) 
						Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(WlayerF)
						Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = tf.Variable(WlayerB)			
			else:
				WlayerF = randomNormal([n_h[l1-1], n_h[l1]]) 
				WlayerB = randomNormal([n_h[l1], n_h[l1-1]]) 
				Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(WlayerF)
				Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = tf.Variable(WlayerB)
				
			Blayer = tf.zeros(n_h[l1])
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(Blayer)

			if(supportSkipLayers):
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				
def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf2)
	return neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=networkIndex)
	#return neuralNetworkPropagationAEANNtest(x, networkIndex=1)

def neuralNetworkPropagationAEANNautoencoderLayer(x, layer, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder=True, layer=layer, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=1):
	pred, target = neuralNetworkPropagationAEANN(x, autoencoder=False, layer=numberOfLayers, networkIndex=networkIndex)
	return pred
	
#not currently used;
#def neuralNetworkPropagationAEANNtest(x, networkIndex=1):
#   return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=None, networkIndex=networkIndex)
#def neuralNetworkPropagationAEANNtestLayer(x, l, networkIndex=1):
#   return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=l, networkIndex=networkIndex)


	
def neuralNetworkPropagationAEANNdimensionalityReduction(x, networkIndex=1):

	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
		
	for l1 in range(1, numberOfLayers):	#ignore first/last layer
		
		A, Z, outputTarget = neuralNetworkPropagationLayerForward(l1, AprevLayer, False, networkIndex)

		Atransposed = tf.transpose(A)
		if(useCorrelationMatrix):
			correlationMatrix = ANNtf2_algorithmLIANN_math.calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=0.0, getOffDiagonalCorrelationMatrix=True)	#off diagonal correlation matrix is required so that do not duplicate k1->k2 and k2->k1 correlations	#CHECKTHIS: nanReplacementValue
			#print("correlationMatrix = ", correlationMatrix)
			#print("correlationMatrix.shape = ", correlationMatrix.shape)
		
		if(useCorrelationMatrix):
			k1maxCorrelation = correlationMatrix.max(axis=0)
			k2maxCorrelation = correlationMatrix.max(axis=1)
			k1maxCorrelation = np.maximum(k1maxCorrelation, k2maxCorrelation)
			#k1maxCorrelationIndex = correlationMatrix.argmax(axis=0)	#or axis=1
			
			k1maxCorrelation = tf.convert_to_tensor(k1maxCorrelation)
		else:
			#incomplete;
			for k1 in range(n_h[l1]):
				#calculate maximum correlation;
				k1maxCorrelation = 0.0
				for k2 in range(n_h[l1]):
					if(k1 != k2):
						Ak1 = Atransposed[k1]	#Ak: 1d vector of batchsize
						Ak2 = Atransposed[k2]	#Ak: 1d vector of batchsize
						k1k2correlation = calculateCorrelation(Ak1, Ak2)	#undefined
			
		#generate masks (based on highly correlated k/neurons);
		#print("k1maxCorrelation = ", k1maxCorrelation)
		k1PassArray = tf.less(k1maxCorrelation, maxCorrelation)
		k1FailArray = tf.logical_not(k1PassArray)
		#print("k1PassArray = ", k1PassArray)
		#print("k1FailArray = ", k1FailArray)
		k1PassArrayF = tf.expand_dims(k1PassArray, axis=0)
		k1FailArrayF = tf.expand_dims(k1FailArray, axis=0)
		k1PassArrayB = tf.expand_dims(k1PassArray, axis=1)
		k1FailArrayB = tf.expand_dims(k1FailArray, axis=1)
		k1PassArrayF = tf.cast(k1PassArrayF, tf.float32)
		k1FailArrayF = tf.cast(k1FailArrayF, tf.float32)
		k1PassArrayB = tf.cast(k1PassArrayB, tf.float32)
		k1FailArrayB = tf.cast(k1FailArrayB, tf.float32)
					
		#apply masks to weights (randomise specific k/neurons);					
		if(supportSkipLayers):
			for l2 in range(0, l1):
				if(l2 < l1):
					#randomize or zero
					if(supportDimensionalityReductionRandomise):
						WlayerFrand = randomNormal([n_h[l2], n_h[l1]])
						WlayerBrand = randomNormal([n_h[l1], n_h[l2]])
					else:
						WlayerFrand = tf.zeros([n_h[l2], n_h[l1]], dtype=tf.dtypes.float32)
						WlayerBrand = tf.zeros([n_h[l1], n_h[l2]], dtype=tf.dtypes.float32)				
					WlayerF = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
					WlayerB = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]
					WlayerFrand = tf.multiply(WlayerFrand, k1FailArrayF)
					WlayerBrand = tf.multiply(WlayerBrand, k1FailArrayB)
					#print("WlayerFrand + ", WlayerFrand)
					WlayerF = tf.multiply(WlayerF, k1PassArrayF)
					WlayerB = tf.multiply(WlayerB, k1PassArrayB)
					#print("WlayerF + ", WlayerF)
					WlayerF = tf.add(WlayerF, WlayerFrand)
					WlayerB = tf.add(WlayerB, WlayerBrand)
					Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = WlayerF
					Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = WlayerB
		else:
			if(supportDimensionalityReductionRandomise):
				WlayerFrand = randomNormal([n_h[l1-1], n_h[l1]]) 
				WlayerBrand = randomNormal([n_h[l1], n_h[l1-1]])
			else:
				WlayerFrand = tf.zeros([n_h[l1-1], n_h[l1]], dtype=tf.dtypes.float32)
				WlayerBrand = tf.zeros([n_h[l1], n_h[l1-1]], dtype=tf.dtypes.float32)		
			WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
			WlayerB = Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")]
			WlayerFrand = tf.multiply(WlayerFrand, k1FailArrayF)
			WlayerBrand = tf.multiply(WlayerBrand, k1FailArrayB)
			WlayerF = tf.multiply(WlayerF, k1PassArrayF)
			WlayerB = tf.multiply(WlayerB, k1PassArrayB)
			WlayerF = tf.add(WlayerF, WlayerFrand)
			WlayerB = tf.add(WlayerB, WlayerBrand)
			Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = WlayerF
			Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = WlayerB
								
		AprevLayer = A
		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
			
def neuralNetworkPropagationAEANN(x, autoencoder, layer, networkIndex=1):

	outputPred = None
	outputTarget = None

	outputPred = x #in case layer=0
	
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	if(autoencoder):
		maxLayer = layer
	else:
		if(layer is None):
			maxLayer = numberOfLayers
		else:
			maxLayer = layer
		
	for l1 in range(1, maxLayer+1):

		A, Z, outputTarget = neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder, networkIndex)

		if(autoencoder):
			if(l1 == numberOfLayers):
				outputPred = tf.nn.softmax(Z)
			else:
				if(l1 == layer):
					outputPred = neuralNetworkPropagationLayerBackwardAutoencoder(l1, A, networkIndex)
		else:
			if(l1 == numberOfLayers):
				outputPred = tf.nn.softmax(Z)
			else:
				outputPred = A
				
		A = tf.stop_gradient(A)	#only train weights for layer l1

		AprevLayer = A
		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
			
	return outputPred, outputTarget

def neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder, networkIndex=1):
	outputTarget = None
	
	if(supportSkipLayers):
		if(autoencoder):
			outputTargetList = []
		Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
		for l2 in range(0, l1):
			WlayerF = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
			Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")]))	
			if(autoencoder):
				outputTargetListPartial = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
				outputTargetList.append(outputTargetListPartial)
		if(autoencoder):
			outputTarget = tf.concat(outputTargetList, axis=1)
			#print("outputTarget.shape = ", outputTarget.shape)
	else:	
		WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
		if(autoencoder):
			outputTarget = AprevLayer
		Z = tf.add(tf.matmul(AprevLayer, WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")])
	A = activationFunction(Z)
	
	return A, Z, outputTarget

def neuralNetworkPropagationLayerBackwardAutoencoder(l1, A, networkIndex=1):
	if(supportSkipLayers):
		outputPredList = []
		for l2 in range(0, l1):
			WlayerB = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]
			Zback = tf.matmul(A, WlayerB)
			outputPredPartial = tf.nn.sigmoid(Zback)
			#print("outputPredPartial.shape = ", outputPredPartial.shape)
			outputPredList.append(outputPredPartial)
		outputPred = tf.concat(outputPredList, axis=1)
		#print("outputPred.shape = ", outputPred.shape)
	else:
		WlayerB = Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")]
		Zback = tf.matmul(A, WlayerB)
		outputPred = tf.nn.sigmoid(Zback)
		
	return outputPred
		

def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
