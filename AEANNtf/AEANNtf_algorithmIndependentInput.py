"""AEANNtf_algorithmIndependentInput.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see AEANNtf_main.py

# Usage:
see AEANNtf_main.py

# Description:
AEANNtf algorithm independent input - define autoencoder generated artificial neural network for independent input (non-sequential)

Greedy layer construction using autoencoders

(intermediate layer dimensionality reduction via autoencoder bottleneck requires redundancy in input data, e.g. image)

"""

import tensorflow as tf
import numpy as np
import copy
from ANNtf_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf_operations
import ANNtf_globalDefs
#import LIANNtf_algorithmLIANN_math	#required for supportDimensionalityReduction
np.set_printoptions(suppress=True)

#debug parameters;
debugSmallNetwork = False	#not supported #small network for debugging matrix output
debugSmallBatchSize = False	#not supported #small batch size for debugging matrix output
debugSingleLayerOnly = False
debugFastTrain = False	#not supported

#select learningAlgorithm:
learningAlgorithmAEANN = True #AEANN backprop; default algorithm
#learningAlgorithmLIANN = False	#support disabled (see LIANNtf_algorithmLIANN)	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmNone = False	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
#learningAlgorithmRUANNSUANN = False	#incomplete #perform stocastic update of weights (SUANN) based on RUANN (hypothetical AEANN hidden layer neuron activation/relaxation state modifications)

#intialise network properties (configurable);	
autoencoderPrediction = "previousLayer"	#autoencoder (backwards connections) predicts previous layer	#orig AEANNtf/AEANNpt implementation
#autoencoderPrediction = "inputLayer" 	#autoencoder (backwards connections) predicts input layer 	#orig AEORtf autoencoder_simulation2 implementation
supportSkipLayers = True #fully connected skip layer network
if(supportSkipLayers):
	autoencoderPrediction = "allPreviousLayers"		#optional	#orig AEANNtf implementation
	supportSkipLayersF = True
	if(autoencoderPrediction == "allPreviousLayers"):
		supportSkipLayersB = False
	else:
		supportSkipLayersB = True	#optional	#add full connectivity to decoder	#not in orig AEANNtf/AEANNpt implementation
else:
	supportSkipLayersF = False
	supportSkipLayersB = False
	
supportMultipleNetworks = True	#optional (required to activate set trainMultipleNetworks=True in AEANNtf_main)


#intialise network properties;
generateLargeNetwork = True	#required #CHECKTHIS: autoencoder does not require bottleneck	#for default AEANN operations
largeBatchSize = False	#not supported	#else train each layer using entire training set
generateNetworkStatic = False	#optional
generateDeepNetwork = True	#optional	#used for algorithm testing
if(generateDeepNetwork):
	generateNetworkStatic = True	#True: autoencoder requires significant number of neurons to retain performance?

#learning algorithm customisation;
generateVeryLargeNetwork = False
if(learningAlgorithmAEANN):
	pass
	#supportDimensionalityReduction = True	#optional	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN) - this dimensionality reduction method is designed to be used in combination with a large autoencoder hidden layer (> input/output layer), as opposed to a small (bottlenecked) autoencoder hidden layer
elif(learningAlgorithmNone):
	#can pass different task datasets through a shared randomised net
	generateVeryLargeNetwork = True
	#supportDimensionalityReduction = False
#elif(learningAlgorithmLIANN):
#	if(not debugSmallNetwork):
#		generateVeryLargeNetwork = True	#default: True
#	supportDimensionalityReduction = True	#mandatory	#correlated neuron detection; dimensionality reduction via neuron atrophy or weight reset (see LIANN)

if(generateVeryLargeNetwork):
	generateLargeNetworkRatio = 100	#100	#default: 10
else:
	if(generateLargeNetwork):
		generateLargeNetworkRatio = 3
	else:
		generateLargeNetworkRatio = 1

#supportDimensionalityReductionLimitFrequency = False
#if(supportDimensionalityReduction):
#	supportDimensionalityReductionRandomise	= True	#randomise weights of highly correlated neurons, else zero them (effectively eliminating neuron from network, as its weights are no longer able to be trained)
#	maxCorrelation = 0.95	#requires tuning
#	supportDimensionalityReductionLimitFrequency = True
#	if(supportDimensionalityReductionLimitFrequency):
#		supportDimensionalityReductionLimitFrequencyStep = 1000


#network/activation parameters;
#forward excitatory connections;
Wf = {}
Wb = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
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
		
	firstHiddenLayerNumberNeurons = num_input_neurons*generateLargeNetworkRatio
	if(debugSingleLayerOnly):
		numberOfLayers = 1
	else:
		if(generateDeepNetwork):
			numberOfLayers = 3
		else:
			numberOfLayers = 2
			
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf_operations.defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
			
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
					if(supportSkipLayersF):
						WlayerF = randomNormal([n_h[l2], n_h[l1]]) 
						Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(WlayerF)
					if(autoencoderPrediction=="allPreviousLayers"):
						WlayerB = randomNormal([n_h[l1], n_h[l2]]) 
						Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = tf.Variable(WlayerB)	
					elif(supportSkipLayersB):
						if(autoencoderPrediction=="previousLayer"):
							l3 = l1-1
						elif(autoencoderPrediction=="inputLayer"):
							l3 = 0
						WlayerB = randomNormal([n_h[l2+1], n_h[l3]]) 
						Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = tf.Variable(WlayerB)	
				if(not supportSkipLayersB and autoencoderPrediction!="allPreviousLayers"):
					if(autoencoderPrediction=="previousLayer"):
						l2 = l1-1
					elif(autoencoderPrediction=="inputLayer"):
						l2 = 0
					WlayerB = randomNormal([n_h[l1], n_h[l2]]) 
					Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = tf.Variable(WlayerB)
			else:
				l2 = l1-1
				WlayerF = randomNormal([n_h[l2], n_h[l1]]) 
				if(autoencoderPrediction=="previousLayer"):
					l2 = l1-1
				elif(autoencoderPrediction=="inputLayer"):
					l2 = 0
				WlayerB = randomNormal([n_h[l1], n_h[l2]]) 
				Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(WlayerF)
				Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = tf.Variable(WlayerB)
				
			Blayer = tf.zeros(n_h[l1])
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(Blayer)

			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
	
	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			Blayer = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer	= tf.Variable(Blayer)	#not currently used
			
def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf)
	return neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=networkIndex)
	#return neuralNetworkPropagationAEANNtest(x, networkIndex=1)

#if(ANNtf_algorithm.supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
   pred, _ = neuralNetworkPropagationAEANN(x, autoencoder=False, layer=l, networkIndex=networkIndex)
   return pred
#def neuralNetworkPropagationAEANNtestLayer(x, l, networkIndex=1):
#   return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=l, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNautoencoderLayer(x, layer, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder=True, layer=layer, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=1):
	pred, target = neuralNetworkPropagationAEANN(x, autoencoder=False, layer=numberOfLayers, networkIndex=networkIndex)
	return pred
	

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred
	
#def neuralNetworkPropagationAEANNdimensionalityReduction(x, networkIndex=1):
#
#	AprevLayer = x
#	Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
#		
#	for l1 in range(1, numberOfLayers):	#ignore first/last layer
#		
#		A, Z, outputTarget = neuralNetworkPropagationLayerForward(l1, AprevLayer, False, networkIndex)
#	
#		LIANNtf_algorithmLIANN_math.neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf=Wf, Wfname="Wf", Wb=Wb, Wbname="Wb", updateAutoencoderBackwardsWeights=True, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)
#		
#		AprevLayer = A	#CHECKTHIS: note uses A value prior to weight updates
#		Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
#		Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A

			
def neuralNetworkPropagationAEANN(x, autoencoder, layer, networkIndex=1):

	outputPred = None
	outputTarget = None

	outputPred = x #in case layer=0
	
	AprevLayer = x
	Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	Ztrace[generateParameterNameNetwork(networkIndex, 0, "Ztrace")] = tf.zeros(AprevLayer.shape)	#set to zero as not used (just used for shape initialisation)
	
	if(autoencoder):
		maxLayer = layer
	else:
		if(layer is None):
			maxLayer = numberOfLayers
		else:
			maxLayer = layer
		
	for l1 in range(1, maxLayer+1):
		#print("l1 = ", l1)
		
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
		Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
		Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
			
	return outputPred, outputTarget

def neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder, networkIndex=1):
	outputTarget = None
	
	if(autoencoder):
		if(autoencoderPrediction=="allPreviousLayers"):
			outputTargetList = []
			for l2 in range(0, l1):
				outputTargetListPartial = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
				outputTargetList.append(outputTargetListPartial)
			outputTarget = tf.concat(outputTargetList, axis=1)
			#print("outputTarget.shape = ", outputTarget.shape)
		if(autoencoderPrediction=="previousLayer"):
			outputTarget = AprevLayer
		elif(autoencoderPrediction=="inputLayer"):
			outputTarget = Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")]
	
	if(supportSkipLayersF):
		Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
		for l2 in range(0, l1):
			WlayerF = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]
			Z = tf.add(Z, tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], WlayerF))	
		Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
	else:	
		WlayerF = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]
		Bs = B[generateParameterNameNetwork(networkIndex, l1, "B")]
		Z = tf.add(tf.matmul(AprevLayer, WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")])
	A = activationFunction(Z)
	
	return A, Z, outputTarget

def neuralNetworkPropagationLayerBackwardAutoencoder(l1, A, networkIndex=1):
	if(autoencoderPrediction=="allPreviousLayers"):
		outputPredList = []
		for l2 in range(0, l1):
			WlayerB = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]
			Zback = tf.matmul(A, WlayerB)
			outputPredPartial = tf.nn.sigmoid(Zback)
			#print("outputPredPartial.shape = ", outputPredPartial.shape)
			outputPredList.append(outputPredPartial)
		outputPred = tf.concat(outputPredList, axis=1)
		#print("outputPred.shape = ", outputPred.shape)
	elif(supportSkipLayersB):
		if(autoencoderPrediction=="inputLayer"):
			Zback = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, 0, "Ztrace")].shape)
		elif(autoencoderPrediction=="previousLayer"):
			Zback = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1-1, "Ztrace")].shape)
		for l2 in range(0, l1):
			WlayerB = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]
			if(l2 == l1-1):
				Apartial = A #allow backprop through current encoder layer only
			else:
				Apartial = Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")]
			ZbackPartial = tf.matmul(Apartial, WlayerB)
			Zback = tf.add(Zback, ZbackPartial)	
		outputPred = tf.nn.sigmoid(Zback)
	else:
		WlayerB = Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")]
		Zback = tf.matmul(A, WlayerB)
		outputPred = tf.nn.sigmoid(Zback)
		
	return outputPred
		

def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
