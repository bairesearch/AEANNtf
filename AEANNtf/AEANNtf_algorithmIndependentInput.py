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
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import copy

#supportSkipLayers = True #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template

debugFastTrain = False	#not supported
debugSmallBatchSize = False	#not supported #small batch size for debugging matrix output

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
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global learningRate
	global weightDecayRate	
	
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

	if(generateLargeNetwork):
		firstHiddenLayerNumberNeurons = num_input_neurons*3
	else:
		firstHiddenLayerNumberNeurons = num_input_neurons
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2
			
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
			
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l in range(1, numberOfLayers+1):
			#forward excitatory connections;
			WlayerF = randomNormal([n_h[l-1], n_h[l]]) 
			WlayerB = randomNormal([n_h[l], n_h[l-1]]) 
			Blayer = tf.zeros(n_h[l])
			Wf[generateParameterNameNetwork(networkIndex, l, "Wf")] = tf.Variable(WlayerF)
			Wb[generateParameterNameNetwork(networkIndex, l, "Wb")] = tf.Variable(WlayerB)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(Blayer)


def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf2)
	return neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=networkIndex)
	#return neuralNetworkPropagationAEANNtest(x, networkIndex=1)

def neuralNetworkPropagationAEANNautoencoderLayer(x, layer, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder=True, layer=layer, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=numberOfLayers, networkIndex=networkIndex)
	
def neuralNetworkPropagationAEANNtest(x, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=None, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNtestLayer(x, l, autoencoder=False, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, autoencoder, layer=l, networkIndex=networkIndex)


def neuralNetworkPropagationAEANN(x, autoencoder, layer, networkIndex=1):

	AprevLayer = x
	
	if(autoencoder):
		maxLayer = layer
	else:
		if(layer is None):
			maxLayer = numberOfLayers
		else:
			maxLayer = layer
	
	output = x #in case layer=0
	
	for l in range(1, maxLayer+1):
	
		WlayerF = Wf[generateParameterNameNetwork(networkIndex, l, "Wf")]
		Z = tf.add(tf.matmul(AprevLayer, WlayerF), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)

		if(autoencoder):
			if(l == numberOfLayers):
				output = tf.nn.softmax(Z)
			else:
				if(l == layer):				
					#go backwards
					WlayerB = Wb[generateParameterNameNetwork(networkIndex, l, "Wb")]
					Z = tf.matmul(A, WlayerB)
					output = tf.nn.sigmoid(Z)
		else:
			if(l == numberOfLayers):
				output = tf.nn.softmax(Z)
			else:
				output = A
				
		A = tf.stop_gradient(A)	#only train weights for layer l

		AprevLayer = A

	return output



def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
