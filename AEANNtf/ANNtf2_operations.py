"""ANNtf2_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf operations

"""

import tensorflow as tf
import numpy as np
import ANNtf2_globalDefs
import math

debugSingleLayerNetwork = False


#if(useBinaryWeights) or if(generateFirstLayerSDR)
	

def generateParameterName(l, arrayName):
	parameterName = "l" + str(l) + arrayName
	return parameterName
def generateParameterNameSkipLayers(lprior, l, arrayName):	#support skip layers
	parameterName = "lprior" + str(lprior) + "l" + str(l) + arrayName
	return parameterName
#support multiple networks:			
def generateParameterNameNetwork(networkIndex, l, arrayName):
	parameterName = "n" + str(networkIndex) + "l" + str(l) + arrayName
	return parameterName
def generateParameterNameNetworkSkipLayers(networkIndex, lprior, l, arrayName):	#support skip layers
	parameterName = "n" + str(networkIndex) + "lprior" + str(lprior) + "l" + str(l) + arrayName
	return parameterName

#support sequential inputs:		
#used by SANI:
def generateParameterNameSeq(l, s, arrayName):
	parameterName = "l" + str(l) + "s" + str(s) + arrayName
	return parameterName
def generateParameterNameSeqSkipLayers(lprior, l, s, arrayName):	#support skip layers
	parameterName = "lprior" + str(lprior) + "l" + str(l) + "s" + str(s) + arrayName
	return parameterName
#used by AEANN:
#support multiple networks:	
def generateParameterNameNetworkSeq(networkIndex, l, s, arrayName):
	parameterName = "n" + str(networkIndex) + "l" + str(l) + "s" + str(s) + arrayName
	return parameterName	
def generateParameterNameNetworkSeqSkipLayers(networkIndex, lprior, l, s, arrayName):
	parameterName = "n" + str(networkIndex) + "lprior" + str(lprior) + "l" + str(l) + "s" + str(s) + arrayName
	return parameterName

		
def printShape(tensor, tensorName):
	print(tensorName + ".shape = ")
	print(tensor.shape)
	
def printAverage(tensor, tensorName, indentation):
	tensorAverage = tf.reduce_mean(tf.dtypes.cast(tensor, tf.float32))
	indentationString = ""
	for i in range(indentation):
		indentationString = indentationString + "\t"
	print(indentationString + tensorName + "Average: %f" % (tensorAverage))

def calculateLossCrossEntropy(y_pred, y_true, datasetNumClasses, costCrossEntropyWithLogits=False, oneHotEncoded=False, reduceMean=True):
	if(costCrossEntropyWithLogits):
		cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(y_pred), labels=tf.cast(y_true, tf.float32))
		if(reduceMean):
			cost = tf.reduce_mean(cost)
	else:
		if(not oneHotEncoded):
			y_true = tf.one_hot(y_true, depth=datasetNumClasses)
		y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
		cost = -(y_true * tf.math.log(y_pred))
		if(reduceMean):
			cost = tf.reduce_sum(cost)
	
	return cost

def calculateLossMeanSquaredError(y_pred, y_true):
	loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
	return loss

def calculateAccuracy(y_pred, y_true):
	correct_prediction = calculateCorrectPrediction(y_pred, y_true) 
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
	
def calculateCorrectPrediction(y_pred, y_true):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return correct_prediction

def filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=-1):
	rowFilter = (train_y == classTargetFilterIndex)
	#print("rowFilter = ", rowFilter)
	train_xFiltered = train_x[rowFilter]
	train_yFiltered = train_y[rowFilter]
	return train_xFiltered, train_yFiltered

def filterNParraysByClassTargetInverse(train_x, train_y, classTargetFilterIndex=-1):
	rowFilter = (train_y != classTargetFilterIndex)
	#print("rowFilter = ", rowFilter)
	train_xFiltered = train_x[rowFilter]
	train_yFiltered = train_y[rowFilter]
	return train_xFiltered, train_yFiltered
  
def generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize):
	#shuffleSize = shuffleBufferSize
	trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_x, train_y)
	trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
	return trainData

#generate a single batch;
def generateTFbatch(test_x, test_y, batchSize):
	xShape = list(test_x.shape)
	yShape = list(test_y.shape)
	xShape[0] = batchSize
	yShape[0] = batchSize
	xShape = tuple(xShape)
	yShape = tuple(yShape)
	#print("test_x.shape = ", test_x.shape)
	#print("test_y.shape = ", test_y.shape)
	testBatchX = np.resize(test_x, xShape)
	testBatchY = np.resize(test_y, yShape)
	#print("testBatchX.shape = ", testBatchX.shape)
	#print("testBatchY.shape = ", testBatchY.shape)
	#print("testBatchX = ", testBatchX)
	#print("testBatchY = ", testBatchY)
	return testBatchX, testBatchY
	
def generateTFtrainDataUnbatchedFromNParrays(train_x, train_y):
	#print("train_x.shape = ", train_x.shape)
	#print("train_y.shape = ", train_y.shape)
	trainDataUnbatched = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	return trainDataUnbatched

def generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize):
	trainData = trainDataUnbatched.repeat().shuffle(shuffleSize).batch(batchSize).prefetch(1)	#do not repeat
	return trainData


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, generateLargeNetwork=False, generateNetworkStatic=False, generateDeepNetwork=False):
	if(debugSingleLayerNetwork):
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = defineNetworkParametersANNsingleLayer(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
	else:
		if(generateLargeNetwork):
			firstHiddenLayerNumberNeurons = num_input_neurons*3
		else:
			firstHiddenLayerNumberNeurons = num_input_neurons
		if(generateDeepNetwork):
			numberOfLayers = 6
		else:
			numberOfLayers = 2
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)
	return n_h, numberOfLayers, numberOfNetworks, datasetNumClasses

def defineNetworkParametersANNsingleLayer(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks):

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	datasetNumClasses = n_y
	n_h_0 = n_x
	n_h_3 = n_y
	n_h = [n_h_0, n_h_3]	
	numberOfLayers = len(n_h)-1
	
	print("defineNetworkParametersANNsingleLayer, n_h = ", n_h)
	
	return 	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses
	

def defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic):

	#configuration:	
	if(generateNetworkStatic):
		networkDivergenceType = "linearStatic"
	else:
		#networkDivergenceType = "nonLinearConverging"
		networkDivergenceType = "linearConverging"
		if(networkDivergenceType == "nonLinearConverging"):
			networkOptimumConvergenceAngle = 0.7	#if angle > 0.5, then more obtuse triange, if < 0.5 then more acute triangle	#fractional angle between 0 and 90 degrees
			networkDivergence = 1.0-networkOptimumConvergenceAngle 
		
	#Network parameters
	n_h = []
	datasetNumClasses = 0
		
	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	datasetNumClasses = n_y
	n_h_first = n_x
	previousNumberLayerNeurons = n_h_first
	n_h.append(n_h_first)

	for l in range(1, numberOfLayers):	#for every hidden layer
		if(networkDivergenceType == "linearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int((firstHiddenLayerNumberNeurons-num_output_neurons) * ((l-1)/(numberOfLayers-1)) + num_output_neurons)
			#print("n_h_x = ", n_h_x)
			#previousNumberLayerNeurons = n_h_x
			n_h.append(n_h_x)
		elif(networkDivergenceType == "nonLinearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int(previousNumberLayerNeurons*networkDivergence)
			n_h.append(n_h_x)
			previousNumberLayerNeurons = n_h_x
		elif(networkDivergenceType == "linearStatic"):
			n_h_x = firstHiddenLayerNumberNeurons
			n_h.append(n_h_x)
		elif(networkDivergenceType == "linearDivergingThenConverging"):
			#not yet coded
			print("defineNetworkParametersANN error: linearDivergingThenConverging not yet coded")
			exit()
		else:
			print("defineNetworkParametersANN error: unknown networkDivergenceType")
			exit()

	n_h_last = n_y
	n_h.append(n_h_last)
	
	print("defineNetworkParameters, n_h = ", n_h)
	
	return 	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses
	
def tileDimension(x, dimensionToTile, numberOfTiles, addDimension):

	#print("x = ", x)
	#print("dimensionToTile = ", dimensionToTile)
	#print("numberOfTiles = ", numberOfTiles)	
	
	if(addDimension):
		x = tf.expand_dims(x, dimensionToTile)
		
	xNumberOfDimensions = (tf.size(x.shape)).numpy()
	#print("xNumberOfDimensions = ", xNumberOfDimensions)
	multiplesDimension = [1] * xNumberOfDimensions
	multiplesDimension[dimensionToTile] = numberOfTiles
	
	multiples = tf.constant(multiplesDimension, tf.int32)
	xTiled = tf.tile(x, multiples)

	#print("xTiled = ", xTiled)
	
	return xTiled
	
def convertFloatToBool(xFloat):
	xInt = tf.dtypes.cast(xFloat, dtype=tf.dtypes.int32)
	xBool = tf.dtypes.cast(xFloat, dtype=tf.dtypes.bool)
	return xBool
	
def convertSignOutputToBool(xSignOutput):
	xSignOutput = tf.maximum(xSignOutput, 0)
	xBool = tf.dtypes.cast(xSignOutput, dtype=tf.dtypes.bool)
	return xBool
