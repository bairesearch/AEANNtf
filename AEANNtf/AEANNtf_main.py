"""AEANNtf_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
	AEANNsequentialInput only:
conda install nltk
conda install spacy
python3 -m spacy download en_core_web_md

# Usage:
python3 AEANNtf_main.py

# Description:
AEANN - train an AEANN (autoencoder artificial neural network)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

from itertools import zip_longest
from ANNtf2_operations import *
import random
import ANNtf2_loadDataset


costCrossEntropyWithLogits = False
algorithmAEANN = "AEANNindependentInput"	#autoencoder generated artificial neural network
#algorithmAEANN = "AEANNsequentialInput"	#autoencoder generated artificial neural network
if(algorithmAEANN == "AEANNindependentInput"):
	import AEANNtf_algorithmIndependentInput as AEANNtf_algorithm
elif(algorithmAEANN == "AEANNsequentialInput"):
	import AEANNtf_algorithmSequentialInput as AEANNtf_algorithm

suppressGradientDoNotExistForVariablesWarnings = True

if(algorithmAEANN == "AEANNindependentInput"):
	useSmallSentenceLengths = False

	trainMultipleFiles = False
	trainMultipleNetworks = True	#optional
	if(trainMultipleNetworks):
		#numberOfNetworks = 10
		numberOfNetworks = int(100/AEANNtf_algorithm.generateLargeNetworkRatio) #normalise the number of networks based on the network layer size
		if(numberOfNetworks == 1):	#train at least 2 networks (required for tensorflow code execution consistency)
			trainMultipleNetworks = False
	else:
		numberOfNetworks = 1
elif(algorithmAEANN == "AEANNsequentialInput"):
	#performance enhancements for development environment only: 
	trainMultipleFiles = False	#can set to true for production (after testing algorithm)
	if(trainMultipleFiles):
		batchSize = 10	#max 1202	#defined by wiki database extraction size
	else:
		batchSize = 1	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup

	#should be defined as preprocessor defs (non-variable);
	AEANNsequentialInputTypeCharacters = 0
	AEANNsequentialInputTypeWords = 1
	AEANNsequentialInputTypeSentences = 2
	AEANNsequentialInputTypeParagraphs = 3	
	AEANNsequentialInputTypeArticles = 4
	AEANNsequentialInputTypes = ["characters", "words", "sentences", "paragraphs"]
	AEANNsequentialInputNumberOfTypes = len(AEANNsequentialInputTypes)

	AEANNsequentialInputType = AEANNsequentialInputTypeWords
	AEANNsequentialInputTypeName = AEANNsequentialInputTypes[AEANNsequentialInputType] #eg "words"
	AEANNsequentialInputTypeMax = AEANNsequentialInputTypeParagraphs	#0:characters, 1:words, 2:sentences, 3:paragraphs
	AEANNsequentialInputTypeMinWordVectors = True	#only train network using word vector level input (no lower abstractions)	#lookup input vectors for current level input type (e.g. if words, using a large word2vec database), else generate input vectors using lowever level AEANN network #not supported by AEANNsequentialInputType=characters
	AEANNsequentialInputTypeMaxWordVectors = True	#enable during testing	#only train network using word vector level input (no higher abstractions)	#will flatten any higher level abstractions defined in AEANNsequentialInputTypeMax down to word vector lists (sentences)
	AEANNsequentialInputTypeTrainWordVectors = (not AEANNsequentialInputTypeMinWordVectors)

	useSmallSentenceLengths = True
	if(useSmallSentenceLengths): 
		AEANNsequentialInputTypesMaxLength = [10, 10, 10, 10]	#temporarily reduce input size for debug/processing speed
	else:
		AEANNsequentialInputTypesMaxLength = [100, 100, 100, 100]	#implementation does not require this to be equal for each type (n_h[0] does not have to be identical for each network)	#inputs shorter than max length are padded

	if(AEANNtf_algorithm.networkEqualInputVectorDimensions):
		if(AEANNsequentialInputTypeMinWordVectors):
			wordVectorLibraryNumDimensions = 300	#https://spacy.io/models/en#en_core_web_md (300 dimensions)
			inputVectorNumDimensions = wordVectorLibraryNumDimensions
		else:
			asciiNumberCharacters = 128
			inputVectorNumDimensions = asciiNumberCharacters
		AEANNsequentialInputTypesVectorDimensions = [inputVectorNumDimensions, inputVectorNumDimensions, inputVectorNumDimensions, inputVectorNumDimensions]
	else:
		print("AEANNtf_main error; only AEANNtf_algorithmSequentialInput.networkEqualInputVectorDimensions is currently coded")
		exit()
		#AEANNsequentialInputTypesVectorDimensions = [asciiNumberCharacters, wordVectorNumDimensions, getNumNetworkNodes(AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeWords], getNumNetworkNodes(AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeSentences])]
	
	#maxWordLength=AEANNsequentialInputTypesMaxLength[0]	#in characters
	#maxSentenceLength=AEANNsequentialInputTypesMaxLength[1]	#in words
	#maxParagraphLength=AEANNsequentialInputTypesMaxLength[2]	#in sentences
	#maxCorpusLength=AEANNsequentialInputTypesMaxLength[3]	#in paragraphs
		
	if(AEANNsequentialInputTypeMinWordVectors):
		AEANNsequentialInputTypeMin = AEANNsequentialInputTypeWords
		trainMultipleNetworks = False
		#numberOfNetworks = AEANNsequentialInputType+1-AEANNsequentialInputTypeWords	#use word vectors from preexisting database
	else:
		trainMultipleNetworks = True
		AEANNsequentialInputTypeMin = 0
		#numberOfNetworks = AEANNsequentialInputType+1
	numberOfNetworks = AEANNsequentialInputNumberOfTypes	#full, even though not all networks may be used


if(trainMultipleFiles):
	randomiseFileIndexParse = True
	fileIndexFirst = 0
	fileIndexLast = batchSize-1
	#if(useSmallSentenceLengths):
	#	fileIndexLast = 11
	#else:
	#	fileIndexLast = 1202	#defined by wiki database extraction size
else:
	randomiseFileIndexParse = False
			
if(algorithmAEANN == "AEANNindependentInput"):
	dataset = "SmallDataset"
elif(algorithmAEANN == "AEANNsequentialInput"):
	dataset = "wikiXmlDataset"
	#if(AEANNsequentialInputTypeMinWordVectors):
	#	numberOfFeaturesPerWord = 1000	#used by wordToVec
	paddingTagIndex = 0.0	#set to 0 so that data is not propagated through network for padded (ie out of sequence range) sequential inputs	#OLD: -1	


if(dataset == "SmallDataset"):
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
	#trainMultipleFiles = False	#required
	smallDatasetDefinitionsHeader = {'index':0, 'name':1, 'fileName':2, 'classColumnFirst':3}	
	smallDatasetDefinitions = [
	(0, "New Thyroid", "new-thyroid.data", True),
	(1, "Swedish Auto Insurance", "UNAVAILABLE.txt", False),	#AutoInsurSweden.txt BAD
	(2, "Wine Quality Dataset", "winequality-whiteFormatted.csv", False),
	(3, "Pima Indians Diabetes Dataset", "pima-indians-diabetes.csv", False),
	(4, "Sonar Dataset", "sonar.all-data", False),
	(5, "Banknote Dataset", "data_banknote_authentication.txt", False),
	(6, "Iris Flowers Dataset", "iris.data", False),
	(7, "Abalone Dataset", "UNAVAILABLE", False),	#abaloneFormatted.data BAD
	(8, "Ionosphere Dataset", "ionosphere.data", False),
	(9, "Wheat Seeds Dataset", "seeds_datasetFormatted.txt", False),
	(10, "Boston House Price Dataset", "UNAVAILABLE", False)	#housingFormatted.data BAD
	]
	dataset2FileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("dataset2FileName = ", dataset2FileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
	
debugUseSmallSequentialInputDataset = False
if(debugUseSmallSequentialInputDataset):
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset4FileNameStart = "Xdataset4Part"
xmlDatasetFileNameEnd = ".xml"



def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	if(algorithmAEANN == "AEANNindependentInput"):
		return AEANNtf_algorithm.defineTrainingParameters(dataset)
	elif(algorithmAEANN == "AEANNsequentialInput"):
		print("defineTrainingParameters requires AEANNindependentInput, execute AEANNtf_algorithm.defineTrainingParameters directly")

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	if(algorithmAEANN == "AEANNindependentInput"):
		return AEANNtf_algorithm.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks)	
	elif(algorithmAEANN == "AEANNsequentialInput"):
		print("defineNetworkParameters requires AEANNindependentInput, execute AEANNtf_algorithm.defineNetworkParameters directly")

def defineNeuralNetworkParameters():
	return AEANNtf_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return AEANNtf_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	return AEANNtf_algorithm.neuralNetworkPropagation(x, networkIndex)

#if(ANNtf2_algorithm.supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex, l):
	return AEANNtf_algorithm.neuralNetworkPropagationLayer(x, networkIndex, l)
def neuralNetworkPropagationAllNetworksFinalLayer(x):
	return AEANNtf_algorithm.neuralNetworkPropagationAllNetworksFinalLayer(x)
	
def trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, autoencoder=False, s=None, l=None):
	
	#print("trainMultipleFiles error: does not support greedy training for AEANN")
	#for l in range(1, numberOfLayers+1):
	if(algorithmAEANN == "AEANNindependentInput"):
		if(AEANNtf_algorithm.learningAlgorithmAEANN):
			loss, acc = executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, autoencoder, s, l)
		elif(AEANNtf_algorithm.learningAlgorithmLIANN or AEANNtf_algorithm.learningAlgorithmNone):
			if(l==numberOfLayers):
				loss, acc = executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, autoencoder, s, l)
		else:
			print("trainBatch error: algorithmAEANN==AEANNindependentInput and AEANNtf_algorithm.learningAlgorithm unknown")
			exit()
	elif(algorithmAEANN == "AEANNsequentialInput"):
		loss, acc = executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, autoencoder, s, l)
	#loss, acc before gradient descent
	
	if((algorithmAEANN == "AEANNindependentInput") and AEANNtf_algorithm.supportDimensionalityReduction):
		executeLIANN = False
		if(AEANNtf_algorithm.supportDimensionalityReductionLimitFrequency):
			if(batchIndex % AEANNtf_algorithm.supportDimensionalityReductionLimitFrequencyStep == 0):
				executeLIANN = True
		else:
			executeLIANN = True
		if(executeLIANN):
			AEANNtf_algorithm.neuralNetworkPropagationAEANNdimensionalityReduction(batchX, networkIndex)	

	pred = None
	if(display):
		if(algorithmAEANN == "AEANNindependentInput"):
			if(AEANNtf_algorithm.learningAlgorithmAEANN):
				loss, acc = calculatePropagationLoss(batchX, batchY, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, autoencoder, s, l)	#display l autoencoder loss	
			elif(AEANNtf_algorithm.learningAlgorithmLIANN or AEANNtf_algorithm.learningAlgorithmNone): 
				loss, acc = calculatePropagationLoss(batchX, batchY, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, autoencoder, s, l=numberOfLayers)	#display final layer loss
			else:
				print("trainBatch error: algorithmAEANN==AEANNindependentInput and AEANNtf_algorithm.learningAlgorithm unknown")
				exit()
		elif(algorithmAEANN == "AEANNsequentialInput"):
			pass #not possible to reexecute calculatePropagationLoss, as it is executed iteratively across s and l	#use loss, acc before gradient optimisation
		if(l is not None):
			print("l: %i, networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (l, networkIndex, batchIndex, loss, acc))			
		else:
			print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
			
def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1, autoencoder=False, s=None, l1=None):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, autoencoder, s, l1)

	#must syncronise with defineNeuralNetworkParameters;
	if(algorithmAEANN == "AEANNindependentInput"):
		#train specific layer weights;
		Wlist = []
		Blist = []
		if(AEANNtf_algorithm.supportSkipLayers):
			for l2 in range(0, l1):
				if(l2 < l1):
					Wlist.append(AEANNtf_algorithm.Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")])
					if(l1 != numberOfLayers):
						Wlist.append(AEANNtf_algorithm.Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")])
		else:
			Wlist.append(AEANNtf_algorithm.Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")])
			if(l1 != numberOfLayers):
				Wlist.append(AEANNtf_algorithm.Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")])
		Blist.append(AEANNtf_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithmAEANN == "AEANNsequentialInput"):
		#train specific layer weights;
		Wflist = []
		Wblist = []
		Wlflist = []
		if(AEANNtf_algorithm.lateralAutoencoder):
			Wlblist = []
		Blist = []
		
		if(AEANNtf_algorithm.supportFullLayerConnectivity):
			if(AEANNtf_algorithm.verticalConnectivity):
				for l2 in range(1, l1):
					if(l2 < l1):	#first layer does not have vertical connections (just pass through of sequential input x):
						if(AEANNtf_algorithm.verticalPropagationInfluentialSequenceIndependentWeights):
							if(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsAbsolute):
								verticalPropagationInfluentialSequenceSizeMax = AEANNtf_algorithm.getNumberSequentialInputs(AEANNtf_algorithm.n_h) #numberOfLayers-(l1+1)
							elif(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsRelative):
								verticalPropagationInfluentialSequenceSizeMax = 2
							for s2 in range(verticalPropagationInfluentialSequenceSizeMax):
								Wflist.append(AEANNtf_algorithm.Wf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wf")])
								if(autoencoder and AEANNtf_algorithm.verticalAutoencoder):
									Wblist.append(AEANNtf_algorithm.Wb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wb")])
						else:
							Wflist.append(AEANNtf_algorithm.Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")])
							if(autoencoder and AEANNtf_algorithm.verticalAutoencoder):
								Wblist.append(AEANNtf_algorithm.Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")])
			if(AEANNtf_algorithm.lateralConnectivity):
				for l2 in range(l1, numberOfLayers+1):
					if(AEANNtf_algorithm.lateralPropagationInfluentialSequenceIndependentWeights):
						if(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsAbsolute):
							lateralPropagationInfluentialSequenceSizeMax = AEANNtf_algorithm.getNumberSequentialInputs(AEANNtf_algorithm.n_h) #numberOfLayers-(l1+1)
						elif(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsRelative):
							print("error: lateralPropagationInfluentialSequenceIndependentWeights and !supportSequentialConnectivityIndependentWeightsAbsolute")		
						for s2 in range(lateralPropagationInfluentialSequenceSizeMax):
							Wlflist.append(AEANNtf_algorithm.Wlf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlf")])
							if(autoencoder and AEANNtf_algorithm.lateralAutoencoder):
								Wlblist.append(AEANNtf_algorithm.Wlb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlb")])
					else:
						Wlflist.append(AEANNtf_algorithm.Wlf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlf")])
						if(autoencoder and AEANNtf_algorithm.lateralAutoencoder):
							Wlblist.append(AEANNtf_algorithm.Wlb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlb")])
		else:
			if(AEANNtf_algorithm.verticalConnectivity):
				if(l1 > 1):	#first layer does not have vertical connections (just pass through of sequential input x):
					if(AEANNtf_algorithm.verticalPropagationInfluentialSequenceIndependentWeights):
						if(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsAbsolute):
							verticalPropagationInfluentialSequenceSizeMax = AEANNtf_algorithm.getNumberSequentialInputs(AEANNtf_algorithm.n_h) #numberOfLayers-(l1+1)
						elif(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsRelative):
							verticalPropagationInfluentialSequenceSizeMax = 2
						for s2 in range(verticalPropagationInfluentialSequenceSizeMax):
							Wflist.append(AEANNtf_algorithm.Wf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wf")])
							if(autoencoder and AEANNtf_algorithm.verticalAutoencoder):
								Wblist.append(AEANNtf_algorithm.Wb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wb")])
					else:
						Wflist.append(AEANNtf_algorithm.Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")])
						if(autoencoder and AEANNtf_algorithm.verticalAutoencoder):
							Wblist.append(AEANNtf_algorithm.Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")])
			if(AEANNtf_algorithm.lateralConnectivity):
				if(AEANNtf_algorithm.lateralPropagationInfluentialSequenceIndependentWeights):
					if(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsAbsolute):
						lateralPropagationInfluentialSequenceSizeMax = AEANNtf_algorithm.getNumberSequentialInputs(AEANNtf_algorithm.n_h) #numberOfLayers-(l1+1)
					elif(AEANNtf_algorithm.supportSequentialConnectivityIndependentWeightsRelative):
						print("error: lateralPropagationInfluentialSequenceIndependentWeights and !supportSequentialConnectivityIndependentWeightsAbsolute")
					for s2 in range(lateralPropagationInfluentialSequenceSizeMax):
						Wlflist.append(AEANNtf_algorithm.Wlf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlf")])
						if(autoencoder and AEANNtf_algorithm.lateralAutoencoder):
							Wlblist.append(AEANNtf_algorithm.Wlb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlb")])
				else:
					Wlflist.append(AEANNtf_algorithm.Wlf[generateParameterNameNetwork(networkIndex, l1, "Wlf")])
					if(autoencoder and AEANNtf_algorithm.lateralAutoencoder):
						Wlblist.append(AEANNtf_algorithm.Wlb[generateParameterNameNetwork(networkIndex, l1, "Wlb")])
		Blist.append(AEANNtf_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])	
		
		trainableVariables = []
		if(AEANNtf_algorithm.verticalConnectivity):
			trainableVariables.extend(Wflist)
			if(autoencoder and AEANNtf_algorithm.verticalAutoencoder):
				trainableVariables.extend(Wblist)
		if(AEANNtf_algorithm.lateralConnectivity):
			trainableVariables.extend(Wlflist)
			if(autoencoder and AEANNtf_algorithm.lateralAutoencoder):
				trainableVariables.extend(Wlblist)						
		trainableVariables.extend(Blist)
		#trainableVariables = Wflist + Wblist + Wlflist + Wlblist + Blist

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
		
	return loss, acc
					
def calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1, autoencoder=False, s=None, l=None):
	acc = 0	#only valid for softmax class targets 

	if(algorithmAEANN == "AEANNindependentInput"):
		if(l == numberOfLayers):
			pred = AEANNtf_algorithm.neuralNetworkPropagationAEANNfinalLayer(x, networkIndex)
			target = y 
			loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
			acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
			#print("target = ", target)
			#print("pred = ", pred)
			#print("x = ", x)
			#print("y = ", y)
			#print("2 loss = ", loss)
			#print("2 acc = ", acc)
		else:
			pred, target = AEANNtf_algorithm.neuralNetworkPropagationAEANNautoencoderLayer(x, l, networkIndex)
			loss = calculateLossMeanSquaredError(pred, target)
			#print("target = ", target)
			#print("pred = ", pred)
			#print("1 loss = ", loss)
	elif(algorithmAEANN == "AEANNsequentialInput"):
		if(autoencoder):
			if(AEANNtf_algorithm.calculateLossAcrossAllActivations):
				#print("AEANNtf_algorithm.getAtraceTarget")
				target = AEANNtf_algorithm.getAtraceTarget(networkIndex)
				#print("AEANNtf_algorithm.neuralNetworkPropagationAEANN 1")
				optimisationRequired, pred, _ = AEANNtf_algorithm.neuralNetworkPropagationAEANN(x, autoencoder=True, semisupervisedEncoder=False, layerFirst=l, layerLast=l, sequentialInputFirst=s, sequentialInputLast=s, networkIndex=networkIndex)
			else:
				#print("AEANNtf_algorithm.neuralNetworkPropagationAEANN 2")
				optimisationRequired, pred, target = AEANNtf_algorithm.neuralNetworkPropagationAEANN(x, autoencoder=True, semisupervisedEncoder=False, layerFirst=l, layerLast=l, sequentialInputFirst=s, sequentialInputLast=s, networkIndex=networkIndex)
			if(optimisationRequired):
				loss = calculateLossMeanSquaredError(pred, target)
				#print("target = ", target)
				#print("pred = ", pred)
				print("loss = ", loss)
			else:
				loss = tf.Variable(0.0)
		else:
			print("calculatePropagationLoss AEANNsequentialInput error: only autoencoder learning currently coded")
			pass

	return loss, acc



#if(ANNtf2_algorithm.supportMultipleNetworks):

def testBatchAllNetworksFinalLayer(batchX, batchY, datasetNumClasses, numberOfLayers):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	
	pred = neuralNetworkPropagationAllNetworksFinalLayer(AfinalHiddenLayerTensor)
	acc = calculateAccuracy(pred, batchY)
	print("Combined network: Test Accuracy: %f" % (acc))
	
	
	
def trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display):
	
	AfinalHiddenLayerList = []
	#print("numberOfNetworks = ", numberOfNetworks)
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)	
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor.shape = ", AfinalHiddenLayerTensor.shape)
	
	executeOptimisationAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, optimizer)

	pred = None
	if(display):
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, costCrossEntropyWithLogits)
		print("Combined network: batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))
						
def executeOptimisationAllNetworksFinalLayer(x, y, datasetNumClasses, optimizer):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	Wlist = []
	Blist = []
	Wlist.append(AEANNtf_algorithm.WallNetworksFinalLayer)
	Blist.append(AEANNtf_algorithm.BallNetworksFinalLayer)
	trainableVariables = Wlist + Blist

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
			
def calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits):
	acc = 0	#only valid for softmax class targets 
	pred = neuralNetworkPropagationAllNetworksFinalLayer(x)
	#print("calculatePropagationLossAllNetworksFinalLayer: pred = ", pred)
	target = y
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 

	return loss, acc
	
	
		
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, AEANNsequentialInputTypesMaxLength, useSmallSentenceLengths, AEANNsequentialInputTypeTrainWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp

def processDatasetAEANN(AEANNsequentialInputTypeIndex, inputVectors):

	percentageDatasetTrain = ANNtf2_loadDataset.percentageDatasetTrain
	
	datasetNumExamples = inputVectors.shape[0]

	datasetNumExamplesTrain = int(float(datasetNumExamples)*percentageDatasetTrain/100.0)
	datasetNumExamplesTest = int(float(datasetNumExamples)*(100.0-percentageDatasetTrain)/100.0)
	
	train_x = inputVectors[0:datasetNumExamplesTrain, :]
	test_x = inputVectors[-datasetNumExamplesTest:, :]	
		
	return train_x, train_y, test_x, test_y
	
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
			
			print("f = ", f)
	
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			trainDataIndex = 0

			#greedy code;
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					for networkIndex in range(1, maxNetwork+1):
						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l=l)
						
					#trainMultipleNetworks code;
					if(l == maxLayer):
						if(trainMultipleNetworks):
							#train combined network final layer
							trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display)

				#trainMultipleNetworks code;
				if(trainMultipleNetworks and (l == maxLayer)):
					testBatchAllNetworksFinalLayer(testBatchX, testBatchY, datasetNumClasses, numberOfLayers)
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

def listDimensions(a):
	if not type(a) == list:
		return ""
	else:
		if len(a) == 0:
			return ""
		else:
			return (str(len(a)) + " " + listDimensions(a[0]))
								
def trainSequentialInput(trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#Model constants
	learningRate, trainingSteps, batchSize, displayStep, numEpochs = AEANNtf_algorithm.defineTrainingParameters(dataset, paddingTagIndex)
	networks_n_h = AEANNtf_algorithm.defineNetworkParameters(AEANNsequentialInputTypesMaxLength, AEANNsequentialInputTypesVectorDimensions, numberOfNetworks)	
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f

			#AEANN specific code;
							
			articles = loadDataset(fileIndex)
			
			if(AEANNsequentialInputTypeMaxWordVectors):
				#flatten any higher level abstractions defined in AEANNsequentialInputTypeMax down to word vector lists (sentences);
				articles = flattenNestedListToSentences(articles)
					
			#numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y
			
			#print("articles = ", articles)
			#print("listDimensions(articles) = ", listDimensions(articles))
			
			#generate random batches/samples from/of articles set;
			batchesList = []	#all batches
			for batchIndex in range(int(trainingSteps)):
				#FUTURE: is randomisation an appropriate batch generation algorithm (rather than guaranteeing every sample is added to a batch)?
				sampleIndexFirst = 0
				sampleIndexLast = len(articles)-1
				sampleIndexShuffledArray = generateRandomisedIndexArray(sampleIndexFirst, sampleIndexLast, arraySize=batchSize)
				paragraphs = [articles[i] for i in sampleIndexShuffledArray]
				batchesList.append(paragraphs)
	
			print("listDimensions(batchesList) = ", listDimensions(batchesList))
		
			for batchIndex, batch in enumerate(batchesList):
				#print("listDimensions(batch) = ", listDimensions(batch))
				#print("batch = ", batch)
				
				AEANNsequentialInputTypeMaxTemp = None
				batchNestedList = batch
				#print("listDimensions(batchNestedList) = ", listDimensions(batchNestedList))
				AEANNsequentialInputTypeMaxTemp = AEANNsequentialInputTypeMax
							
				if(AEANNsequentialInputTypeMaxWordVectors and AEANNsequentialInputTypeMinWordVectors):
					trainSequentialInputNetwork(batchIndex, AEANNsequentialInputTypeWords, batchNestedList, None, optimizer)
				else:	
					layerInputVectorListGenerated = trainSequentialInputNetworkRecurse(batchIndex, AEANNsequentialInputTypeMaxTemp, batchNestedList, optimizer)	#train all AEANN networks (at each layer of abstraction)
					#trainSequentialInputNetwork(batchIndex, AEANNsequentialInputTypeWords, batchNestedList, layerInputVectorListGenerated, optimizer)	#CHECKTHIS is not required

def flattenNestedListToSentences(articles):
	articlesFlattened = []
	nestedList = articles
	for AEANNsequentialInputTypeIndex in range(AEANNsequentialInputTypeArticles, AEANNsequentialInputTypeSentences, -1):
		#print("AEANNsequentialInputTypeIndex = ", AEANNsequentialInputTypeIndex)
		flattenedList = []
		for content in nestedList:
			flattenedList.extend(content)
		#print("flattenedList = ", flattenedList)
		nestedList = flattenedList	#for recursion
	AEANNsequentialInputTypeMaxTemp = AEANNsequentialInputTypeWords
	articlesFlattened = nestedList
	#print("articles = ", articles)
	#print("listDimensions(articlesFlattened) = ", listDimensions(articlesFlattened))
	return articlesFlattened
							
def trainSequentialInputNetworkRecurse(batchIndex, AEANNsequentialInputTypeIndex, batchNestedList, optimizer):

	higherLayerInputVectorList = []
	maxNumberNestedListElements = AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeIndex]
	for nestedListElementIndex in range(maxNumberNestedListElements):
		print("nestedListElementIndex = ", nestedListElementIndex)
		batchNestedListElement = []	#batched
		for nestedList in batchNestedList:
			if(nestedListElementIndex < len(nestedList)):
				batchNestedListElement.append(nestedList[nestedListElementIndex])
			else:	
				emptyList = []
				batchNestedListElement.append(emptyList)
		#print("listDimensions(batchNestedListElement) = ", listDimensions(batchNestedListElement))
		layerInputVectorListGenerated = None
		if(AEANNsequentialInputTypeIndex > AEANNsequentialInputTypeMin):
			layerInputVectorListGenerated = trainSequentialInputNetworkRecurse(batchIndex, AEANNsequentialInputTypeIndex-1, batchNestedListElement, optimizer)	#train lower levels before current level
		higherLayerInputVector = trainSequentialInputNetwork(batchIndex, AEANNsequentialInputTypeIndex, batchNestedListElement, layerInputVectorListGenerated, optimizer)
		higherLayerInputVectorList.append(higherLayerInputVector)
		
	return higherLayerInputVectorList	#shape: numberSequentialInputs x batchSize x inputVecDimensions
		
def trainSequentialInputNetwork(batchIndex, AEANNsequentialInputTypeIndex, batchNestedListElement, layerInputVectorListGenerated, optimizer):

	networkIndex = AEANNsequentialInputTypeIndex

	if(layerInputVectorListGenerated is None):
		#print("!layerInputVectorListGenerated")
		batchInputVectorList = []
		for nestedListElementIndex, nestedListElement in enumerate(batchNestedListElement):	#for every batchIndex
			#print("nestedListElementIndex = ", nestedListElementIndex)
			if(nestedListElement):	#verify list is not empty (check is required to compensate for empty parts of batches);
				#print("nestedListElement = ", nestedListElement)
				if(AEANNsequentialInputTypeIndex == AEANNsequentialInputTypeCharacters):
					#print("AEANNsequentialInputTypeIndex == AEANNsequentialInputTypeCharacters")
					textCharacterList = nestedListElement	#batched
					inputVectorList = AEANNtf_algorithm.generateCharacterVectorInputList(textCharacterList, AEANNsequentialInputTypesVectorDimensions[AEANNsequentialInputTypeIndex], AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeIndex])	#numberSequentialInputs x inputVecDimensions
				elif(AEANNsequentialInputTypeIndex == AEANNsequentialInputTypeWords):
					#print("AEANNsequentialInputTypeIndex == AEANNsequentialInputTypeWords")
					if(AEANNsequentialInputTypeMinWordVectors):		#implied true (because layerInputVectorListGenerated is None)	
						#print("AEANNsequentialInputTypeMinWordVectors")
						textWordList = nestedListElement	#batched
						inputVectorList = AEANNtf_algorithm.generateWordVectorInputList(textWordList, AEANNsequentialInputTypesVectorDimensions[AEANNsequentialInputTypeIndex], AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeIndex])	#numberSequentialInputs x inputVecDimensions
			else:
				inputVectorList = AEANNtf_algorithm.generateBlankVectorInputList(AEANNsequentialInputTypesVectorDimensions[AEANNsequentialInputTypeIndex],  AEANNsequentialInputTypesMaxLength[AEANNsequentialInputTypeIndex])	#empty sequence (seq length = 0)
			#print("inputVectorList = ", inputVectorList)
			batchInputVectorList.append(inputVectorList)
			
		#print("batchInputVectorList = ", batchInputVectorList)
		
		#samplesInputVector = np.array(list(zip_longest(*samplesInputVectorList, fillvalue=paddingTagIndex))).T	#not required as already padded by AEANNtf_algorithmSequentialInput.generateWordVectorInput/generateCharacterVectorInput
		batchInputVector = np.asarray(batchInputVectorList)	#np.array(samplesInputVectorList)
		print("batchInputVector.shape = ", batchInputVector.shape)
		batchInputVector = tf.convert_to_tensor(batchInputVector, dtype=tf.float32)
		#print("batchInputVector = ", batchInputVector)
		print("!layerInputVectorListGenerated: batchInputVector.shape = ", batchInputVector.shape)	#shape: batchSize x numberSequentialInputs x inputVecDimensions
	else:
		#print("layerInputVectorListGenerated = ", layerInputVectorListGenerated)
		batchInputVector = tf.stack(layerInputVectorListGenerated)	#shape: numberSequentialInputs x batchSize x inputVecDimensions
		print("batchInputVector.shape = ", batchInputVector.shape)
		batchInputVector = tf.transpose(batchInputVector, (1, 0, 2))	#shape: batchSize x numberSequentialInputs x inputVecDimensions
		#use existing inputVectors generated from lower layer
		#print("layerInputVectorListGenerated: batchInputVector = ", batchInputVector)
		print("layerInputVectorListGenerated: batchInputVector.shape = ", batchInputVector.shape)
	
	#initialisation;
	AEANNtf_algorithm.setNetworkIndex(networkIndex)
	numberOfSequentialInputs = AEANNtf_algorithm.getNumberSequentialInputs(AEANNtf_algorithm.n_h)
	if(AEANNtf_algorithm.calculateLossAcrossAllActivations):
		AEANNtf_algorithm.initialiseAtrace(networkIndex)	#required for autencoder loss calculations
	if(AEANNtf_algorithm.setInputIndependently):
		AEANNtf_algorithm.neuralNetworkPropagationAEANNsetInput(batchInputVector, networkIndex)

	numberOfLayers = AEANNtf_algorithm.getNumberLayers(AEANNtf_algorithm.n_h)
	#print("numberOfLayers = ", numberOfLayers)
	maxLayer = numberOfLayers	#greedy
		
	#greedy sequence input training
	for s in range(numberOfSequentialInputs):	
		print("s = ", s)
		#greedy layer training;
		for l in range(1, maxLayer+1):
			print("\tl = ", l)
			
			datasetNumClasses = None	#not used
			display = True
			
			if(AEANNtf_algorithm.verticalConnectivity):
				if(AEANNtf_algorithm.verticalAutoencoder):
					#only autoencoder learning is used during training:
					if(not AEANNtf_algorithm.setInputIndependently):
						batchX = x
					else:
						batchX = None	#already set by AEANNtf_algorithm.neuralNetworkPropagationAEANNsetInput(batchInputVector)
					batchY = None	#not used by autoencoder learning
					autoencoder = True
					trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, autoencoder, s, l)
			if(AEANNtf_algorithm.lateralConnectivity):
				if(AEANNtf_algorithm.lateralAutoencoder):
					#lateral connnectivity is already trained above
					pass
				elif(AEANNtf_algorithm.lateralSemisupervised):
					#batchYs = max(s+1, numberOfSequentialInputs-1)	#if last element of sequence, no prediction possible
					#batchY = samplesInputVectors[:, batchYs, :]	#predict next element in sequence
					print("trainSequentialInputNetwork error: lateralSemisupervised not currently supported")
					exit()

	#pred = neuralNetworkPropagationTest(test_x, networkIndex)
	#if(greedy):
	#	print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, test_y)))
	#else:
	#	print("Test Accuracy: %f" % (calculateAccuracy(pred, test_y)))

	higherLayerInputVector = AEANNtf_algorithm.neuralNetworkPropagationTestGenerateHigherLayerInputVector(networkIndex, batchInputVector)	#batchSize x inputVecDimensions		#propagate through network and return activation levels of all neurons	
	return higherLayerInputVector

def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	#print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
	
if __name__ == "__main__":
	if(algorithmAEANN == "AEANNindependentInput"):
		train(greedy=True, trainMultipleNetworks=trainMultipleNetworks)
	elif(algorithmAEANN == "AEANNsequentialInput"):
		trainSequentialInput(trainMultipleFiles=trainMultipleFiles, greedy=True)

