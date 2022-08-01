"""AEANNtf_algorithmSequentialInput.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see AEANNtf_main.py

# Usage:
see AEANNtf_main.py

# Description:
AEANNtf algorithm sequential input - define autoencoder generated artificial neural network for sequential input (characters, words, sentences, paragraphs etc)

Greedy layer construction using autoencoders

Aim: determine if can generate optimum parameters via autoencoder;
"One question that arises with auto-encoders in comparison with DBNs is whether the auto-encoders will fail to learn a useful representation when the number of units is not strictly decreasing from one layer to the next (since the networks could theoretically just learn to be the identity and perfectly minimize the reconstruction error" (Bengio et al. 2007)

Specification: AEANN-AutoencoderSequentialInputArchitectureDevelopment-29December2021a.pdf

"""

import spacy
import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import copy

spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
	#python3 -m spacy download en_core_web_md
	
networkEqualInputVectorDimensions = True	#equal input vector dimensions per AEANN network - simplifies heirarchical processing

debugFastTrain = False

setInputIndependently = True	#set input external to propagation function

verticalConnectivity = True
if(verticalConnectivity):
	verticalAutoencoder = True	#specification implementation
else:
	verticalAutoencoder = False
	
lateralConnectivity = False
if(lateralConnectivity):
	lateralSemisupervised = True	#specification implementation
	lateralAutoencoder = True 	#specification optional (else train lateral connections using semi-supervised method exclusively based on next word prediction)
	lateralConnectivityFirstLayer = False	#not currently supported #optional (first layer has lateral connections; else just pass x input straight to first layer; layer0 becomes redundant)
else:
	lateralAutoencoder = False
	
supportFullLayerConnectivity = False 	#aka supportSkipLayers	#optional	#full connectivity between layers (else only predict current layer based on previous layer); is required because equivalent semantic structure do not necessarily contain the same number of words per phrase (ie syntactical structure is not necessarily identical)
if(supportFullLayerConnectivity):
	supportFullSequentialConnectivity =  True	#optional
else:
	supportFullSequentialConnectivity = False	

calculateLossAcrossAllActivations = False
supportSequentialConnectivityIndependentWeightsAbsolute = False
if(supportFullLayerConnectivity):
	calculateLossAcrossAllActivations = True	#optional? used to generalise the loss calculation process across entire network activations
	supportSequentialConnectivityIndependentWeightsAbsolute = True	#optional
if(supportFullSequentialConnectivity):
	calculateLossAcrossAllActivations = True	#optional? used to generalise the loss calculation process across entire network activations
	supportSequentialConnectivityIndependentWeightsAbsolute = True	#optional	#absolute s independence
	supportSequentialConnectivityIndependentWeightsRelative = False
else:
	supportSequentialConnectivityIndependentWeightsAbsolute = False
	supportSequentialConnectivityIndependentWeightsRelative	= True	#optional #relative s independence	#recommended such that network can still distinguish between current and previous input
	verticalPropagationInfluentialSequenceSizeAll = False
	lateralPropagationInfluentialSequenceSizeAll = False

#see AEANN-AutoencoderSequentialInputArchitectureDevelopment-29December2021a.pdf;
#FUTURE: consider creating independent weights for s1 (higher layer sequential index) also
if(supportFullLayerConnectivity):
	#increase generalisation across minor syntactical structure variations
	if(supportFullSequentialConnectivity):
		#FUTURE; consider apply independent AEANNs as filters across subsets of sentences (ie phrases)
		verticalPropagationInfluentialSequenceSizeAll = True	#specification (green): do not limit sequential influence of (skip layer) vertical propagation to prior sequential input (and its recursive influencers)
		lateralPropagationInfluentialSequenceSizeAll = True	#specification (cyan): do not limit sequential influence of (skip layer) lateral propagation to prior sequential input (and its recursive influencers)	
	if(supportSequentialConnectivityIndependentWeightsAbsolute):
		verticalPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
		lateralPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
	if(supportSequentialConnectivityIndependentWeightsRelative):
		verticalPropagationInfluentialSequenceIndependentWeights = True	#optional
		lateralPropagationInfluentialSequenceIndependentWeights = False
else:
	if(supportFullSequentialConnectivity):
		verticalPropagationInfluentialSequenceSizeAll = False #specification (orange): do not limit sequential influence of vertical propagation to prior sequential input	#False: mandatory - cannot be higher than 2 because last layer only has access to [the last] 2 sequential inputs?
		lateralPropagationInfluentialSequenceSizeAll = False #specification (blue): do not limit sequential influence of lateral propagation to prior sequential input	#optional?
	if(supportSequentialConnectivityIndependentWeightsAbsolute):
		verticalPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
		lateralPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
	if(supportSequentialConnectivityIndependentWeightsRelative):
		verticalPropagationInfluentialSequenceIndependentWeights = True	#optional
		lateralPropagationInfluentialSequenceIndependentWeights = False


neuronsPerLayerMultiplier = 1	#CHECKTHIS

largeBatchSize = False	#requires configuration

#forward excitatory connections;
#weights;
Wf = {}	#forwards
Wb = {}	#backwards (autoencoder)
Wlf = {}	#lateral forward
if(lateralAutoencoder):
	Wlb = {}	#lateral backwards (autoencoder)
#biases;
B = {}	#bias

#support sequential inputs:
Ztrace = {}
Atrace = {}
if(calculateLossAcrossAllActivations):
	ZtraceBackward = {}
	AtraceBackward = {}

#Network parameters
networks_n_h = []	#list of n_h for every network
numberOfNetworks = 0

n_h = []	#for current active network
numberOfLayers = 0	#for current active network
numberOfSequentialInputs = 0	#for current active network

#Global parameter access;
learningRate = None	#not required (as all learning is currently performed externally by AEANNtf_main automated backprop)
batchSize = None	#not required; can be inline rederived from x/activation arrays 
paddingTagIndex = None	#required to prevent passing parameter locally
	
def neuralNetworkPropagationTestGenerateHigherLayerInputVector(networkIndex, batchX):
	if(networkEqualInputVectorDimensions):
		#return highest layer processed for each batch
		#batchSize = x.shape[0]	#already set
		#print("batchSize = ", batchSize)
		#print("batchX = ", batchX)
		numberOfSequentialInputs = batchX.shape[1]
		#print("numberOfSequentialInputs = ", numberOfSequentialInputs)
		networkOutputLayerList = []
		for batchIndex in range(batchSize):
			#print("batchX[batchIndex] = ", batchX[batchIndex])
			s = getNumberValidSequentialInputs(batchX[batchIndex])
			#print("s = ", s)
			l = getNumberLayers2(s)
			networkOutputLayer = Atrace[generateParameterNameNetworkSeq(networkIndex, l, s, "Atrace")]
			networkOutputLayer = networkOutputLayer[batchIndex]
			networkOutputLayerList.append(networkOutputLayer)
		higherLayerInputVectors = tf.stack(networkOutputLayerList)	#expected shape: batchSize x inputVecDimensions
	else:
		print("neuralNetworkPropagationTestGenerateHigherLayerWordVectors error: currently requires networkEqualInputVectorDimensions")
		exit()
		#return concatenateAllNeuronActivations(False, networkIndex)
	return higherLayerInputVectors
	
def getNumberValidSequentialInputs(sequenceTensor):
	sequenceTensorIsPadding = tf.equal(sequenceTensor, paddingTagIndex)
	numberValidSequentialInputs = tf.argmax(sequenceTensorIsPadding, axis=0)	#will create an array with identical values for every index in inputVector
	numberValidSequentialInputs = numberValidSequentialInputs[0]	#get any (eg first) value within this array
	numberValidSequentialInputs = numberValidSequentialInputs + 1	#number of non-padding elements
	numberValidSequentialInputs = numberValidSequentialInputs.numpy()
	return numberValidSequentialInputs

def getAtraceTarget(networkIndex):
	#print("getAtraceTarget")
	return getAtraceActivations(False, networkIndex)

def initialiseAtrace(networkIndex):
	for l1 in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):
			Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			#print("initialiseAtrace Atrace.shape = ", Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")].shape)	

#if(calculateLossAcrossAllActivations):				
def initialiseAtraceBackward(networkIndex):
	for l1 in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):
			ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l1, s, "ZtraceBackward")] = Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Ztrace")]
			AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l1, s, "AtraceBackward")] = Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")]


def getAtraceActivations(backward, networkIndex):
	traceList = concatenateAllNeuronActivations(backward, networkIndex)
	print("traceList = ")
	print(traceList)
	#some are numpy, some are tensorflow..
	traceStack = tf.stack(traceList)
	return traceStack

def concatenateAllNeuronActivations(backward, networkIndex):
	#TODO: verify conactenation batchSize dimension is acceptable
	traceList = []
	for l1 in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):
			if(backward):
				A = AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l1, s, "AtraceBackward")]			
			else:
				A = Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")]
			traceList.append(A)
	return traceList

def generateBlankVectorInputList(AEANNsequentialInputDimensions, AEANNsequentialInputVectorLength):	#seq length = 0
	inputVectorList = []
	for s in range(AEANNsequentialInputVectorLength):
		inputVectorShape = AEANNsequentialInputDimensions
		inputVector = np.full(inputVectorShape, paddingTagIndex, dtype=float)
		inputVectorList.append(inputVector)
	return inputVectorList
	
def generateCharacterVectorInputList(textContentList, AEANNsequentialInputDimensions, AEANNsequentialInputVectorLength):
	inputVectorList = []
	for s in range(AEANNsequentialInputVectorLength):
		if(s < len(textContentList)):
			c = textContentList[s]
			cIndex = ord(x)	#may need to convert to a tf variable
			inputVector = tf.keras.utils.to_categorical(cIndex, num_classes=AEANNsequentialInputDimensions, dtype='float32')	#onehot encode	#formally: asciiNumberCharacters
			inputVectorNP = inputVector.numpy()
			inputVectorList.append(inputVectorNP)	#3D input array	#batchSize x numberInputs x wordVecSize
		else:
			wordVectorShape = AEANNsequentialInputDimensions
			wordVector = np.full(wordVectorShape, paddingTagIndex, dtype=float)
			inputVectorList.append(wordVector)
	return inputVectorList
		
def generateWordVectorInputList(textContentList, AEANNsequentialInputDimensions, AEANNsequentialInputVectorLength):
	inputVectorList = []
	for s in range(AEANNsequentialInputVectorLength):
		if(s < len(textContentList)):
			word = textContentList[s]
			#print("word = ", word)
			doc = spacyWordVectorGenerator(word)
			wordVectorList = doc[0].vector	#verify type numpy
			wordVector = np.array(wordVectorList)
			#print("wordVector = ", wordVector)
			inputVectorList.append(wordVector)
		else:
			wordVectorShape = AEANNsequentialInputDimensions
			wordVector = np.full(wordVectorShape, paddingTagIndex, dtype=float)
			#print("wordVector = ", wordVector)
			inputVectorList.append(wordVector)
	return inputVectorList
	
def getNumNetworkNodes(sequentialInputMaxLength):
	numNetworkNodes = sequentialInputMaxLength*getNumberLayers2(sequentialInputMaxLength)*neuronsPerLayerMultiplier	#CHECKTHIS
	return numNetworkNodes

def setNetworkIndex(networkIndex):
	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	print("networkIndex = ", networkIndex)
	n_h = networks_n_h[networkIndex]
	numberOfLayers = getNumberLayers(n_h)
	numberOfSequentialInputs = getNumberSequentialInputs(n_h)
	
def getNumberLayers2(sequentialInputMaxLength):
	numberOfLayers = sequentialInputMaxLength + 1	#CHECKTHIS
	return numberOfLayers
	
def getNumberLayers(n_h):
	numberOfLayers = len(n_h) - 1	#CHECKTHIS
	return numberOfLayers
	
def getNumberSequentialInputs(n_h):
	numberOfSequentialInputs = len(n_h) - 1	#CHECKTHIS
	return numberOfSequentialInputs
	
def getSequentialInputMaxPropagationLayer(s):
	sequentialInputMaxPropagationLayer = s+1	#CHECKTHIS
	return sequentialInputMaxPropagationLayer
	
def getMinSequentialInputOfLayer(l):
	minSequentialInputOfLayer = l-1	#CHECKTHIS
	return minSequentialInputOfLayer
	
#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset, paddingTagIndexSet):
	global learningRate
	global batchSize
	global paddingTagIndex
		
	if(largeBatchSize):
		batchSize = 1000
		learningRate = 0.05
	else:
		batchSize = 50	#distinguish from sequential input size for debugging
		learningRate = 0.005

	numEpochs = 10	#100 #10
	displayStep = 100
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
	
	paddingTagIndex = paddingTagIndexSet
				
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParameters(AEANNsequentialInputTypesMaxLength, AEANNsequentialInputTypesVectorDimensions, numberOfNetworksSet):

	global networks_n_h
	global numberOfNetworks
	
	#print("numberOfNetworksSet = ", numberOfNetworksSet)
	numberOfNetworks = numberOfNetworksSet
	networks_n_h = defineNetworkParametersAEANN(AEANNsequentialInputTypesMaxLength, AEANNsequentialInputTypesVectorDimensions, numberOfNetworks)
	#print("networks_n_h = ", networks_n_h)
	
	return networks_n_h


def defineNetworkParametersAEANN(AEANNsequentialInputTypesMaxLength, AEANNsequentialInputTypesVectorDimensions, numberOfNetworks):

	#Network parameters
	networks_n_h = []
				
	for networkIndex in range(numberOfNetworks):
	
		n_h = []
		n_x = AEANNsequentialInputTypesVectorDimensions[networkIndex] #datasetNumFeatures
		n_y = AEANNsequentialInputTypesVectorDimensions[networkIndex] 	#datasetNumClasses
		numberOfLayers = getNumberLayers2(AEANNsequentialInputTypesMaxLength[networkIndex])
		
		n_h.append(n_x)
		for l in range(1, numberOfLayers):	#for every hidden layer
			n_h_current = n_x*neuronsPerLayerMultiplier	#TODO: requires calibration; currently use same number of neurons per layer (no bottleneck or fan out)
			n_h.append(n_h_current)
		n_h.append(n_y)
		#print("n_h = ", n_h)
		
		networks_n_h.append(n_h)

	return networks_n_h


#train auto encoded (vertical up) weights first, or semi-supervised (lateral down) weights first?
def defineNeuralNetworkParameters():

	global n_h
	
	randomNormal = tf.initializers.RandomNormal()
	
	print("defineNeuralNetworkParameters: numberOfNetworks = ", numberOfNetworks)
	for networkIndex in range(numberOfNetworks):
		n_h = networks_n_h[networkIndex]	#set active n_h
		numberOfLayers = getNumberLayers(n_h)
		
		for l1 in range(1, numberOfLayers+1):
			if(supportFullLayerConnectivity):
				if(verticalConnectivity):
					for l2 in range(1, l1):
						if(l2 < l1):	#first layer does not have vertical connections (just pass through of sequential input x):
							#vertical connections are always made from lower to higher layers:
							if(verticalPropagationInfluentialSequenceIndependentWeights):
								if(supportSequentialConnectivityIndependentWeightsAbsolute):
									#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
									verticalPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)
								elif(supportSequentialConnectivityIndependentWeightsRelative):
									verticalPropagationInfluentialSequenceSizeMax = 2
								for s2 in range(verticalPropagationInfluentialSequenceSizeMax):
									Wf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wf")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
									if(verticalAutoencoder):
										Wb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wb")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))					
							else:
								Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
								if(verticalAutoencoder):
									Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
				if(lateralConnectivity):
					for l2 in range(l1, numberOfLayers+1):
						#lateral connections are always made from higher to lower layers:
						if(lateralPropagationInfluentialSequenceIndependentWeights):
							if(supportSequentialConnectivityIndependentWeightsAbsolute):
								#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
								lateralPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)	#lower layers have more lateral/downward connections [exact number used/trained depends on sequential input index]	#CHECKTHIS		
							elif(supportSequentialConnectivityIndependentWeightsRelative):
								print("error: lateralPropagationInfluentialSequenceIndependentWeights and !supportSequentialConnectivityIndependentWeightsAbsolute")				
							for s2 in range(lateralPropagationInfluentialSequenceSizeMax):
								Wlf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlf")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
								if(lateralAutoencoder):
									Wlb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlb")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))	
						else:
							Wlf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlf")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
							if(lateralAutoencoder):
								Wlb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlb")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))				
			else:
				if(verticalConnectivity):
					if(l1 > 1):	#first layer does not have vertical connections (just pass through of sequential input x):
						if(verticalPropagationInfluentialSequenceIndependentWeights):
							if(supportSequentialConnectivityIndependentWeightsAbsolute):
								#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
								verticalPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)
							elif(supportSequentialConnectivityIndependentWeightsRelative):
								verticalPropagationInfluentialSequenceSizeMax = 2
							for s2 in range(verticalPropagationInfluentialSequenceSizeMax):
								Wf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wf")] = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
								if(verticalAutoencoder):
									Wb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wb")] = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))						
						else:
							Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
							if(verticalAutoencoder):
								Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
				if(lateralConnectivity):
					if(lateralPropagationInfluentialSequenceIndependentWeights):
						if(supportSequentialConnectivityIndependentWeightsAbsolute):
							#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
							lateralPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)	#lower layers have more lateral/downward connections [exact number used/trained depends on sequential input index]	#CHECKTHIS
						elif(supportSequentialConnectivityIndependentWeightsRelative):
							print("error: lateralPropagationInfluentialSequenceIndependentWeights and !supportSequentialConnectivityIndependentWeightsAbsolute")
						for s2 in range(lateralPropagationInfluentialSequenceSizeMax):
							Wlf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlf")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))
							if(lateralAutoencoder):
								Wlb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlb")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))					
					else:
						Wlf[generateParameterNameNetwork(networkIndex, l1, "Wlf")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))
						if(lateralAutoencoder):
							Wlb[generateParameterNameNetwork(networkIndex, l1, "Wlb")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))

			Blayer = tf.zeros(n_h[l1])	
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(Blayer)

			numberOfSequentialInputs = getNumberSequentialInputs(n_h)
			for s in range(numberOfSequentialInputs):
				Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				if(calculateLossAcrossAllActivations):
					ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l1, s, "ZtraceBackward")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
					AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l1, s, "AtraceBackward")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))

			

def neuralNetworkPropagation(x, networkIndex=1):	#this general function is not used (specific functions called by ANNtf2)
	#return neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=networkIndex)
	return neuralNetworkPropagationAEANN(x, networkIndex)

#def neuralNetworkPropagationAEANNautoencoderLayer(x, layer, networkIndex=1):
#	return neuralNetworkPropagationAEANN(x, autoencoder=True, layer=layer, networkIndex=networkIndex)

#def neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=1):
#	return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=numberOfLayers, networkIndex=networkIndex)
	
#def neuralNetworkPropagationAEANNtest(x, networkIndex=1):
#	return neuralNetworkPropagationAEANN(x, autoencoder=False, layer=None, networkIndex=networkIndex)

#def neuralNetworkPropagationAEANNtestLayer(x, l, autoencoder=False, networkIndex=1):
#	return neuralNetworkPropagationAEANN(x, autoencoder, layer=l, networkIndex=networkIndex)

#x: batchSize x numberInputs x wordVecSize
def neuralNetworkPropagationAEANNsetInput(x, networkIndex):

	numberOfSequentialInputs = x.shape[1]

	minSequentialInput = 0
	maxSequentialInput = numberOfSequentialInputs-1
	print("numberOfSequentialInputs = ", numberOfSequentialInputs)
	
	x = tf.transpose(x, [1, 0, 2])		#numberInputs x batchSize x wordVecSize
	
	for s1 in range(minSequentialInput, maxSequentialInput+1, 1):
		
		xSeq = x[s1]
		#initialise input layer 0 as xSeq;
		#print("xSeq.shape = ", xSeq.shape)	
		Ztrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Ztrace")] = xSeq
		Atrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Atrace")] = xSeq
		
		if(lateralConnectivity and lateralConnectivityFirstLayer):
			#not currently supported
			print("neuralNetworkPropagationAEANNsetInput error: setInputIndependently: lateralConnectivityFirstLayer not currently supported; layer 1 activations not clearly defined")
			exit()
		else:
			#Optional: first sequential input is not propagated past l1 s0 [asserted for clarity]
			Ztrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Ztrace")] = xSeq
			Atrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Atrace")] = xSeq

		
#x = 3D input array	#batchSize x numberInputs x wordVecSize
def neuralNetworkPropagationAEANN(x, autoencoder=False, semisupervisedEncoder=False, layerFirst=None, layerLast=None, sequentialInputFirst=None, sequentialInputLast=None, networkIndex=1):

	optimisationRequired = False	
	output = None
	outputTarget = None
	
	numberOfSequentialInputs = getNumberSequentialInputs(n_h)
	
	if(sequentialInputFirst is None):
		minSequentialInput = 0
	else:
		minSequentialInput = sequentialInputFirst
	if(sequentialInputLast is None):
		maxSequentialInput = numberOfSequentialInputs-1
	else:
		maxSequentialInput = sequentialInputLast		

	if(layerFirst is None):
		minLayer = 1
	else:
		minLayer = layerFirst
	
	if(autoencoder and calculateLossAcrossAllActivations):
		initialiseAtraceBackward(networkIndex)
	#assume neuralNetworkPropagationAEANNsetInput(x) has already been executed

	#implementation for left->right propagation only (FUTURE: bidirectional implementation required)

	if(not setInputIndependently):
		x = tf.transpose(x, [1, 0, 2])		#numberInputs x batchSize x wordVecSize

	#print("neuralNetworkPropagationAEANN: maxSequentialInput = ", maxSequentialInput)
	
	for s1 in range(minSequentialInput, maxSequentialInput+1, 1):
	
		if(not setInputIndependently):
			xSeq = x[s1]	#not used (set by  neuralNetworkPropagationAEANNsetInput(x)
			#initialise input layer 0 as xSeq;
			Ztrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Ztrace")] = xSeq
			Atrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Atrace")] = xSeq
		else:
			xSeq = None
			
		if(s1 != 0):
			sequentialInputMaxPropagationLayer = getSequentialInputMaxPropagationLayer(s1)
			
			if(layerLast is None):
				maxLayer = min(sequentialInputMaxPropagationLayer, numberOfLayers)
			else:
				maxLayer = min(sequentialInputMaxPropagationLayer, layerLast)

			output = x #in case layer=0

			for l1 in range(minLayer, maxLayer+1, 1):
					
				optimisationRequiredl1, Z, autoencoderOutputTarget = neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, verticalConnectivity, lateralConnectivity, forward=True, autoencoder=False, networkIndex=networkIndex)
				if(optimisationRequiredl1):
					optimisationRequired = True
				A = activationFunction(Z)

				#print("autoencoderOutputTarget set")
				
				#update activation records:
				Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Ztrace")] = Z
				Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Atrace")] = A
				
				#generate output for training/prediction;
				if(s1 == maxSequentialInput):
					if(l1 == maxLayer):
						if(autoencoder):			
							#propagate backwards to immediately previously connected layer(s);
							_, Z, autoencoderOutputPred = neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, verticalAutoencoder, lateralAutoencoder, forward=False, autoencoder=True, networkIndex=networkIndex)
						elif(semisupervisedEncoder):
							print("neuralNetworkPropagationAEANN error: semisupervisedEncoder not yet coded")
							exit()
							output = tf.nn.sigmoid(Z)
						else:
							output = tf.nn.sigmoid(Z)

				A = tf.stop_gradient(A)	#only train weights for layer l1, sequentialInput s1
					#CHECKTHIS: if(calculateLossAcrossAllActivations): are gradients automatically propagated through activation trace arrays?
	
	if(optimisationRequired):
		if(autoencoder):
			if(calculateLossAcrossAllActivations):
				output = getAtraceActivations(True, networkIndex)
			else:
				output = autoencoderOutputPred
				outputTarget = autoencoderOutputTarget
		
	return optimisationRequired, output, outputTarget

def neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, vertical, lateral, forward=True, autoencoder=False, networkIndex=0):

	optimisationRequired = False
	autoencoderOutput = None
	
	numberOfSequentialInputs = getNumberSequentialInputs(n_h)
	Zrecord = Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Ztrace")]
	Z = tf.zeros(Zrecord.shape)	#Z initialisation

	if(not calculateLossAcrossAllActivations):
		autoencoderOutputList = []

	if(forward):
		if(l1 == 1):
			if(not setInputIndependently):
				if(lateralConnectivityFirstLayer):
					#pass through x input directly to first layer;
					Zpartial = xSeq
					tf.add(Z, Zpartial)
					#not currently supported
					print("neuralNetworkPropagationAEANNsequentialInputLayer error: !setInputIndependently: lateralConnectivityFirstLayer not currently supported; layer 1 activations not clearly defined")
					exit()
				Ztrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Ztrace")] = xSeq
				Atrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Atrace")] = xSeq

	if(vertical):
		#first layer does not have vertical connections (just pass through of sequential input x):
		if(supportFullLayerConnectivity):
			l2min = 1
		else:
			l2min = max(1, l1-1)	#ie l2 = l1-1 only
		for l2 in range(l2min, l1):
			if(l2 < l1):
				#vertical connections are always made from lower to higher layers:
				if(verticalPropagationInfluentialSequenceSizeAll):
					s2min = getMinSequentialInputOfLayer(l2)
					print("supportFullLayerConnectivity; verticalPropagationInfluentialSequenceSizeAll: s1 = ", s1, ", l1 = ", l1, ", s2min = ", s2min)	
				else:
					s2min = s1-1	#previous sequential input								
				s2max = s1	#current sequential input
				for s2Index, s2 in enumerate(range(s2min, s2max+1, 1)):
					optimisationRequired = True
					Alayer2 = Atrace[generateParameterNameNetworkSeq(networkIndex, l2, s2, "Atrace")]
					#calculate s2w: s2 weights index (may be different than s2)	
					if(verticalPropagationInfluentialSequenceIndependentWeights):
						if(supportSequentialConnectivityIndependentWeightsAbsolute):
							s2w = s2
						elif(supportSequentialConnectivityIndependentWeightsRelative):
							s2w = s2Index
					else:
						s2w = None	#not used
					if(supportFullLayerConnectivity):
						if(verticalPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2w, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2w, "Wb")]	
						else:
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]										
					else:
						if(verticalPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSeq(networkIndex, l1, s2w, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSeq(networkIndex, l1, s2w, "Wb")] 	
						else:
							if(forward):
								Wlayer = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] 							
					Zpartial = tf.matmul(Alayer2, Wlayer)
					if(forward):
						Z = tf.add(Z, Zpartial)
						if(not calculateLossAcrossAllActivations):
							autoencoderOutputList.append(Alayer2)
					else:
						if(calculateLossAcrossAllActivations):
							Apartial = tf.nn.sigmoid(Zpartial)
							ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "ZtraceBackward")] = Zpartial
							AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "AtraceBackward")] = Apartial	
						else:
							Apartial = tf.nn.sigmoid(Zpartial)
							autoencoderOutputList.append(Apartial)
		Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
	if(lateral):
		if(supportFullLayerConnectivity):
			l2max = numberOfLayers+1
		else:
			l2max = l1	#ie l2 = l only
		for l2 in range(l1, l2max+1):
			#lateral connections are always made from higher to lower layers:
			if(lateralConnectivityFirstLayer or (l1 > 1)):	#if first layer does not have vertical connections (just pass through of sequential input x):
				if(lateralPropagationInfluentialSequenceSizeAll):
					s2min = getMinSequentialInputOfLayer(l2)
					print("supportFullLayerConnectivity; lateralPropagationInfluentialSequenceSizeAll: s1 = ", s1, ", l1 = ", l1, ", s2min = ", s2min)
				else:	
					if(s1 == numberOfSequentialInputs-1):	#or if(getSequentialInputMaxPropagationLayer(s1) == maxLayer)
						s2min = s1	#no lateral connections at final layer
					elif(s1 == 0):
						s2min = 0	#no lateral connections at first layer s0
					else:
						s2min = s1-1	#previous sequential input
				s2max = s1-1	#previous sequential input
				for s2Index, s2 in enumerate(range(s2min, s2max+1, 1)):
					optimisationRequired = True
					Alayer2 = Atrace[generateParameterNameNetworkSeq(networkIndex, l2, s2, "Atrace")]
					#calculate s2w: s2 weights index (may be different than s2)
					if(lateralPropagationInfluentialSequenceIndependentWeights):
						if(supportSequentialConnectivityIndependentWeightsAbsolute):
							s2w = s2
						elif(supportSequentialConnectivityIndependentWeightsRelative):
							s2w = s2Index
					else:
						s2w = None	#not used
					if(supportFullLayerConnectivity):
						if(lateralPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2w, "Wlf")] 
							else:
								Wlayer = Wlb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2w, "Wlb")] 
						else:
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlf")] 
							else:
								Wlayer = Wlb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlb")] 												
					else:
						if(lateralPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSeq(networkIndex, l1, s2w, "Wlf")]
							else:
								Wlayer = Wlb[generateParameterNameNetworkSeq(networkIndex, l1, s2w, "Wlb")]
						else:
							if(forward):
								Wlayer = Wlf[generateParameterNameNetwork(networkIndex, l1, "Wlf")]
							else:
								Wlayer = Wlb[generateParameterNameNetwork(networkIndex, l1, "Wlb")]							
					Zpartial = tf.matmul(Alayer2, WlayerF)
					if(forward):
						Z = tf.add(Z, Zpartial)	
						if(not calculateLossAcrossAllActivations):
							autoencoderOutputList.append(Alayer2)			
					else:
						if(calculateLossAcrossAllActivations):
							Apartial = tf.nn.sigmoid(Zpartial)
							ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "ZtraceBackward")] = Zpartial
							AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "AtraceBackward")] = Apartial
						else:
							Apartial = tf.nn.sigmoid(Zpartial)
							autoencoderOutputList.append(Apartial)
		Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])

	if(optimisationRequired):
		if(not calculateLossAcrossAllActivations):
			#print("autoencoderOutputList = ", autoencoderOutputList)
			autoencoderOutput = tf.concat(autoencoderOutputList, axis=0) 
			
	return optimisationRequired, Z, autoencoderOutput
									

def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
