"""AEANNtf_algorithmSequentialInput.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
AEANN algorithm sequential input - define autoencoder generated artificial neural network for sequential input (characters, words, sentences, paragraphs etc)

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

#FUTURE; consider apply independent AEANNs as filters across subsets of sentences (ie phrases)

spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
	#python3 -m spacy download en_core_web_md
	
networkEqualInputVectorDimensions = True	#equal input vector dimensions per AEANN network - simplifies heirarchical processing

debugFastTrain = False

verticalConnectivity = True
if(verticalConnectivity):
	verticalAutoencoder = True	#specification implementation
lateralConnectivity = True
if(lateralConnectivity):
	lateralSemisupervised = True	#specification implementation
	lateralAutoencoder = True 	#specification optional (else train lateral connections using semi-supervised method exclusively based on next word prediction)
	lateralConnectivityFirstLayer = False	#not currently supported #optional (first layer has lateral connections; else just pass x input straight to first layer; layer0 becomes redundant)
	
supportFullConnectivity = True #mandatory	#full connectivity between layers (else only predict current layer based on previous layer); is required because equivalent semantic structure do not necessarily contain the same number of words per phrase (ie syntactical structure is not necessarily identical)
if(supportFullConnectivity):
	supportSkipLayers = True #mandatory


#see AEANN-AutoencoderSequentialInputArchitectureDevelopment-29December2021a.pdf;
#FUTURE: consider creating independent weights for s1 (higher layer sequential index) also
if(supportSkipLayers):
	#increase generalisation across minor syntactical structure variations
	verticalPropagationInfluentialSequenceSizeAll = True	#specification (green): do not limit sequential influence of (skip layer) vertical propagation to prior sequential input (and its recursive influencers)
	lateralPropagationInfluentialSequenceSizeAll = True	#specification (cyan): do not limit sequential influence of (skip layer) lateral propagation to prior sequential input (and its recursive influencers)
	verticalPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
	lateralPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
else:
	verticalPropagationInfluentialSequenceSizeAll = False #specification (orange): do not limit sequential influence of vertical propagation to prior sequential input	#False: mandatory - cannot be higher than 2 because last layer only has access to [the last] 2 sequential inputs?
	lateralPropagationInfluentialSequenceSizeAll = False #specification (blue): do not limit sequential influence of lateral propagation to prior sequential input	#optional?
	verticalPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)
	lateralPropagationInfluentialSequenceIndependentWeights = True	#optional (may prevent generalisation across minor syntactical structure variations)

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
	
def neuralNetworkPropagationTestGenerateHigherLayerWordVectors(networkIndex, batchX):
	if(networkEqualInputVectorDimensions):
		#return highest layer processed for each batch
		#batchSize = x.shape[0]
		numberOfSequentialInputs = x.shape[1]
		networkOutputLayerList = []
		for batchIndex in range(batchSize):
			s = getNumberValidSequentialInputs(batchX[batchIndex])
			l = getNumberLayers2(s)
			networkOutputLayer = Atrace[generateParameterNameNetworkSeq(networkIndex, l, s, "Atrace")]
			networkOutputLayerList.append(networkOutputLayer)
		higherLayerInputVectors = tf.stack(networkOutputLayerList)
	else:
		print("neuralNetworkPropagationTestGenerateHigherLayerWordVectors error: currently requires networkEqualInputVectorDimensions")
		exit()
		#return concatenateAllNeuronActivations(False, networkIndex)
	return higherLayerInputVectors
	
def getNumberValidSequentialInputs(sequenceTensor):
	sequenceTensorIsPadding = tf.equal(sequenceTensor, paddingTagIndex)
	numberValidSequentialInputs = tf.argmax(sequenceTensorIsPadding, axis=0)

def getAtraceTarget(networkIndex):
	#print("getAtraceTarget")
	return getAtraceActivations(False, networkIndex)

def initialiseAtrace(networkIndex):
	for l1 in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):
			Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			#print("initialiseAtrace Atrace.shape = ", Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")].shape)	
					
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
	
def getMinSequentialInputOfLayer(l):
	minSequentialInputOfLayer = l-1	#CHECKTHIS
	
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
		numberOfLayers = AEANNsequentialInputTypesMaxLength[networkIndex]
		
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
	
	print("numberOfNetworks = ", numberOfNetworks)
	for networkIndex in range(numberOfNetworks):
		n_h = networks_n_h[networkIndex]	#set active n_h
		
		for l1 in range(1, numberOfLayers+1):
			if(supportSkipLayers):
				if(verticalConnectivity):
					for l2 in range(1, l1):
						if(l2 < l1):	#first layer does not have vertical connections (just pass through of sequential input x):
							#vertical connections are always made from lower to higher layers:
							if(verticalPropagationInfluentialSequenceIndependentWeights):
								#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
								verticalPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)
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
							#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
							lateralPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)	#lower layers have more lateral/downward connections [exact number used/trained depends on sequential input index]	#CHECKTHIS
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
							#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
							verticalPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)
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
						#store independent weights for each s in sequence (not robust as near identical sequences may have different seq lengths)
						lateralPropagationInfluentialSequenceSizeMax = getNumberSequentialInputs(n_h) #numberOfLayers-(l1+1)	#lower layers have more lateral/downward connections [exact number used/trained depends on sequential input index]	#CHECKTHIS
						for s2 in range(lateralPropagationInfluentialSequenceSizeMax):
							Wlf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlf")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))
							if(lateralAutoencoder):
								Wlb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlb")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))					
					else:
						Wlf[generateParameterNameNetwork(networkIndex, l1, "Wlf")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))
						if(lateralAutoencoder):
							Wlb[generateParameterNameNetwork(networkIndex, l1, "Wlb")] = tf.Variable(randomNormal([n_h[l1], n_h[l1]]))

			Blayer = tf.zeros(n_h[l1])	
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(Blayer)

			numberOfSequentialInputs = getNumberSequentialInputs(n_h)
			for s in range(numberOfSequentialInputs):
				Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
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
	
	#implementation for left->right propagation only (FUTURE: bidirectional implementation required)
	x = tf.transpose(x, [1, 0, 2])		#numberInputs x batchSize x wordVecSize
	
	for s1 in range(minSequentialInput, maxSequentialInput+1, 1):
		
		xSeq = x[s1]
		#initialise input layer 0 as xSeq;
		#print("xSeq.shape = ", xSeq.shape)	
		Ztrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Ztrace")] = xSeq
		Atrace[generateParameterNameNetworkSeq(networkIndex, l=0, s=s1, arrayName="Atrace")] = xSeq
		
		if(lateralConnectivityFirstLayer):
			#not currently supported
			print("neuralNetworkPropagationAEANNsetInput error: lateralConnectivityFirstLayer not currently supported")
			exit()
		else:
			#Optional: first sequential input is not propagated past l1 s0 [asserted for clarity]
			Ztrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Ztrace")] = xSeq
			Atrace[generateParameterNameNetworkSeq(networkIndex, l=1, s=s1, arrayName="Atrace")] = xSeq

		
#x = 3D input array	#batchSize x numberInputs x wordVecSize
def neuralNetworkPropagationAEANN(x, autoencoder=False, semisupervisedEncoder=False, layerFirst=None, layerLast=None, sequentialInputFirst=None, sequentialInputLast=None, networkIndex=1):
	
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
	
	if(autoencoder):
		initialiseAtraceBackward(networkIndex)
		#neuralNetworkPropagationAEANNsetInput(x) has already been executed
	else:
		neuralNetworkPropagationAEANNsetInput(x)
	
	#implementation for left->right propagation only (FUTURE: bidirectional implementation required)
	#numberOfSequentialInputs = x.shape[1]
	#x = tf.transpose(x, [1, 0, 2])		#numberInputs x batchSize x wordVecSize
	for s1 in range(minSequentialInput, maxSequentialInput+1, 1):
		
		if(s1 != 0):
			sequentialInputMaxPropagationLayer = getSequentialInputMaxPropagationLayer(s1)

			if(layerLast is None):
				maxLayer = min(sequentialInputMaxPropagationLayer, numberOfLayers)
			else:
				maxLayer = min(sequentialInputMaxPropagationLayer, layerLast)

			output = x #in case layer=0

			for l1 in range(minLayer, maxLayer+1, 1):
					
				Z = neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, verticalConnectivity, lateralConnectivity, forward=True, autoencoder=False, networkIndex=networkIndex)
				A = activationFunction(Z)

				#update activation records:
				Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Ztrace")] = Z
				Atrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Atrace")] = A
				
				#generate output for training/prediction;
				if(s1 == maxSequentialInput):
					if(l1 == maxLayer):
						if(autoencoder):			
							#propagate backwards to immediately previously connected layer(s);
							neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, verticalAutoencoder, lateralAutoencoder, forward=False, autoencoder=True, networkIndex=networkIndex)
						elif(semisupervisedEncoder):
							print("neuralNetworkPropagationAEANN error: semisupervisedEncoder not yet coded")
							exit()
							output = tf.nn.sigmoid(Z)
						else:
							output = tf.nn.sigmoid(Z)

				A = tf.stop_gradient(A)	#only train weights for layer l1, sequentialInput s1
					#CHECKTHIS: are gradients automatically propagated through activation trace arrays?
	
	if(autoencoder):
		output = getAtraceActivations(True, networkIndex)
		
	return output

def neuralNetworkPropagationAEANNsequentialInputLayer(xSeq, s1, l1, maxSequentialInput, maxLayer, vertical, lateral, forward=True, autoencoder=False, networkIndex=0):
	
	numberOfSequentialInputs = getNumberSequentialInputs(n_h)
	Zrecord = Ztrace[generateParameterNameNetworkSeq(networkIndex, l1, s1, "Ztrace")]
	Z = tf.zeros(Zrecord.shape)	#Z initialisation

	if(forward):
		if(l1 == 1):
			#pass through x input directly to first layer;
			Zpartial = xSeq
			tf.add(Z, Zpartial)
						
	if(vertical):
		if(supportSkipLayers):
			l2min = 1
		else:
			l2min = l1-1	#ie l2 = l1-1 only
		for l2 in range(l2min, l1):
			if(l2 < l1):	#first layer does not have vertical connections (just pass through of sequential input x):
				#vertical connections are always made from lower to higher layers:
				if(verticalPropagationInfluentialSequenceSizeAll):
					s2min = getMinSequentialInputOfLayer(l2)
					print("supportSkipLayers; verticalPropagationInfluentialSequenceSizeAll: s1 = ", s1, ", l1 = ", l1, ", s2min = ", s2min)	
				else:
					s2min = s1-1	#previous sequential input								
				s2max = s1	#current sequential input
				for s2 in range(s2min, s2max+1, 1):
					Alayer2 = Atrace[generateParameterNameNetworkSeq(networkIndex, l2, s2, "Atrace")]
					if(supportSkipLayers):
						if(verticalPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wb")]	
						else:
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]										
					else:
						if(verticalPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wb")] 	
						else:
							if(forward):
								Wlayer = Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] 
							else:
								Wlayer = Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] 							
					Zpartial = tf.add(tf.matmul(Alayer2, Wlayer), B[generateParameterNameNetwork(networkIndex, l1, "B")])
					if(forward):
						tf.add(Z, Zpartial)
					else:
						Apartial = tf.nn.sigmoid(Zpartial)
						ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "ZtraceBackward")] = Zpartial
						AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "AtraceBackward")] = Apartial	
	if(lateral):
		if(supportSkipLayers):
			l2max = numberOfLayers+1
		else:
			l2max = l1	#ie l2 = l only
		for l2 in range(l1, l2max+1):
			#lateral connections are always made from higher to lower layers:
			if(lateralConnectivityFirstLayer or (l1 > 1)):	#if first layer does not have vertical connections (just pass through of sequential input x):
				if(lateralPropagationInfluentialSequenceSizeAll):
					s2min = getMinSequentialInputOfLayer(l2)
					print("supportSkipLayers; lateralPropagationInfluentialSequenceSizeAll: s1 = ", s1, ", l1 = ", l1, ", s2min = ", s2min)
				else:	
					if(s1 == numberOfSequentialInputs-1):	#or if(getSequentialInputMaxPropagationLayer(s1) == maxLayer)
						s2min = s1	#no lateral connections at final layer
					elif(s1 == 0):
						s2min = 0	#no lateral connections at first layer s0
					else:
						s2min = s1-1	#previous sequential input
				s2max = s1-1	#previous sequential input
				for s2 in range(s2min, s2max+1, 1):
					Alayer2 = Atrace[generateParameterNameNetworkSeq(networkIndex, l2, s2, "Atrace")]
					if(supportSkipLayers):
						if(lateralPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlf")] 
							else:
								Wlayer = Wlb[generateParameterNameNetworkSeqSkipLayers(networkIndex, l2, l1, s2, "Wlb")] 
						else:
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlf")] 
							else:
								Wlayer = Wlb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wlb")] 												
					else:
						if(lateralPropagationInfluentialSequenceIndependentWeights):
							if(forward):
								Wlayer = Wlf[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlf")]
							else:
								Wlayer = Wlb[generateParameterNameNetworkSeq(networkIndex, l1, s2, "Wlb")]
						else:
							if(forward):
								Wlayer = Wlf[generateParameterNameNetwork(networkIndex, l1, "Wlf")]
							else:
								Wlayer = Wlb[generateParameterNameNetwork(networkIndex, l1, "Wlb")]							
					Zpartial = tf.add(tf.matmul(Alayer2, WlayerF), B[generateParameterNameNetwork(networkIndex, l1, "B")])
					if(forward):
						tf.add(Z, Zpartial)						
					else:
						Apartial = tf.nn.sigmoid(Zpartial)
						ZtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "ZtraceBackward")] = Zpartial
						AtraceBackward[generateParameterNameNetworkSeq(networkIndex, l2, s2, "AtraceBackward")] = Apartial
			
	return Z
									

def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
