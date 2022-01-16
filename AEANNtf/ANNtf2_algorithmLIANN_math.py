"""ANNtf2_algorithmLIANN_math.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN math (PCA simulation, correlation matrix, etc)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters

np.set_printoptions(linewidth=np.inf)

#only required by ANNtf2_algorithmLIANN_math:SVD/PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

SVDmatrixTypeExamplesByNeurons = True
SVDmatrixTypeSpikeCoincidence = False

	
#uses np not tf;
def generateSVDinputMatrix(l, n_h, AprevLayer):
	if(SVDmatrixTypeExamplesByNeurons):
		SVDinputMatrix = AprevLayer.numpy()
	elif(SVDmatrixTypeSpikeCoincidence):
		SVDinputMatrix = generateSpikeCoincidenceMatrix(l, n_h, AprevLayer)
	return SVDinputMatrix
	
def generateSpikeCoincidenceMatrix(l, n_h, AprevLayer):
	#creates co-occurence count matrix of dimensionality (num_neurons, num_neurons)
	layerSize = n_h[l-1]	#or AprevLayer.shape[1]
	batchSize = AprevLayer.shape[0]
	spikeCoincidenceMatrix = np.empty([layerSize, layerSize])
	AprevLayerNP = AprevLayer.numpy()	#perform operations in numpy to save dev time
	for k1 in range(layerSize):
		for k2 in range(layerSize):
			spikeCoincidences = np.logical_and(AprevLayerNP[:,k1], AprevLayerNP[:,k2])
			spikeCoincidenceMatrix[k1, k2] = np.sum(spikeCoincidences)
	print("spikeCoincidenceMatrix = ", spikeCoincidenceMatrix)
	return spikeCoincidenceMatrix

def calculateSVD(M, k):		
	#print("M = ", M)		
	U, Sigma, VT = randomized_svd(M, n_components=k, n_iter=5, random_state=None)	#n_iter = ?
	return U, Sigma, VT

def calculateWeights(l, n_h, SVDinputMatrix, U, Sigma, VT):
	#incomplete
	#SVDinputMatrix: batchSize*prevLayerSize
	#U: batchSize*layerSize
	SVDinputMatrixT = np.transpose(SVDinputMatrix)
	layerSize = n_h[l]
	prevLayerSize = n_h[l-1]
	AWnumpy = np.matmul(SVDinputMatrixT, U)  	#incomplete: some function of SVDinputMatrix, U, Sigma, VT	#dimensions: prevLayerSize*layerSize
	AW = tf.convert_to_tensor(AWnumpy)
	return AW

	

def learningAlgorithmStochasticCalculateMetricCorrelation(A):
	#print("A = ", A)	
	meanCorrelation = calculateCorrelationMean(A)
	print("meanCorrelation = ", meanCorrelation)
	metric = 1 - meanCorrelation
	#print("metric = ", metric)
	return metric

def calculateCorrelationMean(A):
	correlationsOffDiagonal = calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=1.0, getOffDiagonalCorrelationMatrix=False)
	meanCorrelation = calculateCorrelationMatrixOffDiagonalsMean(correlationsOffDiagonal)
	#print("meanCorrelation = ", meanCorrelation)
	
	return meanCorrelation
	
	#alternate methods;
	#normalisedStddev = tf.math.reduce_std(A, axis=1)/tf.math.reduce_mean(A, axis=1)	#batched	similarity(A) = stddev(A)/avg(A)
	#mean = tf.math.reduce_mean(A, axis=1)	#batched
	#mean = tf.expand_dims(mean, axis=1)
	#differenceFromMean(tf.subtract(A, mean))
	
def calculateCorrelationMatrixOffDiagonalsMean(correlations):
	#alternate methods:
	correlations = np.abs(correlations)	#account for negative correlations
	meanCorrelation = np.mean(correlations)

	#this method assumes bivariate normality - CHECKTHIS assumption is not violated
	#https://www.researchgate.net/post/average_of_Pearson_correlation_coefficient_values
		#https://stats.stackexchange.com/questions/109028/fishers-z-transform-in-python
	#fishersZ = np.arctanh(correlations)
	#fishersZaverage = np.mean(fishersZ)
	#inverseFishersZ = np.tanh(fishersZaverage)
	#meanCorrelation = inverseFishersZ
	return meanCorrelation
	
def learningAlgorithmStochasticCalculateMetricMaximiseAndEvenSignal(Afinal, metric1Weighting, metric2Weighting):	
	#learning objective functions:
	#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
	#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)		
	
	#print("Afinal = ", Afinal) 

	AfinalThresholded = tf.greater(Afinal, 0.0)	#threshold signal such that higher average weights are not preferenced
	AfinalThresholded = tf.dtypes.cast(AfinalThresholded, dtype=tf.dtypes.float32)	
	#print("Afinal = ", Afinal)
	#print("AfinalThresholded = ", AfinalThresholded)
	
	metric1 = tf.reduce_mean(AfinalThresholded)	#average output across batch, across layer
	
	#stdDevAcrossLayer = tf.math.reduce_std(Afinal, axis=1)	#stddev calculated across layer [1 result per batch index]
	#metric2 = tf.reduce_mean(stdDevAcrossLayer)	#average output across batch
	
	stdDevAcrossBatches = tf.math.reduce_mean(Afinal, axis=0)	 #for each dimension (k neuron in layer); calculate the mean across all batch indices
	metric2 = tf.math.reduce_std(stdDevAcrossBatches)	#then calculate the std dev across these values
	
	metric1 = metric1.numpy()
	metric2 = metric2.numpy()
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
				
	metric1 = metric1*metric1Weighting
	metric2 = metric2*metric2Weighting
	#print("metric1 = ", metric1)
	#print("metric2 = ", metric2)
	if(metric2 != 0):
		metric = metric1/metric2
	else:
		metric = 0.0
	
	return metric
	

def neuronActivationCorrelationMinimisation(networkIndex, n_h, l1, A, randomNormal, Wf, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=False, supportDimensionalityReductionRandomise=True, maxCorrelation=0.95):

	resetNeuronIfSameValueAcrossBatch = True #reset neuron if all values of a neuron k being the same value across the batch
	randomlySelectCorrelatedNeuronToReset = True	#randomly select one of each correlated neuron to reset
	useCorrelationMatrix = True	#only implementation currently available
	
	Atransposed = tf.transpose(A)
	if(useCorrelationMatrix):
		correlationMatrix = calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=0.0, getOffDiagonalCorrelationMatrix=True)	#off diagonal correlation matrix is required so that do not duplicate k1->k2 and k2->k1 correlations	#CHECKTHIS: nanReplacementValue
		#nanReplacementValue=0.0; will set the correlation as 0 if all values of a neuron k being the same value across the batch		
		#print("correlationMatrix = ", correlationMatrix)
		#print("correlationMatrix.shape = ", correlationMatrix.shape)
	
	if(useCorrelationMatrix):
		if(randomlySelectCorrelatedNeuronToReset):
			correlationMatrixRotated = np.transpose(correlationMatrix)
			k1MaxCorrelation = correlationMatrix.max(axis=0)
			k2MaxCorrelation = correlationMatrixRotated.max(axis=0)
			#print("k1MaxCorrelation = ", k1MaxCorrelation)
			#print("k2MaxCorrelation = ", k2MaxCorrelation)
			kSelect = np.random.randint(0, 2, size=k1MaxCorrelation.shape)
			mask1 = kSelect.astype(bool)
			mask2 = np.logical_not(mask1)
			mask1 = mask1.astype(float)
			mask2 = mask2.astype(float)
			k1MaxCorrelation = np.multiply(k1MaxCorrelation, mask1)
			k2MaxCorrelation = np.multiply(k2MaxCorrelation, mask2)
			kMaxCorrelation = np.add(k1MaxCorrelation, k2MaxCorrelation)
			#print("correlationMatrix = ", correlationMatrix)
			#print("correlationMatrixRotated = ", correlationMatrixRotated)
			#print("k1MaxCorrelation = ", k1MaxCorrelation)
			#print("k2MaxCorrelation = ", k2MaxCorrelation)
			#print("mask1 = ", mask1)
			#print("mask2 = ", mask2)
			#print("kMaxCorrelation = ", kMaxCorrelation)
		else:
			k1MaxCorrelation = correlationMatrix.max(axis=0)
			k2MaxCorrelation = correlationMatrix.max(axis=1)
			#k1MaxCorrelation = np.amax(correlationMatrix, axis=0)	#reduce max
			#k2MaxCorrelation = np.amax(correlationMatrix, axis=1)	#reduce max
			kMaxCorrelation = np.maximum(k1MaxCorrelation, k2MaxCorrelation)
		#kMaxCorrelationIndex = correlationMatrix.argmax(axis=0)	#or axis=1
		kMaxCorrelation = tf.convert_to_tensor(kMaxCorrelation, dtype=tf.dtypes.float32)	#make sure same type as A
		#print("kMaxCorrelation;", kMaxCorrelation)
		
		if(resetNeuronIfSameValueAcrossBatch):
			AbatchAllZero = tf.reduce_sum(A, axis=0)
			AbatchAllZero = tf.equal(AbatchAllZero, 0.0)
			AbatchAllZero = tf.cast(AbatchAllZero, tf.float32)
			kMaxCorrelation = tf.add(kMaxCorrelation, AbatchAllZero)	#set kMaxCorrelation[k]=1.0 if AbatchAllZero[k]=True
			#print("AbatchAllZero;", AbatchAllZero)

	else:
		#incomplete;
		for k1 in range(n_h[l1]):
			#calculate maximum correlation;
			k1MaxCorrelation = 0.0
			for k2 in range(n_h[l1]):
				if(k1 != k2):
					Ak1 = Atransposed[k1]	#Ak: 1d vector of batchsize
					Ak2 = Atransposed[k2]	#Ak: 1d vector of batchsize
					k1k2correlation = calculateCorrelation(Ak1, Ak2)	#undefined

	#generate masks (based on highly correlated k/neurons);
	#print("kMaxCorrelation = ", kMaxCorrelation)
	kPassArray = tf.less(kMaxCorrelation, maxCorrelation)
	kFailArray = tf.logical_not(kPassArray)
	#print("kPassArray = ", kPassArray)
	#print("kFailArray = ", kFailArray)
	kPassArrayF = tf.expand_dims(kPassArray, axis=0)
	kFailArrayF = tf.expand_dims(kFailArray, axis=0)
	kPassArrayF = tf.cast(kPassArrayF, tf.float32)
	kFailArrayF = tf.cast(kFailArrayF, tf.float32)
	if(updateAutoencoderBackwardsWeights):
		kPassArrayB = tf.expand_dims(kPassArray, axis=1)
		kFailArrayB = tf.expand_dims(kFailArray, axis=1)
		kPassArrayB = tf.cast(kPassArrayB, tf.float32)
		kFailArrayB = tf.cast(kFailArrayB, tf.float32)

	#apply masks to weights (randomise specific k/neurons);					
	if(supportSkipLayers):
		for l2 in range(0, l1):
			if(l2 < l1):
				#randomize or zero
				if(supportDimensionalityReductionRandomise):
					WlayerFrand = randomNormal([n_h[l2], n_h[l1]])
				else:
					WlayerFrand = tf.zeros([n_h[l2], n_h[l1]], dtype=tf.dtypes.float32)
				Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wfname)] = applyMaskToWeights(Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wfname)], WlayerFrand, kPassArrayF, kFailArrayF)
				if(updateAutoencoderBackwardsWeights):
					if(supportDimensionalityReductionRandomise):
						WlayerBrand = randomNormal([n_h[l1], n_h[l2]])
					else:
						WlayerBrand = tf.zeros([n_h[l1], n_h[l2]], dtype=tf.dtypes.float32)
					Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wbname)] = applyMaskToWeights(Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, Wbname)], WlayerBrand, kPassArrayB, kFailArrayB)		
	else:
		if(supportDimensionalityReductionRandomise):
			WlayerFrand = randomNormal([n_h[l1-1], n_h[l1]]) 
		else:
			WlayerFrand = tf.zeros([n_h[l1-1], n_h[l1]], dtype=tf.dtypes.float32)
		Wf[generateParameterNameNetwork(networkIndex, l1, Wfname)] = applyMaskToWeights(Wf[generateParameterNameNetwork(networkIndex, l1, Wfname)], WlayerFrand, kPassArrayF, kFailArrayF)
		if(updateAutoencoderBackwardsWeights):
			if(supportDimensionalityReductionRandomise):
				WlayerBrand = randomNormal([n_h[l1], n_h[l1-1]])
			else:
				WlayerBrand = tf.zeros([n_h[l1], n_h[l1-1]], dtype=tf.dtypes.float32)		
			Wb[generateParameterNameNetwork(networkIndex, l1, Wbname)] = applyMaskToWeights(Wb[generateParameterNameNetwork(networkIndex, l1, Wbname)], WlayerBrand, kPassArrayB, kFailArrayB)

def applyMaskToWeights(Wlayer, WlayerRand, kPassArray, kFailArray):
	WlayerRand = tf.multiply(WlayerRand, kFailArray)
	Wlayer = tf.multiply(Wlayer, kPassArray)
	Wlayer = tf.add(Wlayer, WlayerRand)
	return Wlayer


	
def calculateOffDiagonalCorrelationMatrix(A, nanReplacementValue=1.0, getOffDiagonalCorrelationMatrix=True):
	Anumpy = A.numpy()
	correlationMatrixX = np.transpose(Anumpy)	#2-D array containing multiple variables and observations. Each row of x represents a variable, and each column a single observation of all those variables. Also see rowvar below.	#https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
	correlationMatrix = np.corrcoef(correlationMatrixX)	#create correlation matrix across all k neuron dimensions (for every sample instance in the batch)

	if(getOffDiagonalCorrelationMatrix):
		diagonalOffset = 0	#exclude diagonal matrix entries
	else:
		diagonalOffset = 1	#include diagonal matrix entries - CHECKTHIS
		
	offdiagonalIndices = np.triu_indices_from(correlationMatrix, diagonalOffset)	#get off diagonal correlations	#https://stackoverflow.com/questions/14129979/mean-of-a-correlation-matrix-pandas-data-fram
	correlationMatrix[np.isnan(correlationMatrix)] = nanReplacementValue	#set all correlationMatrix nan entries as 1.0 (high correlation), to disincentivise all values of a neuron k being the same value across the batch
	correlationsOffDiagonal = correlationMatrix[offdiagonalIndices[0], offdiagonalIndices[1]]
	
	correlationMatrixOffDiagonal = np.copy(correlationMatrix)
	correlationMatrixOffDiagonal[offdiagonalIndices[0], offdiagonalIndices[1]] = 0.0
	
	if(getOffDiagonalCorrelationMatrix):
		return correlationMatrixOffDiagonal
	else:
		return correlationsOffDiagonal
	
