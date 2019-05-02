#==========
# Imports
#==========

import random
import math
import numpy as np
import time
import os
import scipy.spatial.distance as sci

import matplotlib #Plotting tools.
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.image as mpimg

#==========================
# Helper Functions
#==========================

def updateDelta(function,O,k,eps,Delta,LB):
	#We want to update Delta (best individual item seen so far) and tauMin (lower bound on thresholds we keep).
	#To do this, we need to go through all the stored elements in each machine
	#and find the individual value of each of these elements.
	
	maxVal = 0.0 #Store the highest value of a single element that is currently stored in one of the machines.
	for i in O:
		
		set = []
		for element in O[i]:
			val = function([element])
			set.append([element,val])
			if val > maxVal:
				maxVal = val	

	if maxVal >= (1+eps)*Delta: #To reduce communication cost, only update Delta if the new value is at least (1+eps) times larger.
		Delta = maxVal
		
	tauMin = max(LB,Delta)/(2.0*k)/(1.0+eps) #Update tauMin.
	
	return Delta,tauMin,LB

def updateLB(function,S,O):
	#After we have finished updated all the sets S[tau] (the sets held for each threshold tau),
	#Find the S[tau] that currently has the maximum value and use it to update LB (lower bound on guess for OPT).
	#We also completely empty all the machines (i.e. O[i] = []).
	
	maxVal = -1 #maximum value amongst all saved S[tau].
	for tau in S:
		val = function(S[tau]) #Calculate the value of S[tau]
		if val > maxVal: #Update maxVal and maxS if necessary.
			maxVal = val
			maxS = S[tau] #Set with the highest value.	

	LB = maxVal #Update lower bound on guess for OPT.
	
	for i in O:
		O[i] = [] #Empty all machines.
	
	return LB,maxS

def updateThresholds(T,tauMin,delta,eps):
	#Based on the updated values of tauMin and delta,
	#update our dictionary of thresholds.
	
	
	#First, set-up all the new thresholds.
	threshold = 1.0
	while threshold > tauMin: #Get all guesses below 1
		if threshold < delta and threshold not in T:
			T[threshold] = []
		threshold = threshold/(1.0+eps)
	
	threshold = 1.0+eps #Already tried threshold=1 above.
	while threshold < delta: #Get all thresholds above 1
		if threshold > tauMin and threshold not in T:
			T[threshold] = []
		threshold = threshold*(1.0+eps)	
	
	#Next, delete all the sets with guesses are now too low.
	toDelete = [] #First collect all keys below tauMin.
	for key in T:
		if key < tauMin:
			toDelete.append(key)
	for key in toDelete: #Then delete them, one by one.
		del T[key]	
		
	return T		
	
def updateThresholds2(T,vals,tauMin,delta,eps):	
	#Based on the updated values of tauMin and delta,
	#update our dictionary of thresholds (based on our guesses for OPT).
	#Basically the same as the other version, but this is optimized for SieveStreaming++
	#because it saves the current set value (in 'vals') so we don't have to keep recomputing it every time an element arrives.
	
	
	#First, set-up all the new sets.
	threshold = 1.0
	while threshold > tauMin/(1.0+eps): #Get all guesses below 1 (i is negative in line 6 of algorithm).
		if threshold < delta and threshold not in T:
			T[threshold] = []
			vals[threshold] = 0.0
		threshold = threshold/(1.0+eps)
	
	threshold = 1.0+eps #Already tried threshold=1 above.
	while threshold < delta: #Get all thresholds above 1 (i is positive in line 6).
		if threshold > tauMin and threshold not in T:
			T[threshold] = []
			vals[threshold] = 0.0
		threshold = threshold*(1.0+eps)	
	
	#Next, delete all the sets with guesses are now too low.
	toDelete = [] #First collect all keys below tauMin.
	for key in T:
		if key < tauMin:
			toDelete.append(key)
			
	for key in toDelete: #Then delete them, one by one.
		del T[key]
		del vals[key]	
	
	return T
	
def findBestSwap(currentSet,element,function):
	#Given a 'currentSet' and incoming 'element',
	#find the best item to replace (i.e. remove something from 'currentSet' and put in 'element' instead).
	#Return the maxValue achieved by such a swapping as well as the index of the best element to swap out.
	
	#Helper function for preemptionStreaming().
	
	maxValue = 0 #maxValue achieved by swapping 'element' into 'currentSet'.
	for i in range(0,len(currentSet)):
		newSet = currentSet[:] #deep copy of currentSet
		newSet.pop(i) #Remove element at index i.
		newSet.append(element) #Append new element.
		
		val = function(newSet) #Value of set gotten by swapping 'element' into index i.
		if val > maxValue:
			maxValue = val
			maxIndex = i #Save index of best swap.
		
	bestSet = currentSet[:]
	bestSet.pop(maxIndex)
	bestSet.append(element) #Pop out the "best" index and append the current element.
	
	#maxGain = maxVal - function(currentSet) #Find maximum gain of swapping in current element.
	return maxValue,maxIndex		

def filterElements(function,O,STau,S,tau):
	#Helper function for ThresholdSampling() (Algorithm 3 in paper).
	#Basically filters out all elements in each machine below a certain threshold. 
	
	tempO = {}
	count = 0 #count the number of elements remaining after filtering
	currentVal = function(STau+S) #We don't want to keep recalculating this, so do it once outside the loop.
	for i in O:
		tempO[i] = []
		for entry in O[i]:
			marginalGain = function(STau+S+[entry]) - currentVal #Marginal gain of 'entry' to the set 'STau + S'
			if marginalGain >= tau:
				tempO[i].append(entry)
				count += 1
			 
	return tempO,count
	
def distributedSample(O,t):
	#Uniformly sample (and remove) 't' elements from the machines in 'O'.
	#The main idea is to find the total number of elements in all the machines, 'totalItems'.
	#We then sample 't' numbers uniformly at random between 0 and 'totalItems'.
	#These sampled numbers can be thought of as indices as if we put all the elements from all the machines into one big list.
	
	totalItems = 0 #Total number of elements in all the machines.
	for i in O:
		totalItems += len(O[i])
	
	if t < 1:
		return [],totalItems
	
	if t > totalItems:#Cannot sample more elements that we have.
		t = totalItems
 
	inds = random.sample(range(0,totalItems),t) #Pick t random integers between 0 and 'totalItems', this represents the indices of the elements we want to sample.
	inds = np.array(sorted(inds)) #Make sure the indices are sorted numpy array.
	
	T = [] #The actual elements we are sampling.
	j = 0 #Index we are at in inds.
	for i in O:
		while j < len(inds) and inds[j] < len(O[i]):
			T.append( O[i].pop(inds[j]) ) #Pop the appropriate element and append it to T.
			inds = inds - 1 #Adjust indices since we removed an element.
			totalItems = totalItems - 1 #Keep track of the number of items left across all the machines.
			j = j+1 #Go to the next index in inds.
			
		inds = inds-len(O[i]) #Adjust inds so we can move on to the next machine i on O.
	
	return T,totalItems	

def bufferFull(function,S,O,k,eps,Delta,LB,memory):
	#Subroutine to run when a buffer gets full.
	#This runs all the adaptive rounds and updates the S[tau]s
	
	numRounds = 0.0 #Number of adaptive rounds required to clear the buffers.
	
	Delta,tauMin,LB = updateDelta(function,O,k,eps,Delta,LB) #Update Delta (largest single element value) and tauMin (which depends on Delta).
	S = updateThresholds(S,tauMin,Delta,eps) #Add new thresholds if required and remove all thresholds < tauMin.
	numRounds += 1 #1 Adaptive round to do all the stuff above.
			
	tempO,count,index = setupDictionaries(S,O)
	
	while checkLoop(count,S,k): #While there are not full thresholds and unused elements in buffers:
		tempMemory = 0.0
		for tau in S:
			if count[tau] > 0 and len(S[tau]) < k: #If this particular threshold has elements left in buffer AND set isn't full								
				adaptiveRound(function,tau,S,tempO,count,index,k,eps)		
			tempMemory += len(S[tau])
		
		memory.append(tempMemory)	
		numRounds += 1
		LB = dropThresholds(function,S,tempO,count,index,LB,tauMin,k,eps)	

	return numRounds,LB
	
def setupDictionaries(S,O):
	#In order to mimic the parallel adaptive rounds, we need to keep some information. 

	tempO = {} #deep copy of buffers for each tau. 
	count = {} #count[tau] is the total number of items in tempO[tau].
	index = {} #index[tau] is the index of the adaptive round we are in.
	for tau in S:
		tempO[tau] = {} #Each threshold keeps its own items to filter.
		count[tau] = 0.0
		index[tau] = 0.0
		for j in O:
			tempO[tau][j] = O[j][:] #deep copy each list.
			count[tau] += len(O[j])
	
	return tempO,count,index
	
def adaptiveRound(function,tau,S,tempO,count,index,k,eps):
	#Runs a single adaptive round for the desired threshold tau.
	#We mimic parallel rounds by running these adaptiveRounds separately for all the thresholds.

	if index[tau] == 0: #If we just exited the sampling steps, we are on the filtering step.
		tempO[tau],count[tau] = filterElements(function,tempO[tau],S[tau],[],tau)
		index[tau] = index[tau] + 1							
	
	else: #These are the sampling steps.
		diff = int( (1+eps)**(index[tau]+1) - (1+eps)**index[tau] )
		currentVal = function(S[tau])
		if diff <= 1:
			X,count[tau] = distributedSample(tempO[tau],1) #Sample 1 element uniformly from the filtered elements in tempO. Count gives us the number of items remaining in the machines.
			marginalGainX = function(S[tau]+X) - currentVal
			if marginalGainX >= tau*(1-eps): #If the marginal gain of the sampled element is high enough, add it to S.
				S[tau] = S[tau] + X
				index[tau] = index[tau]+1
			else:
				index[tau] = 0
		else: #diff >= 1
			t = min(diff, k-len(S[tau])) #We want to sample diff elements, but make sure that len(S) doesn't go over k. 
			X,count[tau] = distributedSample(tempO[tau],t) #Note that the sampled elements are removed from tempO so they cannot be picked twice.
			marginalGainX = function(S[tau]+X) - currentVal #Find the marginal gain of the set X.
			S[tau] = S[tau] + X #Append the set X to S[tau] no matter what.
			index[tau] = index[tau] + 1
			if marginalGainX/len(X) <= (1-eps)*tau: #If the marginal gain is below a certain threshold
				index[tau] = 0
						
def checkLoop(count,S,k):
	#Check if we need to keep filtering the buffers
	#If at least one tau isn't full (S[tau] < k) and it has unfiltered items
	for tau in S:
		if count[tau]>0 and len(S[tau]) < k:
			return 1
			
	return 0
	
def dropThresholds(function,S,tempO,count,index,oldLB,tauMin,k,eps):
	#batchStreaming updates LB and the thresholds after each adaptive round.

	newLB = oldLB #store maximum value amongst all saved S[tau] (cannot be lower than old max).
	for tau in S:
		val = function(S[tau]) #Calculate the value of S[tau]
		if val > newLB: #Update maxVal and maxS if necessary.
			newLB = val	
		
	tauMin = max( tauMin,newLB/(2.0*k)/(1.0+eps) )  #Update tauMin if new LB gives us a better bound.
	
	drop = [] #Gather up all the thresholds we want to drop.
	for tau in S:
		if tau < tauMin:
			drop.append(tau)
	
	for tau in drop: #Drop the thresholds.
		del S[tau]
		del count[tau]
		del index[tau]
		del tempO[tau]
		
	return newLB

def clearBuffers(O):
	#Given a dictionary of buffers, clear all of them:
	
	for i in O:
		O[i] = []

def findBestSolution(function,S,O,k):
	#Once we are done with the stream, 
	#Find the best set (i.e. each threshold has a stored set, choose the best one).
	
	stored = [] #Each entry is an ( element,function([element]) ) pair
	for i in O:
		for element in O[i]:
			stored.append( [element,function([element])] )
	stored = sorted( stored, key = lambda x: x[1], reverse=True ) #Sort stored elements from highest to lowest single value.		 
	
	maxVal = -1 #maximum value amongst all saved S[tau].
	for tau in S:
			
		val = function(S[tau]) #Calculate the value of S[tau]
		if val > maxVal: #Update maxVal and maxS if necessary.
			maxVal = val
			maxS = S[tau] #Set with the highest value.	
			maxTau = tau
	
	return maxS
		
def bufferFullR(function,S,O,k,eps,Delta,LB,memory,R):
	#Subroutine to run when a buffer gets full.
	#Same as the bufferFull(), but calls adaptiveRoundR instead of adaptiveRound.
	
	numRounds = 0.0 #Number of adaptive rounds required to clear the buffers.
	numComm = 0.0
	Delta,tauMin,LB = updateDelta(function,O,k,eps,Delta,LB) #Update Delta (largest single element value) and tauMin (which depends on Delta).
	S = updateThresholds(S,tauMin,Delta,eps) #Add new thresholds if required and remove all thresholds < tauMin.
	numRounds += 1 #1 Adaptive round to do all the stuff above.
			
	tempO,count,index = setupDictionaries(S,O)
	
	while checkLoop(count,S,k): #While there are not full thresholds and unused elements in buffers:
		tempMemory = 0.0
		for tau in S:
			tempComm = adaptiveRoundR(function,tau,S,tempO,count,index,k,eps,R)		
			numComm += tempComm
			tempMemory += len(S[tau])
		
		memory.append(tempMemory)	
		numRounds += 1
		LB = dropThresholds(function,S,tempO,count,index,LB,tauMin,k,eps)	

	return numRounds,LB,numComm
	
def adaptiveRoundR(function,tau,S,tempO,count,index,k,eps,R):
	#This is the version of the adaptiveRound() function that tries to do R steps at once.
	#This is what allows us to trade off communication for adaptivity.
	
	numComm = 0
	if count[tau] > 0 and len(S[tau]) < k: #If this particular threshold has elements left in buffer AND set isn't full								
		if index[tau] == 0: #If we just exited the sampling steps, we are on the filtering step.
			tempO[tau],count[tau] = filterElements(function,tempO[tau],S[tau],[],tau)
			index[tau] = index[tau] + 1							
		
		else: #These are the sampling steps.
			
			space = k - len(S[tau]) #max number of elements we would add (i.e. if we only have space for 10 elements, we dont want to communicate 100).
			for r in range(1,R+1): #First count up the number of rounds of communication we need.
				diff = int( (1+eps)**(index[tau]+r) - (1+eps)**(index[tau]+r-1) )

				if diff <= 1:
					t = 1
				else:
					t = min(diff, space) #We want to sample diff elements, but make sure that len(S) doesn't go over k. 
					t = max(t,0) #t should not be less than 0.
				
				numComm += t
				space = space - t
				
			T = []
			currentVal = function(S[tau])
			for r in range(1,R+1):
				
				diff = int( (1+eps)**(index[tau]+1) - (1+eps)**(index[tau]) )
				
				if diff <= 1:
					t = 1
				else:
					t = min(diff, k-len(S[tau]+T)) #We want to sample diff elements, but make sure that len(S) doesn't go over k. 
					t = max(t,0) #t cannot be less than 0.
					
				X,count[tau] = distributedSample(tempO[tau],t) #Sample t elements uniformly from the filtered elements in tempO. Count gives us the number of items remaining in the machines.
				T = T+X				
				index[tau] += 1
				marginalGainT = function(S[tau]+T) - currentVal
				if marginalGainT/len(T) < tau*(1-eps): 
					index[tau] = 0
					break
			
			S[tau] = S[tau] + T #Add the first set that doesn't make it.
			
	return numComm					
	
def bufferFull1(function,S,O,k,eps,Delta,LB,memory):
	#Subroutine to run when a buffer gets full.
	#Same as bufferFull, but it calls adaptiveRound1() instead of adaptiveRound().
	
	numRounds = 0.0 #Number of adaptive rounds required to clear the buffers.
	
	Delta,tauMin,LB = updateDelta(function,O,k,eps,Delta,LB) #Update Delta (largest single element value) and tauMin (which depends on Delta).
	S = updateThresholds(S,tauMin,Delta,eps) #Add new thresholds if required and remove all thresholds < tauMin.
	numRounds += 1 #1 Adaptive round to do all the stuff above.
			
	tempO,count,index = setupDictionaries(S,O)
	
	while checkLoop(count,S,k): #While there are not full thresholds and unused elements in buffers:
		tempMemory = 0.0
		for tau in S:
			adaptiveRound1(function,tau,S,tempO,count,index,k,eps)		
			tempMemory += len(S[tau])
		
		memory.append(tempMemory)	
		numRounds += 1
		LB = dropThresholds(function,S,tempO,count,index,LB,tauMin,k,eps)	

	return numRounds,LB	
	
def adaptiveRound1(function,tau,S,tempO,count,index,k,eps):
	if count[tau] > 0 and len(S[tau]) < k: #If this particular threshold has elements left in buffer AND set isn't full								
		if index[tau] == 0: #If we just exited the sampling steps, we are on the filtering step.
			tempO[tau],count[tau] = filterElements(function,tempO[tau],S[tau],[],tau)
			index[tau] = index[tau] + 1							
		
		else: #These are the sampling steps.
			
			currentVal = function(S[tau])
			X,count[tau] = distributedSample(tempO[tau],1) #Sample 1 element uniformly from the filtered elements in tempO. Count gives us the number of items remaining in the machines.
			marginalGainX = function(S[tau]+X) - currentVal
			if marginalGainX >= tau*(1-eps): #If the marginal gain of the sampled element is high enough, add it to S.
				S[tau] = S[tau] + X
				index[tau] = index[tau]+1
			else:
				index[tau] = 0
								
#==================
# Main Algorithms
#==================

def sieveStreamingPlusPlus(stream,k,eps,function):
	#Algorithm 1 in the paper.
	
	#stream: the stream of incoming elements
	#k: number of elements we want to select from the stream.
	#eps: the parameter epsilon that determines granularity of thresholds.
	#function: the submodular function we are trying to maximize.

	delta = 0 #Max value element we have seen so far.
	LB = 0 #Lower bound on OPT (the optimal possible value for this function).
	T = {} #Dictionary of thresholds (based on possible guesses for OPT). Each T[guess] is a list S of elements selected for that threshold. 
	vals = {} #Values of set held for each threshold (so we don't have to keep recomputing it).
	memory = [] #Save amount of memory needed at each step.
	for element in stream: 
		delta = max(delta, function([element]) ) #Update max value element we have seen so far, if appropriate.
		tauMin = max(LB,delta)/(2*k) #Update tau min (lower threshold based on guess for OPT).
		
		updateThresholds2(T,vals,tauMin,delta,eps) #Drop all thresholds that are below tauMin.
		
		tempMemory = 0 #Memory used in this round
		for threshold in T:
			if len(T[threshold]) < k: #If we have not already selected k elements in the set corresponding to this threshold:
				#marginalGain = findMarginalGain(function,T[threshold],[element]) #Calculate the marginal gain of the current element in the stream, relative to this set.
				newVal = function(T[threshold]+[element])
				marginalGain = newVal - vals[threshold]
				if marginalGain > threshold:
					T[threshold].append(element) #If the marginalGain is above the required 'threshold', add the element to the set.
					vals[threshold] = newVal
					LB = max( LB,newVal ) #Update the lower bound on OPT if necessary.
				
			tempMemory += len(T[threshold])
					
		memory.append(tempMemory)	
				
	bestValue = 0 #Set default values.
	bestSet = 0 
	for threshold in T: #find the best set amongst all still-valid thresholds.
		val = function(T[threshold])
		if val > bestValue:
			bestValue = val
			bestSet = T[threshold]
		
	return bestSet,memory			
				
def sieveStreaming(stream,k,eps,function):
	#Original version of sieveStreaming.
	#Based primarily on https://las.inf.ethz.ch/files/badanidiyuru14streaming.pdf
	#Should use more memory than sieveStreamingPlusPlus().
	#Main difference from the PlusPlus version is that there is no LB (lower bound tracking).
	
	#stream: the stream of incoming elements
	#k: number of elements we want to select from the stream.
	#eps: the parameter epsilon that trades off memory for utility (determines granularity of guesses for OPT).
	#function: the submodular function we are trying to maximize.

	delta = 0 #Max value element we have seen so far.
	LB = 0 #Lower bound on OPT (the optimal possible value for this function).
	T = {} #Dictionary of thresholds (based on possible guesses for OPT). Each T[threshold] is a list S of elements selected for that threshold. 
	vals = {}
	memory = [] #Save amount of memory needed at each step.
	for element in stream: 
		delta = max(delta, function([element]) ) #Update max value element we have seen so far, if appropriate.
		tauMin = delta/(2*k) #Update tau min (lower threshold based on guess for OPT).
		
		updateThresholds2(T,vals,tauMin,delta,eps) #Drop all thresholds that are below tauMin.
		
		tempMemory = 0 #Memory used in this round
		for threshold in T:
			if len(T[threshold]) < k: #If we have not already selected k elements in the set corresponding to this threshold:
				#marginalGain = findMarginalGain(function,T[threshold],[element]) #Calculate the marginal gain of the current element in the stream, relative to this set.
				newVal = function(T[threshold]+[element])
				marginalGain = newVal - vals[threshold]
				if marginalGain > threshold:
					T[threshold].append(element) #If the marginalGain is above the required 'threshold', add the element to the set. Also update the lower bound on OPT if necessary.	
					vals[threshold] = newVal
			tempMemory += len(T[threshold])
			
		memory.append(tempMemory)
									
	bestValue = 0 #Set default values.
	bestSet = 0 
	for threshold in T: #find the best set amongst all still-valid thresholds.
		val = function(T[threshold])
		if val > bestValue:
			bestValue = val
			bestSet = T[threshold]

	return bestSet,memory		
			
def preemptionStreaming(stream,k,c,function):
	#Preemption algorithm for submodular streaming.
	#Use this as a comparison for sieveStreamingPlusPlus()
	#The basic idea is that we accept the first k elements from the stream.
	#Then, for each element in the stream, we find the optimal element to swap out from our solution.
	#If this swapping increases the value above some threshold, complete the swap.

	#stream: the stream of incoming elements	
	#k: number of elements we want to select from the stream.
	#c: parameter that determines threshold for swapping.
	#function: the submodular function we are trying to maximize.
	
	S = [] #Solution set.
	currentVal = function(S) #Find value of empty set.
	for element in stream:	
		if len(S) < k:
			S.append(element)
		else:
			maxVal,maxIndex = findBestSwap(S,element,function) #Find maximum possible gain of swapping (i.e. replacing something in 'S' with 'element'). Also return the index of best element to swap out.
		
			maxGain = maxVal - currentVal #Find marginal gain of this swap.
			if maxGain > c*currentVal/k: #If gain is above the desired threshold:
				S.pop(maxIndex) 
				S.append(element) #Swap out the ideal item (i.e. pop out the "best" index and append the current element).
				currentVal = function(S) #Update value of current set.
				
	return S

def batchStreaming(function,stream,M,k,L,thresh1,eps):
	#Algorithm 2 in the paper.
	
	#function: the submodular function we are trying to maximize.
	#stream: the overall stream of incoming elements; stream[t] is a dictionary of elements that come in at time t. In particular, stream[t]['i'] is the list of elements that arrived to machine 'i' at time 't'.
	#M: list of machines/incoming streams.
	#k: number of elements we want to select from the stream.
	#L: each machine has enough memory to store at most L elements
	#eps: the parameter epsilon that trades off memory for utility.
	
	S = {} #Initialize dictionary of thresholds kept at central machine. S[tau] is a list of elements kept for the threshold tau.
	O = {} #Initialize dictionary of lists kept at each machine.
	for i in M:
		O[i] = [] #O[i] consists of the list of elements kept at machine 'i'.
	Delta = 0.0 #Value of maximum single element.
	LB = 0.0 #Lower bound on OPT (based on current set of solutions found).
	tauMin = 0.0 #Smallest threshold to consider.

	times = [] #Make a sorted list of all the timestamps that appear in 'stream'.
	for timestamp in stream:
		times.append(timestamp)
	#times = sorted(times) #Uncomment if we want stream to be in same (sorted) order every time.
					
	memory = []
	totalRounds = 0.0
	for t in times: #For each time step 't' in each we have arriving elements.
		for i in stream[t]: #For each machine 'i' that receives an element at time 't'.
			for element in stream[t][i]:
				O[i].append(element) #Append element to corresponding machine.
				
				if len(O[i]) >= L*thresh1:
					numRounds,LB = bufferFull(function,S,O,k,eps,Delta,LB,memory) #Run the adaptive rounds and stop either when all thresholds are full or all items in buffer have been filtered.
					totalRounds += numRounds
					clearBuffers(O)	
	
	numRounds,LB = bufferFull(function,S,O,k,eps,Delta,LB,memory) #At end of stream, run the adaptive rounds one more time.		
	totalRounds += numRounds
	
	maxS = findBestSolution(function,S,O,k) #Find best solution
	return maxS,memory,totalRounds

def batchStreaming1(function,stream,M,k,L,thresh1,eps):
	#Known as Sample-One-Streaming in the paper.
	#To be honest, we changed the name of the algorithm in the paper after the code
	#was written and I'm scared to change it here in case it breaks something.
	
	#Used as a baseline in section 5.3
	
	S = {} #Initialize dictionary of thresholds kept at central machine. S[tau] is a list of elements kept for the threshold tau.
	O = {} #Initialize dictionary of lists kept at each machine.
	for i in M:
		O[i] = [] #O[i] consists of the list of elements kept at machine 'i'.
	Delta = 0.0 #Value of maximum single element.
	LB = 0.0 #Lower bound on OPT (based on current set of solutions found).
	tauMin = 0.0 #Smallest threshold to consider.

	times = [] #Make a sorted list of all the timestamps that appear in 'stream'.
	for timestamp in stream:
		times.append(timestamp)
	#times = sorted(times) #Uncomment if you want the stream to be in the same (sorted) order every time.
					
	memory = []
	totalRounds = 0.0
	for t in times: #For each time step 't' in each we have arriving elements.
		for i in stream[t]: #For each machine 'i' that receives an element at time 't'.
			for element in stream[t][i]:
				#print element
				O[i].append(element) #Append element to corresponding machine.
				
				if len(O[i]) >= L*thresh1:
					numRounds,LB = bufferFull1(function,S,O,k,eps,Delta,LB,memory) #Run the adaptive rounds and stop either when all thresholds are full or all items in buffer have been filtered.
					totalRounds += numRounds
					clearBuffers(O)	
				
	numRounds,LB = bufferFull1(function,S,O,k,eps,Delta,LB,memory) #At end of stream, run the adaptive rounds one more time.		
	totalRounds += numRounds
	
	maxS = findBestSolution(function,S,O,k) #Find best solution (this includes padding if the solution doesn't have length k).
	return maxS,memory,totalRounds

def batchStreamingR(function,stream,M,k,L,thresh1,eps,R):
	#Version of Batch-SieveStreaming++ that trades off communication for adaptivity
	#Used in section 5.4 of the paper to create the scatter plots.
	
	S = {} #Initialize dictionary of thresholds kept at central machine. S[tau] is a list of elements kept for the threshold tau.
	O = {} #Initialize dictionary of lists kept at each machine.
	for i in M:
		O[i] = [] #O[i] consists of the list of elements kept at machine 'i'.
	Delta = 0.0 #Value of maximum single element.
	LB = 0.0 #Lower bound on OPT (based on current set of solutions found).
	tauMin = 0.0 #Smallest threshold to consider.

	times = [] #Make a sorted list of all the timestamps that appear in 'stream'.
	for timestamp in stream:
		times.append(timestamp)
	#times = sorted(times) #Uncomment if you want the stream to be in the same (sorted) order every time.
					
	memory = []
	totalRounds = 0.0
	totalComm = 0.0
	for t in times: #For each time step 't' in each we have arriving elements.
		for i in stream[t]: #For each machine 'i' that receives an element at time 't'.
			for element in stream[t][i]:
				#print element
				O[i].append(element) #Append element to corresponding machine.
				
				if len(O[i]) >= L*thresh1:
					numRounds,LB,tempComm = bufferFullR(function,S,O,k,eps,Delta,LB,memory,R) #Run the adaptive rounds and stop either when all thresholds are full or all items in buffer have been filtered.
					totalRounds += numRounds
					totalComm += tempComm
					clearBuffers(O)	
				
	numRounds,LB,tempComm = bufferFullR(function,S,O,k,eps,Delta,LB,memory,R) #At end of stream, run the adaptive rounds one more time.		
	totalRounds += numRounds
	totalComm += tempComm
	
	maxS = findBestSolution(function,S,O,k) #Find best solution (this includes padding if the solution doesn't have length k).
	return maxS,memory,totalRounds,totalComm
	
def randomBaseline(stream,k,function,numTrials = 1000):
	#Randomly selects k items from the stream.
	#Returns the average value of the selected items across 'numTrials' samples.
	
	score = 0.0
	for i in range(0,numTrials):
		randomSet = random.sample(stream,k)
		score += function(randomSet)
	score = score/numTrials
	
	return score
		

#=========================
# Twitter Specific Code
#=========================

def tweetDiversity(set): 
	#Give value of a set of tweets.
	#Value is determined by making a dictionary entry for each unique word,
	#then summing up the number of retweets for each word, then taking the log.
	
	#e.g. if we had 2 tweets: "hello bob" with 64 retweets and "bye bob" with 64 retweets then we would have:
	#dic["hello"] = 64
	#dic["bob"] = 64 + 64 = 128
	#dic["bye"] = 64
	#so the value of these tweets would be log(64) + log(128) + log(64) = 6+7+6 = 19
	
	#Format of each tweet in set is:
	#[handle,cleanedWords,retweets/words,date,text]]
	
	power = 0.5 #Square root if 0.5, but can change and it's still submodular.
	dic = {}
	for tweet in set:
		
		user = tweet[0]
		words = tweet[1]
		retweets = tweet[2]
		
		for word in words:
			if word not in dic:
				dic[word] = 0
			dic[word] += retweets
	
	score = 0.0	
	for word in dic:
		#print(word,math.log(dic[word],logBase))
		if dic[word] > 0:
			#points = math.log(dic[word],logBase)
			points = dic[word] ** (power)
		else: #If we have a tweet with no retweets, a word can have value 0 in dic, which is invalid domain for log.
			points = 0
			
		score += points
	
	#global EVALS
	#EVALS += 1
	return score

def loadTwitterData(filename,startDate='2000-01-01',endDate='2020-01-01'):
	#Read data in from given file ('tweets.txt' by default).
	
	#Format of each input line is:
	##[handle,,originalPoster,,tweetText,,cleanedWords,,likes,,retweets,,timestamp,,date]
	
	#Format of each input line is:
	#[handle,cleanedWords,retweets/words,date,text]
	
	count = 0
	tweets = [] #List of tweets.
	M = {} #List of handles (i.e. unique streams we scraped from).
	with open(filename,'r') as f:
		for line in f:
			arr = line.split(',,') #Value are split up by double commas ',,'.
			temp = []
			
			handle = arr[0]
			temp.append(handle) #handle.
			
			cleanText = arr[3]
			cleanText = cleanText.split(' ')
			temp.append(cleanText) #list of (cleaned) words in this tweet.
			
			averageRetweets = int(arr[5])/float(len(cleanText))
			temp.append(averageRetweets) #Number of retweets divided by number of words
			
			timestamp = arr[7].strip()
			temp.append(timestamp) #Date/time of tweet.
			temp.append(arr[2]) #Raw text of tweet.
			
			if timestamp[:10] >= startDate and timestamp[:10] <= endDate and averageRetweets > 0:
				if handle not in M:
					M[handle] = 0
				M[handle] += 1
				count += 1
				tweets.append(temp)		
				
	return tweets,M

def createSimpleStream(tweets):
	#Given a list of tweets of format,
	#[handle,cleanedWords,retweets/words,date,text]
	
	#Output a single stream of elements (i.e. a list), sorted by arrival time (date) if desired.
	#This will be used as the input to the single source algorithms: sieveStreaming(), sieveStreamingPlusPlus(),preemptionStreaming()
	
	stream = []
	for tweet in tweets:
		timestamp = tweet[3]
		stream.append([tweet,timestamp]) #Append the entire formatted tweet, as well as the timestamp, so we can sort easily.
	
	#stream = sorted(stream, key = lambda x: x[1]) #Sort by increasing order of timestamp if desired.
	
	cleanStream = []
	for tweet in stream:
		cleanStream.append(tweet[0]) #Append only the formatted tweet (not the timestamp).
	
	return cleanStream 
		
def createHybridStream(tweets):
	#Given a list of tweets of format,
	#[handle,cleanedWords,retweets/words,date,text]
	
	#Output a stream such that stream[t][i] is a list of tweets arriving to machine i at timestamp t.
	#Used as input for the multi-source batchStreaming-based  algorithms
	
	stream = {}
	for tweet in tweets:
		handle = tweet[0] #This determines the machine each stream is arriving to (i.e. we are assuming each machine scrapes one twitter page).
		timestamp = tweet[3] #This determines the time each tweet arrives (Might also make sense to sort these in reverse order because a scraper would probably scrape the newest tweets first).
		
		if timestamp not in stream:
			stream[timestamp] = {}
		if handle not in stream[timestamp]:
			stream[timestamp][handle] = []
			
		stream[timestamp][handle].append(tweet)	
	
	return stream

#=========================
# YouTube Specific Code
#=========================

def logdet(S,alpha=10.0):
	#Calculate the logdet of a set of vectors S.
	#That is, take the log determinant of the similarity matrix M created by S.
	#M[i][j] is the similarity between vectors S[i] and S[j]/
	
	n = len(S) #Number of vectors in S.	
	if n == 0:
		return 0 #Value of empty set is 0.
		
	M = sci.squareform(sci.pdist(S))
	M = np.exp(-1*M)
	
	M = np.identity(n) + alpha*M #Need to add identity to make the function monotone submodular.
	
	#Take log of determinant of M
	return math.log( np.linalg.det( M ) ) 		

def importVectors(filename):
	#Input .txt file containing list of test vectors.
	#That is, each line in the file is the encoded vector representing a frame from our videos.
	
	#Output will be list where each entry is of format [frameName,frameNumber,[vector]]
	
	vectors = [] #The encoded and normalized vectors representing each frame.
	names = [] #The filename of each corresponding frame.
	dic = {} #Dictionary to keep track of number of frames from each city.
	with open(filename,'r') as f:
		for line in f:
			data = line.split(',')
			
			frameName = data[0] #First entry in each line is the name of the file containing the actual frame.
			names.append(frameName) 
			
			city = frameName.split('-')[0] #frame filenames are of format: 'city-frameNumber'
			if city not in dic:
				dic[city] = 0
			dic[city] += 1 #Update number of frames we've seen from this city.
			
			temp = [] #The rest of the entries are encoded vector.
			for i in range(1,len(data)):
				temp.append( float(data[i].strip()) ) #Each entry is a string and we want a float.
			temp = np.array(temp) #Make it a numpy array.
			
			#temp = temp/np.linalg.norm(temp) #Normalize the vector (i.e. all should be unit vectors to be consistent).
			
			vectors.append([frameName,dic[city],temp])
			
	return vectors,dic		

def simpleStreamYoutube(vectors):
	#Given a list where each entry is of format:
	#[frameName,frameNumber,[vector]]
	
	#Return a single stream of vectors, sorted by timestamp (i.e. frameNumber) if desired.
	
	stream = []
	for entry in vectors:
		frameName = entry[0]
		timestamp = entry[1]
		vector = entry[2]	
		stream.append([vector,timestamp])
	
	#stream = sorted(stream, key = lambda x: x[1]) #Sort by increasing order of timestamp.
	
	sortedStream = []
	for vector in stream:
		sortedStream.append(vector[0]) #Append only the encoded vector (not the timestamp).
	
	#Return sorted (by increasing timestamp) list of vectors.
	return sortedStream

def hybridStreamYoutube(vectors):
	#Given a list where each entry is of format:
	#[frameName,frameNumber,[vector]]
	
	#Return a stream such that stream[t][i] is a list of vectors arriving to machine i at timestamp t.
	
	#Used as input to the multi-source batchStreaming() based algorithms.
	
	stream = {}
	for entry in vectors:
		vector = entry[2] #This determines the machine each stream is arriving to (i.e. we are assuming each machine scrapes one video).
		timestamp = entry[1] #This determines the time each vector arrives 
		name = entry[0].split('-')[0] #Video it comes from.
		
		if timestamp not in stream:
			stream[timestamp] = {}
		if name not in stream[timestamp]:
			stream[timestamp][name] = []
		
		stream[timestamp][name].append(vector)	
	
	return stream


#===================
# Code for graphing	
#===================


def singleSourceK(arr,eps,function,loadData,loadStream,sourceFile,numTrials,testPerc,prefix):
	#Compare sieve-streaming vs sieve-streaming++ vs preemption-streaming
	#Keep eps constant and vary k, taking values from 'arr'.
	
	#Compares utility and memory.
	
	sieveUtilityArr = np.zeros(len(arr)) #Arrays to hold results
	sieveMemoryArr = np.zeros(len(arr))
	sievePPUtilityArr = np.zeros(len(arr))
	sievePPMemoryArr = np.zeros(len(arr))
	preemptionUtilityArr = np.zeros(len(arr))
	preemptionMemoryArr = np.zeros(len(arr))
	
	data,M = loadData(sourceFile) #Load in data from desired file
	sampleSize = int(len(data)*testPerc) #Decide what fraction of the data to to test one.
	print sampleSize
	
	subsample = random.sample(data,sampleSize) #Subsample the data.
	random.shuffle(subsample) #Randomize the order.
	simpleStream = loadStream(subsample) #Turn it into stream format.
	
	for j in range(0,len(arr)):
		k = arr[j]	
		start = time.time()
		
	
	for j in range(0,numTrials):
		
		subsample = random.sample(data,sampleSize)
		random.shuffle(subsample)
		simpleStream = loadStream(subsample)
		
		for i in range(0,len(arr)):
			k = arr[i]
			print k
			
			start = time.time()
			preemptionSolution = preemptionStreaming(simpleStream,k,1,function) #c=1
			preemptionValue = function(preemptionSolution)
			#print 'preemption = ',preemptionValue,time.time()-start
			preemptionUtilityArr[i] += preemptionValue /float(numTrials)
			preemptionMemoryArr[i] += k /float(numTrials)
			
			start = time.time()
			sieveSolution,sieveMemory = sieveStreaming(simpleStream,k,eps,function)
			sieveUtilityArr[i] += function(sieveSolution) /float(numTrials)
			sieveMemoryArr[i] += max(sieveMemory) /float(numTrials)
			#print time.time()-start #Print the runtime of each algorithm, if desired.

			start = time.time()
			sievePPSolution,sievePPMemory = sieveStreamingPlusPlus(simpleStream,k,eps,function)
			sievePPUtilityArr[i] += function(sievePPSolution) /float(numTrials)
			sievePPMemoryArr[i] += max(sievePPMemory) /float(numTrials)
			#print time.time()-start
	
	plt.xlabel('Elements Selected (k)',size = '25')
	plt.ylabel('Utility',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	blue_line = mlines.Line2D([], [], color='r', marker='o', linestyle='-',markersize=5, label='Sieve-Streaming')
	magenta_line = mlines.Line2D([], [], color='m', marker='D', linestyle='-',markersize=5, label='Preemption-Streaming')
	plt.legend(fontsize='x-large',handles=[green_line,blue_line,magenta_line])
	plt.plot(arr, sievePPUtilityArr, '-gs',linewidth=5.0,markersize=8.0)
	plt.plot(arr, sieveUtilityArr, '--ro',linewidth=2.0,markersize=5.0)
	plt.plot(arr, preemptionUtilityArr, '-mD',linewidth=2.0,markersize=5.0)
	savename = prefix+'-single-utility-e'+str(int(eps*100))+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()

	plt.xlabel('Elements Selected (k)',size = '25')
	plt.ylabel('Memory',size='25')
	plt.xticks(kArr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	blue_line = mlines.Line2D([], [], color='r', marker='o', linestyle='-',markersize=5, label='Sieve-Streaming')
	magenta_line = mlines.Line2D([], [], color='m', marker='s', linestyle='-',markersize=5, label='Preemption-Streaming')
	plt.legend(fontsize='x-large',handles=[green_line,blue_line,magenta_line])
	plt.plot(arr, sievePPMemoryArr, '-gs',linewidth=2.0,markersize=5.0)
	plt.plot(arr, sieveMemoryArr, '-ro',linewidth=2.0,markersize=5.0)
	plt.plot(arr, preemptionMemoryArr, '-mD',linewidth=2.0,markersize=5.0)
	plt.ylim(0,plt.ylim()[1])
	plt.xlim(plt.xlim())
	savename = prefix+'-sieve-memory-e'+str(int(eps*100))+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()	
	
# 	print savename #print results if desired.
# 	print 'sieveUtilityArr =',sieveUtilityArr
# 	print 'sieveMemoryArr =',sieveMemoryArr
# 	print 'sievePPUtilityArr =',sievePPUtilityArr
# 	print 'sievePPMemoryArr =',sievePPMemoryArr
# 	print 'preemptionUtilityArr =',preemptionUtilityArr
# 	print 'preemptionMemoryArr =',preemptionMemoryArr

	return 0

def singleSourceEps(arr,k,function,loadData,loadStream,sourceFile,numTrials,testPerc,prefix):
	#Compare sieve-streaming vs sieve-streaming++ vs preemption-streaming
	#Keep k constant and vary eps, taking values from 'arr'.
	
	#Compares utility and memory.
	
	#Basically the same as sieveK so see above for more detailed comments.
	
	sieveUtilityArr = np.zeros(len(arr))
	sieveMemoryArr = np.zeros(len(arr))
	sievePPUtilityArr = np.zeros(len(arr))
	sievePPMemoryArr = np.zeros(len(arr))
	preemptionUtilityArr = np.zeros(len(arr))
	preemptionMemoryArr = np.zeros(len(arr))
	
	data,M = loadData(sourceFile)
	sampleSize = int(len(data)*testPerc)
	
	start = time.time()
	subsample = random.sample(data,sampleSize)
	random.shuffle(subsample)
	simpleStream = loadStream(subsample)
	
	for j in range(0,numTrials):
		print 'Trial',j+1
		subsample = random.sample(data,sampleSize)
		random.shuffle(subsample)
		simpleStream = loadStream(subsample)		
		
		preemptionSolution = preemptionStreaming(simpleStream,k,1,function) #c=1
		preemptionValue = function(preemptionSolution)
		#print j,'preemption = ',preemptionValue,time.time()-start
		
		
		for i in range(0,len(arr)):
			eps = arr[i]
			
			preemptionUtilityArr[i] += preemptionValue /float(numTrials)
			preemptionMemoryArr[i] += k /float(numTrials)
			
			start = time.time()
			sieveSolution,sieveMemory = sieveStreaming(simpleStream,k,eps,function)
			sieveUtilityArr[i] += function(sieveSolution) /float(numTrials)
			sieveMemoryArr[i] += max(sieveMemory) /float(numTrials)

			start = time.time()
			sievePPSolution,sievePPMemory = sieveStreamingPlusPlus(simpleStream,k,eps,function)
			sievePPUtilityArr[i] += function(sievePPSolution) /float(numTrials)
			sievePPMemoryArr[i] += max(sievePPMemory) /float(numTrials)
	
	plt.xlabel(r'Granularity ($\epsilon$)',size = '25')
	plt.ylabel('Utility',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	blue_line = mlines.Line2D([], [], color='r', marker='o', linestyle='-',markersize=5, label='Sieve-Streaming')
	magenta_line = mlines.Line2D([], [], color='m', marker='D', linestyle='-',markersize=5, label='Preemption-Streaming')
	plt.legend(fontsize='x-large',handles=[green_line,blue_line,magenta_line])
	plt.plot(arr, sievePPUtilityArr, '-gs',linewidth=6.0,markersize=9.0)
	plt.plot(arr, sieveUtilityArr, '--ro',linewidth=2.0,markersize=5.0)
	plt.plot(arr, preemptionUtilityArr, '-mD',linewidth=2.0,markersize=5.0)
	savename = prefix+'-single-utility-k'+str(k)+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()

	plt.xlabel(r'Granularity ($\epsilon$)',size = '25')
	plt.ylabel('Memory',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	blue_line = mlines.Line2D([], [], color='r', marker='o', linestyle='-',markersize=5, label='Sieve-Streaming')
	magenta_line = mlines.Line2D([], [], color='m', marker='D', linestyle='-',markersize=5, label='Preemption-Streaming')
	plt.legend(fontsize='x-large',handles=[green_line,blue_line,magenta_line])
	plt.plot(arr, sievePPMemoryArr, '-go',linewidth=2.0,markersize=5.0)
	plt.plot(arr, sieveMemoryArr, '-rs',linewidth=2.0,markersize=5.0)
	plt.plot(arr, preemptionMemoryArr, '-mD',linewidth=2.0,markersize=5.0)
	savename = prefix+'-single-memory-k'+str(k)+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()	
	
# 	print savename
# 	print 'sieveUtilityArr =',sieveUtilityArr
# 	print 'sieveMemoryArr =',sieveMemoryArr
# 	print 'sievePPUtilityArr =',sievePPUtilityArr
# 	print 'sievePPMemoryArr =',sievePPMemoryArr
# 	print 'preemptionUtilityArr =',preemptionUtilityArr
# 	print 'preemptionMemoryArr =',preemptionMemoryArr

	return 0

def multiSourceK(arr,eps,L,function,loadData,loadSingle,loadMulti,sourceFile,numTrials,testPerc,prefix):
	#Compare multi-source algorithms
	
	#Keep 'eps' and 'L' constant and vary k, taking values from 'arr'.
	
	thresh1 = 1.0
	
	sieveUtility = np.zeros(len(arr))
	sieveMemory = np.zeros(len(arr))
	sieveLength = np.zeros(len(arr))
	sieveRounds = np.zeros(len(arr))
	
	batchUtility = np.zeros(len(arr))
	batchMemory = np.zeros(len(arr))
	batchLength = np.zeros(len(arr))
	batchRounds = np.zeros(len(arr))
		
	t1Utility = np.zeros(len(arr)) #This is for Sample-One-Streaming, I just used t1 as a code name.
	t1Memory = np.zeros(len(arr))
	t1Length = np.zeros(len(arr))
	t1Rounds = np.zeros(len(arr))
	
	randomUtility = np.zeros(len(arr))
	
	data,M = loadData(sourceFile) #Load in the data from the 'sourceFile' using the given loading function 'loadData'.
	sampleSize = int(len(data)*testPerc) #Each trial will subsample this loaded data.
	
	for j in range(0,numTrials):
		print j
		subsample = random.sample(data,sampleSize) #Take a random subset of the loaded data.
		random.shuffle(subsample) #Shuffle the subsample so trials are less similar.
		
		singleSource = loadSingle(subsample) #Sieve-Streaming++ and random uses this.
		multiSource = loadMulti(subsample) #Rest use this.
	
		for i in range(0,len(arr)):
			k = arr[i]
			
			start = time.time()
			batchSolution,batchMem,rounds = batchStreaming(function,multiSource,M,k,L,thresh1,eps)
			batchUtility[i] += function(batchSolution) /float(numTrials)
			batchMemory[i] += max(batchMem) /float(numTrials)
			batchLength[i] += len(batchSolution) /float(numTrials)
			batchRounds[i] += rounds /float(numTrials)
			#print time.time()-start
			
			start = time.time()
			t1Solution,t1Mem,rounds = batchStreaming1(function,multiSource,M,k,L,thresh1,eps)
			t1Utility[i] += function(t1Solution) /float(numTrials)
			t1Memory[i] += max(t1Mem) /float(numTrials)
			t1Length[i] += len(t1Solution) /float(numTrials)
			t1Rounds[i] += rounds /float(numTrials)
			#print time.time()-start
 			
			start = time.time()
			sieveSolution,sieveMem = sieveStreamingPlusPlus(singleSource,k,eps,function)
			sieveUtility[i] += function(sieveSolution) /float(numTrials)
			sieveMemory[i] += max(sieveMem) /float(numTrials)
			sieveLength[i] += len(sieveSolution) /float(numTrials)
			sieveRounds[i] += sampleSize/float(numTrials)
			#print time.time()-start
			#print '--'

			randomUtility[i] += randomBaseline(singleSource,k,function) /float(numTrials)
	
	plt.xlabel('Elements Selected (k)',size = '25')
	plt.ylabel('Utility',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='o', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	yellow_line = mlines.Line2D([], [], color='y', marker='x', linestyle=':',markersize=5, label='Random')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line,yellow_line])
	plt.plot(arr, sieveUtility, '-gs',linewidth=2.0,markersize=5.0)
	plt.plot(arr, batchUtility, color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.plot(arr, t1Utility, '-cs',linewidth=2.0,markersize=5.0)
	plt.plot(arr, randomUtility, ':yx',linewidth=2.0,markersize=5.0)
	plt.ylim(0,plt.ylim()[1])
	plt.xlim(plt.xlim())
	savename = prefix+'-multiUtility-L'+str(L)+'-e'+str(int(eps*100))+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
	plt.xlabel('Elements Selected (k)',size = '25')
	plt.ylabel('Adaptive Rounds',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='o', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line])
	plt.semilogy(arr, sieveRounds, '-gs',linewidth=2.0,markersize=5.0)
	plt.semilogy(arr, batchRounds, color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.semilogy(arr, t1Rounds, '-c^',linewidth=2.0,markersize=5.0)
	savename = prefix+'-multiAdaptive-L'+str(L)+'-e'+str(int(eps*100))+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
def multiSourceEps(arr,k,L,function,loadData,loadSingle,loadMulti,sourceFile,numTrials,testPerc,prefix):
	#Compare multi-source algorithms
	
	#Keep 'eps' and 'L' constant and vary k, taking values from 'arr'.
	
	thresh1 = 1.0
	
	sieveUtility = np.zeros(len(arr))
	sieveMemory = np.zeros(len(arr))
	sieveLength = np.zeros(len(arr))
	sieveRounds = np.zeros(len(arr))
	
	batchUtility = np.zeros(len(arr))
	batchMemory = np.zeros(len(arr))
	batchLength = np.zeros(len(arr))
	batchRounds = np.zeros(len(arr))	
	
	t1Utility = np.zeros(len(arr))
	t1Memory = np.zeros(len(arr))
	t1Length = np.zeros(len(arr))
	t1Rounds = np.zeros(len(arr))
	
	randomUtility = np.zeros(len(arr))
	
	data,M = loadData(sourceFile) #Load in the data from the 'sourceFile' using the given loading function 'loadData'.
	sampleSize = int(len(data)*testPerc) #Each trial will subsample this loaded data.
	print sampleSize
	
	for j in range(0,numTrials):
		print j
		subsample = random.sample(data,sampleSize) #Take a random subset of the loaded data.
		random.shuffle(subsample) #Shuffle the subsample so trials are less similar.
		
		singleSource = loadSingle(subsample)
		multiSource = loadMulti(subsample)
	
		for i in range(0,len(arr)):
			eps = arr[i]
			
			start = time.time()
			batchSolution,batchMem,rounds = batchStreaming(function,multiSource,M,k,L,thresh1,eps)
			batchUtility[i] += function(batchSolution) /float(numTrials)
			batchMemory[i] += max(batchMem) /float(numTrials)
			batchLength[i] += len(batchSolution) /float(numTrials)
			batchRounds[i] += rounds /float(numTrials)
			#print time.time()-start
			
			start = time.time()
			t1Solution,t1Mem,rounds = batchStreaming1(function,multiSource,M,k,L,thresh1,eps)
			t1Utility[i] += function(t1Solution) /float(numTrials)
			t1Memory[i] += max(t1Mem) /float(numTrials)
			t1Length[i] += len(t1Solution) /float(numTrials)
			t1Rounds[i] += rounds /float(numTrials)
			#print time.time()-start

			
			start = time.time()
			sieveSolution,sieveMem = sieveStreamingPlusPlus(singleSource,k,eps,function)
			sieveUtility[i] += function(sieveSolution) /float(numTrials)
			sieveMemory[i] += max(sieveMem) /float(numTrials)
			sieveLength[i] += len(sieveSolution) /float(numTrials)
			sieveRounds[i] += sampleSize /float(numTrials)
			#print time.time()-start
			print '--'

			
			randomUtility[i] += randomBaseline(singleSource,k,function) /float(numTrials)

	
	plt.xlabel(r'Granularity ($\epsilon$)',size = '25')
	plt.ylabel('Utility',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='o', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	yellow_line = mlines.Line2D([], [], color='y', marker='x', linestyle=':',markersize=5, label='Random')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line,yellow_line])
	plt.plot(arr, sieveUtility, '-gs',linewidth=2.0,markersize=5.0)
	plt.plot(arr, batchUtility, color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.plot(arr, randomUtility, ':yx',linewidth=2.0,markersize=5.0)
	plt.plot(arr, t1Utility, '-c^',linewidth=2.0,markersize=5.0)
	plt.ylim(0,plt.ylim()[1])
	plt.xlim(plt.xlim())
	savename = prefix+'-multiUtility-L'+str(L)+'-k'+str(k)+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
	plt.xlabel(r'Granularity ($\epsilon$)',size = '25')
	plt.ylabel('Adaptive Rounds',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='s', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line])
	plt.semilogy(arr, sieveRounds, '-gs',linewidth=2.0,markersize=5.0)
	plt.semilogy(arr, batchRounds, color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.semilogy(arr, t1Rounds, '-c^',linewidth=2.0,markersize=5.0)
	savename = prefix+'-multiAdaptive-L'+str(L)+'-k'+str(k)+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
def multiSourceL(arr,k,eps,function,loadData,loadSingle,loadMulti,sourceFile,numTrials,testPerc,prefix):
	
	thresh1 = 1.0
	
	sieveUtility = np.zeros(len(arr))
	sieveMemory = np.zeros(len(arr))
	sieveLength = np.zeros(len(arr))
	sieveRounds = np.zeros(len(arr))
	
	batchUtility = np.zeros(len(arr))
	batchMemory = np.zeros(len(arr))
	batchLength = np.zeros(len(arr))
	batchRounds = np.zeros(len(arr))
		
	t1Utility = np.zeros(len(arr))
	t1Memory = np.zeros(len(arr))
	t1Length = np.zeros(len(arr))
	t1Rounds = np.zeros(len(arr))
	
	randomUtility = np.zeros(len(arr))
	
	data,M = loadData(sourceFile) #Load in the data from the 'sourceFile' using the given loading function 'loadData'.
	print len(data)
	sampleSize = int(len(data)*testPerc) #Each trial will subsample this loaded data.
	print sampleSize
	
	for j in range(0,numTrials):
		print j
		subsample = random.sample(data,sampleSize) #Take a random subset of the loaded data.
		random.shuffle(subsample) #Shuffle the subsample so trials are less similar.
		
		singleSource = loadSingle(subsample)
		multiSource = loadMulti(subsample)
	
		start = time.time()
		sieveSolution,sieveMem = sieveStreamingPlusPlus(singleSource,k,eps,function)
		randVal = randomBaseline(singleSource,k,function) /float(numTrials)
		print time.time()-start
	
		for i in range(0,len(arr)):
			L = arr[i]
			print L
			
			start = time.time()
			batchSolution,batchMem,rounds = batchStreaming(function,multiSource,M,k,L,thresh1,eps)
			batchUtility[i] += function(batchSolution) /float(numTrials)
			batchMemory[i] += max(batchMem) /float(numTrials)
			batchLength[i] += len(batchSolution) /float(numTrials)
			batchRounds[i] += rounds /float(numTrials)
			print time.time()-start
			
			start = time.time()
			t1Solution,t1Mem,rounds = batchStreaming1(function,multiSource,M,k,L,thresh1,eps)
			t1Utility[i] += function(t1Solution) /float(numTrials)
			t1Memory[i] += max(t1Mem) /float(numTrials)
			t1Length[i] += len(t1Solution) /float(numTrials)
			t1Rounds[i] += rounds /float(numTrials)
			print time.time()-start
			
			sieveSolution,sieveMem = sieveStreamingPlusPlus(singleSource,k,eps,function)
			sieveUtility[i] += function(sieveSolution) /float(numTrials)
			sieveMemory[i] += max(sieveMem) /float(numTrials)
			sieveLength[i] += len(sieveSolution) /float(numTrials)
			sieveRounds[i] += sampleSize /float(numTrials)
			#print time.time()-start
			#print '--'
			
			randomUtility[i] += randVal

	plt.xlabel('Buffer Size (B)',size = '25')
	plt.ylabel('Utility',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='o', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	yellow_line = mlines.Line2D([], [], color='y', marker='x', linestyle=':',markersize=5, label='Random')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line,yellow_line])
	plt.semilogx(arr, sieveUtility, '-gs',linewidth=2.0,markersize=5.0)
	plt.semilogx(arr, batchUtility,color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.semilogx(arr, t1Utility, '-c^',linewidth=2.0,markersize=5.0)
	plt.semilogx(arr, randomUtility, ':yx',linewidth=2.0,markersize=5.0)
	savename = prefix+'-multiUtility-k'+str(k)+'-e'+str(int(eps*100))+'.pdf'
	plt.ylim(0,plt.ylim()[1])
	plt.xlim(plt.xlim())
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
	plt.xlabel('Buffer Size (B)',size = '25')
	plt.ylabel('Adaptive Rounds',size='25')
	plt.xticks(arr,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Sieve-Streaming++')
	red_line = mlines.Line2D([], [], color='orange', marker='s', linestyle='-',markersize=5, label='Batch Sieve-Streaming++')
	cyan_line =  mlines.Line2D([], [], color='c', marker='^', linestyle='-',markersize=5, label='Sample-One-Streaming')
	plt.legend(fontsize = 'x-large',handles=[cyan_line,green_line,red_line])
	plt.loglog(arr, sieveRounds, '-gs',linewidth=2.0,markersize=5.0)
	plt.loglog(arr, batchRounds, color='orange', marker='o', linestyle='-',linewidth=2.0,markersize=5.0)
	plt.loglog(arr, t1Rounds, '-c^',linewidth=2.0,markersize=5.0)
	savename = prefix+'-multiAdaptive-k'+str(k)+'-e'+str(int(eps*100))+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
	
# 	print 'lArr = ',arr
# 	print 'batchUtility =',batchUtility
# 	print 't1Utility =',t1Utility
# 	print 'sieveUtility =',sieveUtility
# 	print 'batchRounds =',batchRounds
# 	print 't1Rounds =',t1Rounds
# 	print 'sieveRounds =',sieveRounds

	return 0

def adaptCommScatter(rArr,L,k,eps,function,loadData,loadMulti,sourceFile,numTrials,testPerc,prefix):
	
	thresh1 = 1.0
	
	data,M = loadData(sourceFile) #Load in the data from the 'sourceFile' using the given loading function 'loadData'.
	#print len(data)
	sampleSize = int(len(data)*testPerc) #Each trial will subsample this loaded data.
	#print sampleSize

	commArr = []
	adaptArr = []
	utilArr = []
	colors = []
	

	for j in range(0,numTrials):
		print 'Trial Number:',j+1
		subsample = random.sample(data,sampleSize) #Take a random subset of the loaded data.
		random.shuffle(subsample) #Shuffle the subsample so trials are less similar.
		
		multiSource = loadMulti(subsample)
		
		for i in range(0,len(rArr)):
			R = rArr[i]
			start = time.time()
			solution,memory,rounds,comm = batchStreamingR(function,multiSource,M,k,L,thresh1,eps,R)		
			val = function(solution)
			
			utilArr.append(val)
			commArr.append(comm)
			adaptArr.append(rounds)
			colors.append(i)

	
	#print 'commArr = ',commArr
	#print 'adaptArr = ',adaptArr
	#print 'utilArr = ',utilArr
	
	plt.xlabel('Communication Cost',size = '25')
	plt.ylabel('Adaptive Rounds',size='25')
	#plt.xscale('log')
	#plt.yscale('linear')
	plt.scatter(commArr, adaptArr, c = colors, )
	savename = prefix+'-tradeoffAdapt-e'+str(int(eps*100))+'-k'+str(k)+'-L'+str(L)+'-'+str(sampleSize)+'-'+str(numTrials)+'.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()
		
	#print savename
	return 0

#============
# Main Code
#============

if __name__ == '__main__':

	sourceYoutube = 'YouTube_vectors.txt'
	sourceTwitter = 'tweets.txt'
	numTrials = 5 #Number of trials to run for the graphs (output average).
	testPercY = 1.0 #Fraction YouTube dataset to use when testing
	testPercT = 0.2 #Fraction of Twitter dataset to use when testing 
	#Note that the Twitter function is slow to evaluate and the dataset is large, so using a large value here will result in a long runtime.
	#The preemptionStreaming algorithm is also very slow, so singleSourceEps and singleSourceK will be slow too.

#-------

	#This generates graphs that show how utility and adaptivity vary with L.
	#The first call to multiSourceL() is for the Twitter dataset, the second is for YouTube.
	#The YouTube dataset should use different parameters for k and eps, see Appendix.
	lArr = [4,16,64,256,1024]
	k = 50
	eps = 0.7
	
	#Twitter
	multiSourceL(lArr,k,eps,tweetDiversity,loadTwitterData,createSimpleStream,createHybridStream,sourceTwitter,numTrials,testPercT,'twitter')
	
	#YouTube
	#multiSourceL(lArr,k,eps,logdet,importVectors,simpleStreamYoutube,hybridStreamYoutube,sourceYoutube,numTrials,testPercY,'video')

#-------
	
	#This generates graphs that vary epsilon and keep L (buffer size) and k constant.
	#Note that for singleSource, buffer size doesn't do anything.
	L = 100
	epsArr = [0.1,0.3,0.5,0.7]
	k = 100
	
	#Twitter
	multiSourceEps(epsArr,k,L,tweetDiversity,loadTwitterData,createSimpleStream,createHybridStream,sourceTwitter,numTrials,testPercT,'twitter')
	
	#YouTube
	#multiSourceEps(epsArr,k,L,logdet,importVectors,simpleStreamYoutube,hybridStreamYoutube,sourceYoutube,numTrials,testPercY,'video')
	
	#Twitter
	#singleSourceEps(epsArr,k,logdet,importVectors,simpleStreamYoutube,sourceYoutube,numTrials,testPercY,'video')
	
	#YouTube
	#singleSourceEps(epsArr,k,tweetDiversity,loadTwitterData,createSimpleStream,sourceTwitter,numTrials,testPercT,'twitter')

#-------

	#This generates graphs that vary k and keep L (buffer size) and eps constant.
	#Note that for singleSource, buffer size doesn't do anything.
	L = 100
	kArr = [40,60,80,100]
	eps = 0.25
	
	#Twitter
	multiSourceK(kArr,eps,L,tweetDiversity,loadTwitterData,createSimpleStream,createHybridStream,sourceTwitter,numTrials,testPercT,'twitter')
	
	#YouTube
	#multiSourceK(kArr,eps,L,logdet,importVectors,simpleStreamYoutube,hybridStreamYoutube,sourceYoutube,numTrials,testPercY,'video')
	
	#Twitter
	#singleSourceK(kArr,eps,logdet,importVectors,simpleStreamYoutube,sourceYoutube,numTrials,testPercY,'video')
	
	#YouTube
	#singleSourceK(kArr,eps,tweetDiversity,loadTwitterData,createSimpleStream,sourceTwitter,numTrials,testPercT,'twitter')

#-------

	#This generates graphs that vary L (buffer size) and keep epsilon and  k constant.
	rArr = [1,2,3,4]
	k = 100
	e = 0.5
	L = 299
	
	#Twitter
	adaptCommScatter(rArr,L,k,eps,tweetDiversity,loadTwitterData,createHybridStream,sourceTwitter,numTrials,testPercT,'twitter')
	
	#YouTube
	#adaptCommScatter(rArr,L,k,eps,logdet,importVectors,hybridStreamYoutube,sourceYoutube,numTrials,testPercY,'video')


