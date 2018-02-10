#!/usr/bin/env python

"""
Filename: hw2.py
Author: Alex Meza
Date: 10/30/17
Description:
	...
"""


import numpy as np
import pandas as pd
import scipy.optimize
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from math import log
from math import exp

def main():
	
	#pre stored data base of beers
	data = pd.read_pickle("./beer.pkl")

	#for question 8
	ipa = [row['beer/style']=="American IPA" for ind,row in data.iterrows()]
	train_ipa = ipa[:int(len(ipa)/3)]
	#create feature matrix and label for if beer has abv>=6.5%
	
#QUESTION 1	
	print("Question 1:\n")
	#Create feature matrix and create pickle for easy access
	X = [feature(row) for index, row in data.iterrows()]
	X = np.array(X)
	pd.DataFrame(X).to_pickle("./q1Features")
	
	y = [row['beer/ABV'] >= 6.5 for index, row in data.iterrows()]
	
	#read pickle of data so dont have to recreate matrix
	
	X = pd.read_pickle("./q1Features")
	X = np.array(X)
	#create training, validation, and test sets
	Xtrain = X[:int(len(X)/3)]
	Xval = X[int(len(X)/3):int(2*len(X)/3)]
	Xtest = X[int(2*len(X)/3):]
	
	ytrain = vim y[:int(len(y)/3)]
	yval = y[int(len(y)/3):int(2*len(y)/3)]
	ytest = y[int(2*len(y)/3):]
	
	lam = 1
	theta = train(Xtrain, ytrain, lam)

	acc1 = performance(Xval.dot(theta), yval)
	acc2 = performance(Xtest.dot(theta), ytest)
	print('Accuracy of Validation, and Testing Sets:',acc1, acc2)
	
#QUESTION #2
	print("\nQuestion 2:\n")
	#create feature matrix of occurences of keywords in each review
	#X = [reviewFeature(row['review/text'].lower()) for ind, row in data.iterrows()]
	#X = np.array(X)
	#original command to create pickle for easier access
	#pd.DataFrame(X).to_pickle("./reviewMatrix.pkl")
	
	#read dataframe from pickle(SERIOUSLY SO MUCH FASTER)
	X = pd.read_pickle("./reviewMatrix.pkl")
	X = np.array(X)
	
	#create testing, training, val, sets
	
	Xtrain = X[:int(len(X)/3)]
	Xval = X[int(len(X)/3):int(2*len(X)/3)]
	Xtest = X[int(2*len(X)/3):]
	
	#create classifier based on if “lactic,” “tart,” “sour,” “citric,” “sweet,” “acid,” “hop,” “fruit,” “salt,” “spicy.” are in the review
	lam =1
	theta = train(Xtrain, ytrain, lam)
	theta = np.array(theta)
#Question 3
	print("\nQuestion 3:\n")
	#print out the balanced errorrate of the test
	ber = BER(Xtest.dot(theta), ytest)
	print("Ber of classifier made from keywords in review is: ", ber)

#Question 4
	print("\nQuestion 4:\n")
	C = [0, .01, .1, 1, 10, 100]
		
	theta = [train(Xtrain, ytrain, c) for c in C]
	performances = [performance(Xval.dot(t), yval) for t in theta]
	index_of_max = performances.index(max(performances))
	optimum_theta = theta[index_of_max]

	train_performance = performance(Xtrain.dot(optimum_theta), ytrain)
	val_performance = max(performances)
	test_performance = performance(Xtest.dot(optimum_theta), ytest)

	print("Performance of Optimum Training Prediction: ", train_performance)
	print("Performance of Optimum Validation Prediction: ", val_performance)
	print("Performance of Optimum Testing Prediction: ", test_performance)

	
#Question 5
	print("\nQuestion 5:\n")
	X = X[:, 1:]
	Xtrain = X[:int(len(X)/3)]
	Xval = X[int(len(X)/3):int(2*len(X)/3)]
	Xtest = X[int(2*len(X)/3):]

	pca = PCA(n_components=10)
	pca.fit(Xtrain)
	print("PCA components of Xtrain:\n",pca.components_)

#Question 6
	print("\nQuestin 6:\n")
	X0 = pca.transform(Xtrain)
	print("First data point after dimentionality reduction:\n", X0[0])

#Question 7
	print("\nQuestion 7:\n")
	dim_reduct_error = sum(pca.explained_variance_[2:])*len(Xtrain)
	print("Error of using only 2 components: ", dim_reduct_error)


#Problem 8
	print("\n Question 8: \n")
	pca = PCA(n_components=2)

	X_reduct_train = pca.fit_transform(Xtrain).tolist()
	for (is_ipa, [x,y]) in zip(train_ipa, X_reduct_train):
		if(is_ipa):
			plt.scatter(x,y,c='r')
		else:
			plt.scatter(x,y,c='b')	

	plt.show()
def train(X_train, y_train, lam):
	theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X_train[0]), fprime, pgtol = 10,     args = (X_train, y_train, lam))
	return theta

#performance of method 
def performance(scores, y):
	predictions = [s > 0 for s in scores]
	correct = [(a==b) for (a,b) in zip(predictions,y)]
	acc = sum(correct) * 1.0 / len(correct)
 
	return acc
def BER(scores, y):
	
	predictions = [s >0  for s in scores]
	falsepos, falseneg, truepos, trueneg =0, 0,0,0
	for (a,b) in zip(predictions,y):
		if(a==b):
			if(a==True):
				truepos+=1
			else:
				trueneg+=1
		elif(a==True):
			falsepos+=1
		else:
			falseneg+=1
	print ("False Positives:" ,falsepos," False Negatives:", falseneg, " True Positives: ", truepos, " True Negatives: ", trueneg)
	return .5*((falseneg/(falseneg+truepos)+falsepos/(falsepos+trueneg)))

#get review feature vector for feature matrix
def reviewFeature(review):
	
	occurences =[]
	keywords = ["lactic","tart","sour","citric", 'sweet', 'acid', 'hop','fruit', 'salt', 'spicy']
	occurences.append(1)
	
	for word in keywords:
		count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), review))
		occurences.append(count)
		
	return occurences

#get feature for feature matrix
def feature(datum):
	feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
	return feat

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
	loglikelihood = 0
	for i in range(len(X)):
		logit = inner(X[i], theta)
		loglikelihood -= log(1 + exp(-logit))
		if not y[i]:
			loglikelihood -= logit
	for k in range(len(theta)):
		loglikelihood -= lam * theta[k]*theta[k]
	# for debugging
	# print("ll =" + str(loglikelihood))
	return -float(loglikelihood)

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
	dl = [0]*len(theta)
	for i in range(len(X)):
		logit = inner(X[i], theta)
		for k in range(len(theta)):
			dl[k] += X[i][k] * (1 - sigmoid(logit))
			if not y[i]:
				dl[k] -= X[i][k]
	for k in range(len(theta)):
		dl[k] -= lam*2*theta[k]
	return np.array([-x for x in dl])


def inner(x,y):
	return sum([x[i]*y[i] for i in range(len(x))])
 
def sigmoid(x):
	return 1.0 / (1 + exp(-x))

if __name__ == '__main__':
	main()