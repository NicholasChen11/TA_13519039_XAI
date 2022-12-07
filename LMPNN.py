import numpy
import matplotlib.pyplot as plt

from Metric import Euclidean
from utils import arrayAddition, arrayMultConst, countDifference

class LMPNN():
	def __init__(self, k_neighbors=5, metric=Euclidean, p=2):
		self.k_neighbors = k_neighbors
		self.metric = metric
		self.p = p

	def fit(self, X_train, y_train):
		"""Fit the pseudo-nearest neighbors classifier from the training dataset.

		Args:
				X_train (Matrix): matrix of size (n_samples, n_features)
				y_train (Array): array of size (n_samples)
		"""
		self.X = X_train
		self.y = y_train
		self.train = {}
		n_samples = len(X_train)

		# Dividing the train data by data's label
    # self.train = {
    #   "label_1" : [data_1, data_2]
    #   "label_2" : [data_3, data_4]
    # }
		for i in range(n_samples):
			curr_sample_data = X_train[i]
			curr_sample_target = y_train[i]
			
			if curr_sample_target not in self.train:
				self.train[curr_sample_target] = []
				
			self.train[curr_sample_target] += [curr_sample_data]
	
	def predict(self, X_test):
		"""Predict the class of given X_test

    Args:
        X_test (Matrix): matrix of size (n_tests, n_features)

    Returns:
        Array: array of size (n_tests)
    """
		n_tests = len(X_test)
		result = [-1 for _ in range(n_tests)]
		
		for i in range(n_tests):
			curr_test = X_test[i]
			localKNeighbors = {}
			target_distance = {}
      
      # Calculating distance in each data's label
      # target_distance = {
      #   "label_1" : weightedSum_1
      #   "label_2" : weightedSum_2
      # } 
			for target in self.train.keys():
				KNeighbors = self.findKNeighbors(curr_test, target)
				localKNeighbors[target] = self.getLocalKNeighbors(KNeighbors, target)
				target_distance[target] = self.getTargetDistance(localKNeighbors[target], curr_test)
				
			result[i] = min(target_distance, key=target_distance.get)
			
		return numpy.array(result)
	
	def findKNeighbors(self, currTest, target):
		"""Get list of indices of k smallest distance from currTest to every point in self.train['target']

    Args:
        currTest (Array): array of size (n_features)
        target (int): target of the data

    Returns:
        Array: list of indices with size (self.k_neighbors)
    """
		distanceArray = self.getDistanceArray(currTest, target)
		KNeighbors = numpy.argsort(distanceArray)[0:self.k_neighbors]
		
		return KNeighbors
		
	def getDistanceArray(self, currTest, target):
		"""Count distance from currTest to every point in self.train['target']

    Args:
        currTest (Array): array of size (n_features)
        target (int): target of the data

    Returns:
        Array: list of distance from currTest to every point in self.train['target']
    """
		n_samples = len(self.train[target])
		distanceArray = [0 for _ in range(n_samples)]
		
		for i in range(n_samples):
			distanceArray[i] = self.metric(self.train[target][i], currTest, self.p)
			
		return distanceArray
	
	def getLocalKNeighbors(self, KNeighbors, target):
		localKNeighbors = []

		for i in range(self.k_neighbors):
			localVector = [0 for _ in range(len(self.X[0]))]
			for j in range(i+1):
				localVector = arrayAddition(localVector, self.train[target][KNeighbors[j]])
			localKNeighbors += [arrayMultConst(localVector, 1/(i+1))]
		
		return localKNeighbors

	def getTargetDistance(self, localKNeighbors, currTest):
		targetDistance = 0

		for i in range(len(localKNeighbors)):
			targetDistance += 1/(i+1) * self.metric(localKNeighbors[i], currTest, self.p)
		
		return targetDistance
	
	def score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		totalMiss = countDifference(y_test, y_predict)
		
		return 1 - totalMiss/len(y_test)
		
	def showScoreInGraph(self, k_neighbors_list, X_test, y_test):
		initial_k = self.k_neighbors
		train_accuracy = numpy.empty(len(k_neighbors_list))
		test_accuracy = numpy.empty(len(k_neighbors_list))
		
		for i, k in enumerate(k_neighbors_list):
			self.k_neighbors = k
        
      # Compute training and test data accuracy
			train_accuracy[i] = self.score(self.X, self.y)
			test_accuracy[i] = self.score(X_test, y_test)
    
    # reset k_neigbors to initial value
		self.k_neighbors = initial_k
		
		plt.plot(k_neighbors_list, train_accuracy, label='Training dataset Accuracy')
		plt.plot(k_neighbors_list, test_accuracy, label='Testing dataset Accuracy')
		
		plt.legend()
		plt.xlabel('k_neighbors')
		plt.ylabel('Accuracy')
		plt.show()