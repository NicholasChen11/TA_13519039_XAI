import numpy
import matplotlib.pyplot as plt

from Metric import Euclidean
from utils import countDifference, most_frequent

class KNN():
  def __init__(self, k_neighbors=5, metric=Euclidean, p=2):
    self.k_neighbors = k_neighbors
    self.metric = metric
    self.p = p
    
  def fit(self, X_train, y_train):
    """Fit the k-nearest neighbors classifier from the training dataset.

    Args:
        X_train (Matrix): matrix of size (n_samples, n_features)
        y_train (Array): array of size (n_samples)
    """
    self.X = X_train
    self.y = y_train
  
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
      currTest = X_test[i]
      KNeighbors = self.findKNeighbors(currTest)
      y_KNeighbors = list(map(lambda idx : self.y[idx], KNeighbors))
      result[i] = most_frequent(y_KNeighbors)
    
    return numpy.array(result)
      
  def findKNeighbors(self, currTest):
    """Get list of indices of k smallest distance from currTest to every point in self.X

    Args:
        currTest (Array): array of size (n_features)

    Returns:
        Array: list of indices with size (self.k_neighbors)
    """
    distanceArray = self.getDistanceArray(currTest)
    KNeighbors = numpy.argsort(distanceArray)[0:self.k_neighbors]
    
    return KNeighbors

  def getDistanceArray(self, currTest):
    """Count distance from currTest to every point in self.X

    Args:
        currTest (Array): array of size (n_features)

    Returns:
        Array: list of distance from currTest to every point in self.X
    """
    n_samples = len(self.X)
    distanceArray = [0 for _ in range(n_samples)]
    
    for i in range(n_samples):
      distanceArray[i] = self.metric(self.X[i], currTest, self.p)
    
    return distanceArray

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