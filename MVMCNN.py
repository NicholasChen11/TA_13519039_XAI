import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Metric import Euclidean
from utils import arrayAddition, arrayMultConst, countDifference

class MVMCNN():
    def __init__(self, k_neighbors=5, metric=Euclidean, p=2, clustering_method="KMeans"):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.p = p
        self.clustering_method = clustering_method
    
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

        for label in self.train:
            listOfNCluster = [2,3,4,5]
            dataSize = len(self.train[label])
            maxScore = -1
            maxLabel = []

            # search for n with highest silhouette_score
            for n in listOfNCluster:
                kmeans = KMeans(n_clusters=n)
                kmeans.fit(self.train[label])
                kmeansLabel = kmeans.predict(self.train[label])
                score = silhouette_score(self.train[label], kmeansLabel)

                if maxScore < score:
                    maxScore = score
                    maxLabel = kmeansLabel.copy()
            
            # assign new cluster data structure to self.train
            cluster = {}
            for i in range(dataSize):
                curr_data = self.train[label][i]
                curr_target = maxLabel[i]

                if curr_target not in cluster:
                    cluster[curr_target] = []
                    
                cluster[curr_target] += [curr_data]
            
            self.train[label] = cluster
        # new self.train structure:
        # self.train = {
        #   "label_1" : {
        #       "cluster_1": [data_1, data_2],
        #       "cluster_2": [data_5, data_6],
        #   }
        #   "label_2" : {
        #       "cluster_1": [data_3, data_4],
        #       "cluster_2": [data_7, data_8],
        #   }
        # }
    
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
            #   "label_1" : {
            #       "cluster_1": weightedSum_1,
            #       "cluster_2": weightedSum_2,
            #   }
            #   "label_2" : {
            #       "cluster_1": weightedSum_3,
            #       "cluster_2": weightedSum_4,
            #   }
            # }
            for target in self.train.keys():
                localKNeighbors[target] = {}
                target_distance[target] = {}
                for cluster in self.train[target]:
                    KNeighbors = self.findKNeighbors(curr_test, target, cluster)
                    localKNeighbors[target][cluster] = self.getLocalKNeighbors(KNeighbors, target, cluster)
                    target_distance[target][cluster] = self.getTargetDistance(localKNeighbors[target][cluster], curr_test)

            # search most minimum distance
            firstKey = list(target_distance.keys())[0]
            minDistance = target_distance[firstKey][0]
            minKey = firstKey
            for target in self.train:
                for cluster in self.train[target]:
                    if minDistance > target_distance[target][cluster]:
                        minDistance = target_distance[target][cluster]
                        minKey = target

            result[i] = minKey
            # result[i] = min(target_distance, key=target_distance.get)

        return numpy.array(result)

    def predict_proba(self, X_test):
        n_tests = len(X_test)
        class_types = list(set(self.y))
        class_types.sort()
        result = [[0 for _ in range(len(class_types))] for _ in range(n_tests)]
        
        for i in range(n_tests):
            curr_test = X_test[i]
            localKNeighbors = {}
            target_distance = {}
            total_p = 0

            # Calculating distance in each data's label
            # target_distance = {
            #   "label_1" : {
            #       "cluster_1": weightedSum_1,
            #       "cluster_2": weightedSum_2,
            #   }
            #   "label_2" : {
            #       "cluster_1": weightedSum_3,
            #       "cluster_2": weightedSum_4,
            #   }
            # }
            for target in self.train.keys():
                localKNeighbors[target] = {}
                target_distance[target] = {}
                for cluster in self.train[target]:
                    KNeighbors = self.findKNeighbors(curr_test, target, cluster)
                    localKNeighbors[target][cluster] = self.getLocalKNeighbors(KNeighbors, target, cluster)
                    target_distance[target][cluster] = self.getTargetDistance(localKNeighbors[target][cluster], curr_test)

            # search most minimum distance for each class
            # format result: [[ False_value, True_value ]]
            # Calculation Process
            #   for each class (class_idx) in every data (i), calculate:
            #   p[i][class_idx] (stand for: probability) = 1 / target_distance[class_idx]
            #   total_p[i] = total all probability in data 'i'
            #   result[i][class_idx] = p[i][class_idx] / total_p[i]
            for classIdx in range(len(class_types)):
                target = class_types[classIdx]
                minDistance = target_distance[target][0]
                for cluster in self.train[target]:
                    if minDistance > target_distance[target][cluster]:
                        minDistance = target_distance[target][cluster]

                result[i][classIdx] = 1 / minDistance
                total_p += result[i][classIdx]

            for classIdx in range(len(result[i])):
                result[i][classIdx] = result[i][classIdx] / total_p

        return numpy.array(result)
        
    def findKNeighbors(self, currTest, target, cluster):
        """Get list of indices of k smallest distance from currTest to every point in self.train['target']

        Args:
            currTest (Array): array of size (n_features)
            target (int): target of the data

        Returns:
            Array: list of indices with size (self.k_neighbors)
        """
        distanceArray = self.getDistanceArray(currTest, target, cluster)
        KNeighbors = numpy.argsort(distanceArray)[0:self.k_neighbors]
        
        return KNeighbors
        
    def getDistanceArray(self, currTest, target, cluster):
        """Count distance from currTest to every point in self.train['target']

        Args:
            currTest (Array): array of size (n_features)
            target (int): target of the data

        Returns:
            Array: list of distance from currTest to every point in self.train['target']
        """
        n_samples = len(self.train[target][cluster])
        distanceArray = [0 for _ in range(n_samples)]
        
        for i in range(n_samples):
            distanceArray[i] = self.metric(self.train[target][cluster][i], currTest, self.p)
            
        return distanceArray
        
    def getLocalKNeighbors(self, KNeighbors, target, cluster):
        localKNeighbors = []
        
        for i in range(self.k_neighbors):
            localVector = [0 for _ in range(len(self.X[0]))]
            for j in range(i+1):
                localVector = arrayAddition(localVector, self.train[target][cluster][KNeighbors[j]])
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
