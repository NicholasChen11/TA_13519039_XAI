from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy

from KNN import KNN
from PNN import PNN
from LMPNN import LMPNN
from MVMCNN import MVMCNN
# from Metric import Chebyshev, Euclidean, Manhattan, Minkowski
# from utils import arrayAddition, arrayMultConst
  
# Loading data
irisData = load_iris()
# print(f"Nicho {irisData.items()}")
# print(f"Nicho {irisData.keys()}")
# print(f"Nicho {irisData.data}")
# print(f"Nicho {type(irisData.data)}")
  
# Create feature and target arrays
X = irisData.data
y = irisData.target
k = 7
  
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# target label
print("y test")
print(y_test)

# sklearn KNN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(knn.score(X_test, y_test))

# my KNN
myknn = KNN(k_neighbors=k)
myknn.fit(X_train, y_train)
print()
print("my KNN")
print(myknn.predict(X_test))
print(myknn.score(X_test, y_test))
myknn.showScoreInGraph(numpy.arange(1, 9), X_test, y_test)

# PNN
pnn = PNN(k_neighbors=k)
pnn.fit(X_train, y_train)
print()
print("PNN")
print(pnn.predict(X_test))
print(pnn.score(X_test, y_test))
pnn.showScoreInGraph(numpy.arange(1, 9), X_test, y_test)

# LMPNN
lmpnn = LMPNN(k_neighbors=k)
lmpnn.fit(X_train, y_train)
print()
print("LMPNN")
print(lmpnn.predict(X_test))
print(lmpnn.score(X_test, y_test))
lmpnn.showScoreInGraph(numpy.arange(1, 9), X_test, y_test)

# MVMCNN
mvmcnn = MVMCNN(k_neighbors=k)
mvmcnn.fit(X_train, y_train)
print()
print("MVMCNN")
print(mvmcnn.predict(X_test))
print(mvmcnn.score(X_test, y_test))
mvmcnn.showScoreInGraph(numpy.arange(1, 9), X_test, y_test)






# print(Euclidean([1,2,3], [4,5,6]))
# print(Manhattan([1,2,3], [4,5,6]))
# print(Chebyshev([1,1,3], [4,5,6]))
# print(Minkowski([1,2,3], [4,5,6], 1))
# print(Minkowski([1,2,3], [4,5,6], 2))
# print(f"Nicho countDiff {countDifference([1,2,3], [1,1,2])}")
# a1 = [1,2,3]
# a2 = [4,5,6]
# print(arrayMultConst(a1, 2))
# print(a1)
# print(a2)