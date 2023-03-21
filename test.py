from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy

from KNN import KNN
from PNN import PNN
from LMPNN import LMPNN
from MVMCNN import MVMCNN
# from Metric import Chebyshev, Euclidean, Manhattan, Minkowski
from utils import arrayAddition, arrayMultConst, readCSV
import shap
  
# Loading data
# irisData = load_iris()

# bentuk data: 
# - tuple isi 2 nilai, x dan y
# - x adalah matrix [351, 34], 351 baris, 34 kolom
# - y adalah array [351], 351 baris 
ionosphereData = readCSV("ionosphere_csv.csv")

# print(f"Nicho {irisData.items()}")
# print(f"Nicho {irisData.keys()}")
# print(f"Nicho {irisData.data}")
# print(f"Nicho {type(irisData.data)}")
  
# Create feature and target arrays
# X = irisData.data
# y = irisData.target
k = 7
k_range = numpy.arange(1, 12)
X = ionosphereData[0]
y = ionosphereData[1]
y = list(map(lambda a : a == 'g', y))
  
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# target label
print("y test")
print(y_test)
print("")

# # sklearn KNN
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)
# print("sklearn KNN")
# print(knn.predict(X_test))
# print("Score: " + str(knn.score(X_test, y_test)))

# # Get shap values
# shap.initjs()
# explainer = shap.KernelExplainer(knn.predict, numpy.array(X_test))
# svc_shap_values = explainer.shap_values(numpy.array(X_test), nsamples=100)
# shap.summary_plot(svc_shap_values, numpy.array(X_test))

# # my KNN
# myknn = KNN(k_neighbors=k)
# myknn.fit(X_train, y_train)
# print()
# print("my KNN")
# print(myknn.predict(X_test))
# print(myknn.score(X_test, y_test))
# myknn.showScoreInGraph(k_range, X_test, y_test)

# # Get shap values
# shap.initjs()
# explainer = shap.KernelExplainer(myknn.predict, numpy.array(X_test))
# svc_shap_values = explainer.shap_values(numpy.array(X_test), nsamples=100)
# shap.summary_plot(svc_shap_values, numpy.array(X_test))

# # PNN
# pnn = PNN(k_neighbors=k)
# pnn.fit(X_train, y_train)
# print()
# print("PNN")
# print(pnn.predict(X_test))
# print(pnn.score(X_test, y_test))
# pnn.showScoreInGraph(k_range, X_test, y_test)

# # Get shap values
# shap.initjs()
# explainer = shap.KernelExplainer(pnn.predict, numpy.array(X_test))
# svc_shap_values = explainer.shap_values(numpy.array(X_test), nsamples=100)
# shap.summary_plot(svc_shap_values, numpy.array(X_test))

# # LMPNN
# lmpnn = LMPNN(k_neighbors=k)
# lmpnn.fit(X_train, y_train)
# print()
# print("LMPNN")
# print(lmpnn.predict(X_test))
# print(lmpnn.score(X_test, y_test))
# lmpnn.showScoreInGraph(k_range, X_test, y_test)

# # Get shap values
# shap.initjs()
# explainer = shap.KernelExplainer(lmpnn.predict, numpy.array(X_test))
# svc_shap_values = explainer.shap_values(numpy.array(X_test), nsamples=100)
# shap.summary_plot(svc_shap_values, numpy.array(X_test))

# MVMCNN
mvmcnn = MVMCNN(k_neighbors=k)
mvmcnn.fit(X_train, y_train)
print()
print("MVMCNN")
print(mvmcnn.predict(X_test))
print(mvmcnn.score(X_test, y_test))
mvmcnn.showScoreInGraph(k_range, X_test, y_test)

# Get shap values
shap.initjs()
explainer = shap.KernelExplainer(mvmcnn.predict, numpy.array(X_test))
svc_shap_values = explainer.shap_values(numpy.array(X_test), nsamples=100)
shap.summary_plot(svc_shap_values, numpy.array(X_test))



# note: F1 score, prec, recall harus dihitung
# bikin average accuracy, F1 score, prec, recall untuk k yang sudah dihitung


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