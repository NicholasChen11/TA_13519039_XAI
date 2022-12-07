import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X= np.random.rand(50,2)
Y= 2 + np.random.rand(50,2)
Z= np.concatenate((X,Y))
Z= pd.DataFrame(Z) #converting into data frame for ease

KMean= KMeans(n_clusters=2)
KMean.fit(Z)
label=KMean.predict(Z)

sns.scatterplot(Z, x=0, y=1, hue=label)
plt.show()
print(len(Z))
print(len(label))
print(f'Silhouette Score(n=2): {silhouette_score(Z, label)}')