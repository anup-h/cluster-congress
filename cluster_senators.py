import numpy as np
from sklearn.cluster import KMeans
from sklearn import  preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

parties = pd.read_csv("party_labels.csv")

labels_and_numbers = pd.read_csv("scores.csv").drop(columns = ['Unnamed: 0'])
df = labels_and_numbers.drop(columns = ["Name", "Handle"])
df = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
feature_names = df.columns
pca = PCA(n_components=2).fit(df)
df = pca.transform(df)
# print(df)
for i in range(len(pca.singular_values_)):
    df[:, i] = df[:,  i]/pca.singular_values_[i]

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
labels_and_numbers["Group"] = kmeans.labels_

plt.scatter(df[:, 0], df[:, 1], c = ["b" if i == 0 else ("r" if i==1 else "g")  for i in parties["Label"]])
plt.show()
plt.scatter(df[:, 0], df[:, 1], c = kmeans.labels_)
plt.show()


