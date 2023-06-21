from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("..")
os.chdir("..")
# print(os.listdir())
D = pd.read_csv("dataset.csv")

documents = D["Text"]



vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

clusters = kmeans.labels_
pca = PCA(n_components=2, random_state=42)
pca_vecs = pca.fit_transform(X.toarray())
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

plt.figure(figsize=(12, 7))
plt.title("TF-IDF + KMeans Legal Class", fontdict={"fontsize": 18})

plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})

sns.scatterplot(data=documents, x=x0, y=x1, hue=clusters, palette="viridis")
plt.show()



# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     print

# print("\n")
# print("Prediction")

# Y = vectorizer.transform(["chrome browser to open."])
# prediction = model.predict(Y)
# print(prediction)

# Y = vectorizer.transform(["My cat is hungry."])
# prediction = model.predict(Y)
# print(prediction)

