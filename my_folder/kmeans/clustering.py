import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("customers.csv", usecols=["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"])
data = df.values
model = KMeans(n_clusters=7)
model.fit(data)

labels = model.labels_
centroids = model.cluster_centers_

for i in range(7):
    # Plot the points. Hint: Try using principal component analysis (PCA) to narrow down features.
    datapoints = data[np.where(labels==i)]
    plt.plot(datapoints[:,3],datapoints[:,4],'k.')
    # Plot the centroids.
    centers = plt.plot(centroids[i,3],centroids[i,4],'x')
    plt.setp(centers,markersize=20.0)
    plt.setp(centers,markeredgewidth=5.0)
    
plt.xlim([0,10000])
plt.ylim([0,15000])
plt.show()