import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)","Spending Score (1-100)"]]
wcss_list = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i,init = "k-means++",random_state = 42)
  kmeans.fit(X)
  wcss_list.append(kmeans.inertia_)

plt.plot(range(1,11),wcss_list)
plt.title("The Elbow Method Graph")
plt.xlabel("NUmber of Clusters (K)")
plt.ylabel("WCSS_List")
plt.show()

# Training the K-Means model on a dataset
kmeans = KMeans(n_clusters=5, init = 'k-means++', random_state = 42)
y_predict = kmeans.fit_predict(X)

## Visualizing the Clusters
X_array = X.values #Converting dataframe to Numpy Array
# for cluster 1
plt.scatter(X_array[y_predict == 0,0],X_array[y_predict == 0,1], s =100, c = "Green",label = "Cluster 1")
# for cluster 2
plt.scatter(X_array[y_predict == 1,0], X_array[y_predict == 1,1], s = 100, c = "Red", label = "Cluster 2")
# for cluster 3
plt.scatter(X_array[y_predict == 2,0],X_array[y_predict == 2,1], s = 100, c = "Blue", label = "Cluster 3")
#for cluster 4
plt.scatter(X_array[y_predict == 3,0], X_array[y_predict == 3,1], s = 100, c = "Yellow", label = "Cluster 4")
#for cluster 5
plt.scatter(X_array[y_predict == 4,0], X_array[y_predict == 4,1], s = 100, c = "Pink", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'Cyan',
            label = "Centroid")

plt.title("Clusters of Customers")
plt.xlabel("Annual Income(K$)")
plt.ylabel("SpendingScore(1-100)")
plt.legend()
plt.show()
# Save model
joblib.dump(kmeans, "cluster_model.pkl")

print("✅ Model saved as 'cluster_model.pkl'")
