import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
weights = np.load('user_weights.npy')
scaler = StandardScaler()
scaled_data_user = scaler.fit_transform(weights)

kmeans_u = KMeans(n_clusters=4, random_state=42)
kmeans_u.fit(scaled_data_user)
user_cluster_labels = kmeans_u.labels_
np.save('cluster_user_emb.npy',user_cluster_labels)