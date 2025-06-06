# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\acer\Desktop\AI&ML\day8.py\baacfb12d925d58d6f59578f898a0291.csv")

# Step 2: Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Use Elbow Method to find optimal K
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.show()

# Step 5: Fit K-Means with optimal K (e.g., 5)
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Visualize Clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='rainbow')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

# Step 7: Evaluate with Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
print("Silhouette Score:", score)