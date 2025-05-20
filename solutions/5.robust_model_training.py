import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Chargement des données prétraitées
import joblib
preprocessor = joblib.load('models/obesity_preprocessor.pkl')

df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
X_preprocessed = preprocessor.transform(X)

# Nombre optimal de clusters déterminé précédemment
optimal_k = 7  # À ajuster selon les résultats précédents

# 1. Clustering hiérarchique agglomératif
print("Entrainement du modèle de clustering hiérarchique agglomératif...")
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_preprocessed)

# 2. DBSCAN - Density-Based Spatial Clustering of Applications with Noise
print("Entrainement du modèle DBSCAN...")
# Le paramètre eps nécessite un ajustement en fonction des données
# Recherche du meilleur eps avec k-distance graph
from sklearn.neighbors import NearestNeighbors

# Calcul des distances aux k voisins les plus proches
k = 5  # nombre de voisins à considérer
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X_preprocessed)
distances, indices = neigh.kneighbors(X_preprocessed)

# Tri des distances au k-ième voisin pour le graphique k-distance
k_distances = np.sort(distances[:, k-1])

# Visualisation du graphique k-distance pour déterminer eps
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances)
plt.xlabel('Points triés par distance')
plt.ylabel(f'Distance au {k}ème voisin le plus proche')
plt.title('Graphique k-distance pour déterminer eps')
plt.grid(True)
plt.show()

# Sélection d'une valeur eps basée sur le "coude" du graphique k-distance
# Pour cet exemple, on estime la valeur d'eps
eps_value = 0.5  # À ajuster en fonction du graphique k-distance
min_samples = 5  # Nombre minimum de points pour former un cluster dense

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_preprocessed)

# Nombre de clusters trouvés par DBSCAN (en excluant le bruit, label -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Nombre de clusters trouvés par DBSCAN: {n_clusters_dbscan}")
print(f"Nombre de points de bruit: {list(dbscan_labels).count(-1)}")

# 3. Modèle de mélange gaussien (GMM)
print("Entrainement du modèle de mélange gaussien...")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm.fit(X_preprocessed)
gmm_labels = gmm.predict(X_preprocessed)

# 4. Spectral Clustering
print("Entrainement du modèle de clustering spectral...")
spectral = SpectralClustering(n_clusters=optimal_k, random_state=42, affinity='nearest_neighbors')
spectral_labels = spectral.fit_predict(X_preprocessed)

# Évaluation des performances de chaque modèle
# Note: DBSCAN peut avoir des points de bruit (label -1) qui ne sont pas inclus dans le calcul du score de silhouette
models = {
    'K-means': KMeans(n_clusters=optimal_k, random_state=42).fit_predict(X_preprocessed),
    'Agglomerative': agg_labels,
    'Spectral': spectral_labels,
    'GMM': gmm_labels
}

# DBSCAN est traité séparément en raison de la possibilité de points de bruit
dbscan_no_noise = X_preprocessed[dbscan_labels != -1]
dbscan_labels_no_noise = dbscan_labels[dbscan_labels != -1]

# Calcul et affichage des scores de silhouette pour chaque modèle
print("\nScores de silhouette:")
for name, labels in models.items():
    score = silhouette_score(X_preprocessed, labels)
    print(f"{name}: {score:.4f}")

# Score de silhouette pour DBSCAN (si au moins 2 clusters ont été trouvés, en excluant le bruit)
if n_clusters_dbscan >= 2:
    dbscan_score = silhouette_score(dbscan_no_noise, dbscan_labels_no_noise)
    print(f"DBSCAN (sans points de bruit): {dbscan_score:.4f}")
else:
    print("DBSCAN: Non applicable (moins de 2 clusters)")

# Visualisation des résultats avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)

# Création de subplots pour comparer les résultats
plt.figure(figsize=(20, 15))

# 1. K-means
plt.subplot(2, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=models['K-means'], cmap='viridis', s=50, alpha=0.8)
plt.title('K-means Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# 2. Agglomerative Clustering
plt.subplot(2, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=models['Agglomerative'], cmap='viridis', s=50, alpha=0.8)
plt.title('Agglomerative Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# 3. Spectral Clustering
plt.subplot(2, 2, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=models['Spectral'], cmap='viridis', s=50, alpha=0.8)
plt.title('Spectral Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# 4. GMM
plt.subplot(2, 2, 4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=models['GMM'], cmap='viridis', s=50, alpha=0.8)
plt.title('Gaussian Mixture Model')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualisation de DBSCAN séparément (avec distinction des points de bruit)
plt.figure(figsize=(10, 8))
unique_labels = set(dbscan_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Points de bruit en noir
        col = 'k'
    
    mask = dbscan_labels == k
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], marker='.', label=f'Cluster {k}' if k != -1 else 'Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()

# Sauvegarde des modèles
joblib.dump(agg_clustering, 'models/agg_clustering_model.pkl')
joblib.dump(dbscan, 'models/dbscan_model.pkl')
joblib.dump(gmm, 'models/gmm_model.pkl')
joblib.dump(spectral, 'models/spectral_model.pkl')

print("\nTous les modèles ont été entraînés et sauvegardés.")