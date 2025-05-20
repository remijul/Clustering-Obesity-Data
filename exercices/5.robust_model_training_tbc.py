import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# On suppose que nous avons déjà les données prétraitées X_preprocessed
# et le nombre optimal de clusters optimal_k

# 1. Clustering hiérarchique agglomératif
print("Entrainement du modèle de clustering hiérarchique agglomératif...")
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_preprocessed)

# À compléter:
# 1. Rechercher la valeur optimale pour eps dans DBSCAN en utilisant le graphique k-distance
# 2. Entraîner le modèle DBSCAN avec les paramètres appropriés
# 3. Entraîner un modèle de mélange gaussien (GMM)
# 4. Entraîner un modèle de clustering spectral
# 5. Évaluer les performances de chaque modèle avec le score de silhouette
# 6. Visualiser les résultats avec PCA pour tous les modèles
# 7. Sauvegarder les modèles