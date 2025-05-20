import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# On suppose que nous avons déjà les données prétraitées X_preprocessed
# Sinon, chargez-les à partir du pipeline de prétraitement

# Chargement des données prétraitées (supposons que nous avons déjà X_preprocessed)
# Si ce n'est pas le cas, on peut charger et prétraiter les données
import joblib
preprocessor = joblib.load('models/obesity_preprocessor.pkl')

df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
X_preprocessed = preprocessor.fit_transform(X)

# Recherche du nombre optimal de clusters
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_preprocessed)
    inertia.append(kmeans.inertia_)
    # Calcul du score de silhouette
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_preprocessed, labels))

# Visualisation de la méthode du coude
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.grid(True)
plt.show()

# À compléter:
# 1. Visualiser le score de silhouette
# 2. Déterminer le nombre optimal de clusters
# 3. Entraîner le modèle K-means avec le nombre optimal de clusters
# 4. Ajouter les labels de cluster au DataFrame original
# 5. Visualiser les clusters avec PCA
# 6. Analyser les caractéristiques de chaque cluster
# 7. Sauvegarder le modèle