import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Chargement des données prétraitées (supposons que nous avons déjà X_preprocessed)
# Si ce n'est pas le cas, on peut charger et prétraiter les données
import joblib
preprocessor = joblib.load('models/obesity_preprocessor.pkl')

df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
X_preprocessed = preprocessor.fit_transform(X)

# Identification des colonnes par type
# Colonnes numériques
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Colonnes catégorielles à encoder avec OneHotEncoder
categorical_features_onehot = ['Gender', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Colonnes catégorielles à encoder avec OrdinalEncoder
categorical_features_ordinal = ['family_history_with_overweight']

# 1. Détermination du nombre optimal de clusters avec la méthode du coude
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

# Visualisation du score de silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.title('Score de silhouette')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.grid(True)
plt.tight_layout()
plt.show()

# Selon les graphiques, choisissons un nombre optimal de clusters (par exemple, 4)
# Note : le score de silhouette indique un K optimal = 4, mais après analyse du contexte des données, 7 clusters pourraient optimal pour suivre la variable "NObeyesdad"
# voir source des données : https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
optimal_k = 7  # À ajuster en fonction des résultats

# 2. Entraînement du modèle K-means avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_preprocessed)
cluster_labels = kmeans.labels_
print("Labels", cluster_labels)

# Ajout des labels de cluster au DataFrame original
df['cluster'] = cluster_labels
print("Distribution des clusters:")
print(df['cluster'].value_counts())

# 3. Visualisation des clusters en utilisant PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)
print("Variance expliquée (ratio):")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title(f'Clusters K-means (k={optimal_k})')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 4. Analyse des caractéristiques de chaque cluster
print("\nAnalyse des clusters:")
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    cluster_data = df[df['cluster'] == i]
    print(f"Nombre d'observations: {len(cluster_data)}")
    print("\nStatistiques des variables numériques:")
    print(cluster_data[numeric_features].describe().mean())
    print("\nDistribution des variables catégorielles:")
    for col in categorical_features_onehot + categorical_features_ordinal:
        print(f"\n{col}:")
        print(cluster_data[col].value_counts(normalize=True))

# 5. Sauvegarde du modèle
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("\nModèle K-means sauvegardé sous 'models/kmeans_model.pkl'")