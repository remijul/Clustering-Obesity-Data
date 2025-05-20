import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# On suppose que nous avons déjà le modèle kmeans et les données prétraitées

# Évaluation avec le score de silhouette
silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
print(f"Score de silhouette: {silhouette_avg:.4f}")

# Évaluation avec l'indice de Calinski-Harabasz
ch_score = calinski_harabasz_score(X_preprocessed, cluster_labels)
print(f"Indice de Calinski-Harabasz: {ch_score:.4f}")

# Évaluation avec l'indice de Davies-Bouldin
db_score = davies_bouldin_score(X_preprocessed, cluster_labels)
print(f"Indice de Davies-Bouldin: {db_score:.4f}")

# À compléter:
# 1. Calculer les scores de silhouette pour chaque échantillon
# 2. Visualiser la distribution des scores de silhouette par cluster
# 3. Comparer les métriques pour différentes valeurs de k
# 4. Créer un radar chart pour visualiser les caractéristiques des clusters
# 5. Interpréter les résultats et tirer des conclusions sur la qualité du clustering