import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# On suppose que nous avons déjà les modèles entraînés et les labels pour chaque modèle
# Organisés dans un dictionnaire 'models'

# Évaluation avec des métriques internes
print("Évaluation avec des métriques internes:")
print("\nScore de silhouette (plus élevé est meilleur):")
for name, labels in models.items():
    # Pour DBSCAN, gérer séparément les points de bruit
    if name == 'DBSCAN' and -1 in labels:
        valid_indices = labels != -1
        if len(set(labels[valid_indices])) >= 2:
            score = silhouette_score(X_preprocessed[valid_indices], labels[valid_indices])
            print(f"{name}: {score:.4f} (excluant les points de bruit)")
        else:
            print(f"{name}: Non applicable (moins de 2 clusters sans bruit)")
    else:
        try:
            score = silhouette_score(X_preprocessed, labels)
            print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"{name}: Erreur - {e}")

# À compléter:
# 1. Calculer et afficher l'indice de Calinski-Harabasz pour chaque modèle
# 2. Calculer et afficher l'indice de Davies-Bouldin pour chaque modèle
# 3. Calculer les métriques externes (ARI et AMI) en comparant avec NObeyesdad
# 4. Visualiser les clusters avec PCA et t-SNE
# 5. Analyser la distribution des tailles de clusters pour chaque modèle
# 6. Créer un tableau récapitulatif des performances
# 7. Évaluer la pureté des clusters par rapport à NObeyesdad
# 8. Déterminer le modèle le plus performant selon l'ensemble des métriques