import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# On suppose que nous avons déjà le DataFrame 'performance_df' des performances des modèles

# Classement des modèles selon chaque métrique
print("Classement des modèles selon le score de silhouette:")
silhouette_ranking = performance_df.sort_values('Silhouette', ascending=False)
print(silhouette_ranking[['Model', 'Silhouette']])

# À compléter:
# 1. Classer les modèles selon les autres métriques (Calinski-Harabasz, Davies-Bouldin, ARI, AMI)
# 2. Créer un score composite en normalisant toutes les métriques
# 3. Classer les modèles selon le score composite
# 4. Visualiser les scores composites avec un graphique à barres
# 5. Créer un radar chart pour comparer les performances des modèles
# 6. Sélectionner le meilleur modèle et expliquer les raisons du choix
# 7. Sauvegarder le modèle sélectionné comme modèle final
# 8. Ajouter les labels du modèle final au DataFrame original et sauvegarder