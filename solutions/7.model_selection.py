import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder

# Chargement des données prétraitées (supposons que nous avons déjà X_preprocessed)
# Si ce n'est pas le cas, on peut charger et prétraiter les données
import joblib
preprocessor = joblib.load('models/obesity_preprocessor.pkl')

df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
X_preprocessed = preprocessor.fit_transform(X)

# Récupération des résultats d'évaluation
# On assume que le DataFrame 'performance_df' créé dans la tâche précédente est disponible
performance_df = pd.read_csv("data/performance_df.csv")

# Sinon, on le recrée ici
if 'performance_df' not in locals():
    # Code pour recréer performance_df...
    pass

# 1. Classement des modèles selon chaque métrique
print("Classement des modèles selon chaque métrique:")

# Classement par silhouette (plus élevé est meilleur)
silhouette_ranking = performance_df.sort_values('Silhouette', ascending=False)
print("\nClassement par score de silhouette:")
print(silhouette_ranking[['Model', 'Silhouette']])

# Classement par Calinski-Harabasz (plus élevé est meilleur)
ch_ranking = performance_df.sort_values('Calinski-Harabasz', ascending=False)
print("\nClassement par indice de Calinski-Harabasz:")
print(ch_ranking[['Model', 'Calinski-Harabasz']])

# Classement par Davies-Bouldin (plus bas est meilleur)
db_ranking = performance_df.sort_values('Davies-Bouldin', ascending=True)
print("\nClassement par indice de Davies-Bouldin:")
print(db_ranking[['Model', 'Davies-Bouldin']])

# Classement par ARI (plus élevé est meilleur)
ari_ranking = performance_df.sort_values('ARI', ascending=False)
print("\nClassement par Adjusted Rand Index:")
print(ari_ranking[['Model', 'ARI']])

# Classement par AMI (plus élevé est meilleur)
ami_ranking = performance_df.sort_values('AMI', ascending=False)
print("\nClassement par Adjusted Mutual Information:")
print(ami_ranking[['Model', 'AMI']])

# 2. Création d'un score composite pour chaque modèle
# Normalisation des scores pour Davies-Bouldin (conversion pour que plus élevé soit meilleur)
performance_df['DB_normalized'] = 1 / (1 + performance_df['Davies-Bouldin'])

# Normalisation min-max des scores pour les avoir tous entre 0 et 1
for metric in ['Silhouette', 'Calinski-Harabasz', 'DB_normalized', 'ARI', 'AMI']:
    if performance_df[metric].notna().any():  # Vérifier qu'il y a au moins une valeur non-NA
        min_val = performance_df[metric].min()
        max_val = performance_df[metric].max()
        if max_val > min_val:  # Éviter la division par zéro
            performance_df[f'{metric}_norm'] = (performance_df[metric] - min_val) / (max_val - min_val)
        else:
            performance_df[f'{metric}_norm'] = 1
    else:
        performance_df[f'{metric}_norm'] = 0

# Calcul du score composite (moyenne des scores normalisés)
performance_df['Composite_Score'] = performance_df[['Silhouette_norm', 'Calinski-Harabasz_norm', 
                                                   'DB_normalized_norm', 'ARI_norm', 'AMI_norm']].mean(axis=1)

# Classement final selon le score composite
final_ranking = performance_df.sort_values('Composite_Score', ascending=False)
print("\nClassement final selon le score composite:")
print(final_ranking[['Model', 'Composite_Score', 'Silhouette', 'Calinski-Harabasz', 
                     'Davies-Bouldin', 'ARI', 'AMI']])

# 3. Sélection du meilleur modèle
best_model_name = final_ranking.iloc[0]['Model']
print(f"\nLe meilleur modèle selon le score composite est: {best_model_name}")

# 4. Visualisation des scores composites
plt.figure(figsize=(12, 6))
bars = plt.bar(performance_df['Model'], performance_df['Composite_Score'], color='skyblue')
plt.title('Score composite par modèle')
plt.xlabel('Modèle')
plt.ylabel('Score composite')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Ajout des valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Visualisation radar des performances pour chaque modèle
# Définition des métriques pour le radar chart
metrics = ['Silhouette_norm', 'Calinski-Harabasz_norm', 'DB_normalized_norm', 'ARI_norm', 'AMI_norm']
metric_labels = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin\n(inversé)', 'ARI', 'AMI']

# Création du radar chart
from math import pi

# Nombre de variables
N = len(metrics)

# Calcul des angles pour chaque métrique
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Pour fermer le polygone

# Initialisation de la figure
plt.figure(figsize=(15, 10))

# Couleurs pour chaque modèle
colors = plt.cm.viridis(np.linspace(0, 1, len(performance_df)))

# Sous-figures pour chaque modèle
for i, (_, row) in enumerate(performance_df.iterrows()):
    ax = plt.subplot(2, 3, i+1, polar=True)
    
    # Valeurs pour le modèle courant
    values = [row[m] for m in metrics]
    values += values[:1]  # Pour fermer le polygone
    
    # Tracé des valeurs
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'], color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Ajout des étiquettes
    plt.xticks(angles[:-1], metric_labels, size=10)
    
    # Ajout du titre
    plt.title(row['Model'], size=15, color=colors[i], y=1.1)
    
    # Ajustement des limites de l'axe y
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 6. Sélection finale du modèle et explication
print("\nSélection finale du modèle:")
print(f"Le modèle choisi est: {best_model_name}")

# Explication détaillée des performances du modèle choisi
best_model_row = final_ranking.iloc[0]
print("\nPerformances détaillées:")
print(f"- Score de silhouette: {best_model_row['Silhouette']:.4f}")
print(f"- Indice de Calinski-Harabasz: {best_model_row['Calinski-Harabasz']:.4f}")
print(f"- Indice de Davies-Bouldin: {best_model_row['Davies-Bouldin']:.4f}")
print(f"- Adjusted Rand Index: {best_model_row['ARI']:.4f}")
print(f"- Adjusted Mutual Information: {best_model_row['AMI']:.4f}")
print(f"- Score composite: {best_model_row['Composite_Score']:.4f}")

# Nombre de clusters dans le modèle choisi
print(f"\nNombre de clusters: {best_model_row['Clusters']}")
if 'Noise Points' in best_model_row and best_model_row['Noise Points'] > 0:
    print(f"Nombre de points de bruit: {best_model_row['Noise Points']}")

# 7. Sauvegarde du modèle sélectionné comme modèle final
best_model = None
if best_model_name == 'K-means':
    best_model = joblib.load('models/kmeans_model.pkl')
elif best_model_name == 'Agglomerative':
    best_model = joblib.load('models/agg_clustering_model.pkl')
elif best_model_name == 'DBSCAN':
    best_model = joblib.load('models/dbscan_model.pkl')
elif best_model_name == 'GMM':
    best_model = joblib.load('models/gmm_model.pkl')
elif best_model_name == 'Spectral':
    best_model = joblib.load('models/spectral_model.pkl')

# Sauvegarde du modèle final
joblib.dump(best_model, 'models/final_model.pkl')
print(f"\nLe modèle {best_model_name} a été sauvegardé comme modèle final sous 'models/final_model.pkl'")

# 8. Sauvegarde des labels du modèle final dans le DataFrame original
# Récupération des labels du modèle choisi
if best_model_name == 'K-means':
    final_labels = best_model.labels_
elif best_model_name == 'Agglomerative':
    final_labels = best_model.labels_
elif best_model_name == 'DBSCAN':
    final_labels = best_model.fit_predict(X_preprocessed)
elif best_model_name == 'GMM':
    final_labels = best_model.predict(X_preprocessed)
elif best_model_name == 'Spectral':
    final_labels = best_model.labels_

# Ajout des labels au DataFrame original
df['cluster'] = final_labels

# Sauvegarde du DataFrame avec les clusters
df.to_csv('data/obesity_with_clusters.csv', index=False)
print("DataFrame avec les clusters sauvegardé sous 'data/obesity_with_clusters.csv'")