import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Chargement du modèle et des données (si non déjà fait)
import joblib
kmeans = joblib.load('models/kmeans_model.pkl')
preprocessor = joblib.load('models/obesity_preprocessor.pkl')

df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
X_preprocessed = preprocessor.transform(X)
cluster_labels = kmeans.labels_

# 1. Évaluation avec le score de silhouette
silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
print(f"Score de silhouette: {silhouette_avg:.4f}")
# Interprétation: -1 (pire) à 1 (meilleur), valeurs > 0.5 indiquent une bonne séparation

# 2. Évaluation avec l'indice de Calinski-Harabasz
ch_score = calinski_harabasz_score(X_preprocessed, cluster_labels)
print(f"Indice de Calinski-Harabasz: {ch_score:.4f}")
# Interprétation: plus la valeur est élevée, meilleure est la séparation des clusters

# 3. Évaluation avec l'indice de Davies-Bouldin
db_score = davies_bouldin_score(X_preprocessed, cluster_labels)
print(f"Indice de Davies-Bouldin: {db_score:.4f}")
# Interprétation: 0 (meilleur) à ∞, valeurs plus basses indiquent une meilleure séparation

# 4. Visualisation du score de silhouette par cluster
from sklearn.metrics import silhouette_samples

# Calcul des scores de silhouette pour chaque échantillon
silhouette_values = silhouette_samples(X_preprocessed, cluster_labels)

# Création d'un DataFrame avec les scores de silhouette et les labels de cluster
silhouette_df = pd.DataFrame({
    'cluster': cluster_labels,
    'silhouette_score': silhouette_values
})

# Visualisation des distributions de scores de silhouette par cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='silhouette_score', data=silhouette_df)
plt.axhline(y=silhouette_avg, color='red', linestyle='--', label=f'Score moyen: {silhouette_avg:.4f}')
plt.title('Distribution des scores de silhouette par cluster')
plt.xlabel('Cluster')
plt.ylabel('Score de silhouette')
plt.legend()
plt.grid(True)
plt.show()

# 5. Comparaison des scores de silhouette pour différentes valeurs de k
k_range = range(2, 11)
silhouette_scores = []
ch_scores = []
db_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels_temp = kmeans_temp.fit_predict(X_preprocessed)
    
    # Calcul des métriques
    silhouette_scores.append(silhouette_score(X_preprocessed, cluster_labels_temp))
    ch_scores.append(calinski_harabasz_score(X_preprocessed, cluster_labels_temp))
    db_scores.append(davies_bouldin_score(X_preprocessed, cluster_labels_temp))

# Visualisation des métriques
plt.figure(figsize=(15, 5))

# Score de silhouette
plt.subplot(1, 3, 1)
plt.plot(k_range, silhouette_scores, 'bo-')
plt.title('Score de silhouette')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score')
plt.grid(True)

# Indice de Calinski-Harabasz
plt.subplot(1, 3, 2)
plt.plot(k_range, ch_scores, 'go-')
plt.title('Indice de Calinski-Harabasz')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score')
plt.grid(True)

# Indice de Davies-Bouldin
plt.subplot(1, 3, 3)
plt.plot(k_range, db_scores, 'ro-')
plt.title('Indice de Davies-Bouldin')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Radar chart pour visualiser les caractéristiques des clusters
# Sélection des caractéristiques numériques pour le radar chart
#numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
numeric_features = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE']

# Calcul des moyennes normalisées pour chaque cluster
cluster_means = {}
scaler = StandardScaler()
df_numeric = df[numeric_features]
df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_features)
df_numeric_scaled['cluster'] = cluster_labels

for i in range(kmeans.n_clusters):
    cluster_means[i] = df_numeric_scaled[df_numeric_scaled['cluster'] == i][numeric_features].mean()

# Création du radar chart
from math import pi

# Nombre de variables
categories = numeric_features
N = len(categories)

# Calcul des angles pour chaque variable
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Pour fermer le polygone

# Initialisation de la figure
plt.figure(figsize=(12, 8))
ax = plt.subplot(111, polar=True)

# Couleurs pour chaque cluster
colors = plt.cm.viridis(np.linspace(0, 1, kmeans.n_clusters))

# Tracé pour chaque cluster
for i in range(kmeans.n_clusters):
    values = cluster_means[i].values.tolist()
    values += values[:1]  # Pour fermer le polygone
    
    # Tracé des valeurs
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}', color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# Ajout des étiquettes
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.title("Profil des clusters (caractéristiques normalisées)", size=15)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

# 7. Conclusion sur la qualité du clustering
print("\nConclusion sur la qualité du clustering:")
print(f"Score de silhouette: {silhouette_avg:.4f} - ", end="")
if silhouette_avg < 0.25:
    print("Clustering de faible qualité")
elif silhouette_avg < 0.5:
    print("Clustering de qualité moyenne")
elif silhouette_avg < 0.75:
    print("Bon clustering")
else:
    print("Excellent clustering")

print(f"Indice de Calinski-Harabasz: {ch_score:.4f}")
print(f"Indice de Davies-Bouldin: {db_score:.4f}")

# Interprétation basée sur toutes les métriques
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
best_k_ch = k_range[np.argmax(ch_scores)]
best_k_db = k_range[np.argmin(db_scores)]

print(f"\nNombre optimal de clusters selon:")
print(f"- Score de silhouette: {best_k_silhouette}")
print(f"- Indice de Calinski-Harabasz: {best_k_ch}")
print(f"- Indice de Davies-Bouldin: {best_k_db}")