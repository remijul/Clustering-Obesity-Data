import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# Chargement des données et des modèles
import joblib

# Chargement du dataset original
df = pd.read_csv('data/obesity_data.csv', sep=';')
X = df.drop('NObeyesdad', axis=1)
y_true = df['NObeyesdad']  # Pour l'évaluation externe si disponible

# Chargement du préprocesseur et application
preprocessor = joblib.load('models/obesity_preprocessor.pkl')
X_preprocessed = preprocessor.transform(X)

# Chargement des modèles
kmeans = joblib.load('models/kmeans_model.pkl')
agg_clustering = joblib.load('models/agg_clustering_model.pkl')
dbscan = joblib.load('models/dbscan_model.pkl')
gmm = joblib.load('models/gmm_model.pkl')
spectral = joblib.load('models/spectral_model.pkl')

# Obtention des labels pour chaque modèle
kmeans_labels = kmeans.labels_
agg_labels = agg_clustering.labels_
dbscan_labels = dbscan.fit_predict(X_preprocessed)
gmm_labels = gmm.predict(X_preprocessed)
spectral_labels = spectral.labels_

# Organisation des modèles et labels dans des dictionnaires
models = {
    'K-means': kmeans_labels,
    'Agglomerative': agg_labels,
    'DBSCAN': dbscan_labels,
    'GMM': gmm_labels,
    'Spectral': spectral_labels
}

# 1. Évaluation avec des métriques internes
print("Évaluation avec des métriques internes:")
print("\nScore de silhouette (plus élevé est meilleur):")
for name, labels in models.items():
    # Pour DBSCAN, on exclut les points de bruit (-1) du calcul
    if name == 'DBSCAN' and -1 in labels:
        valid_indices = labels != -1
        if len(set(labels[valid_indices])) >= 2:  # Au moins 2 clusters nécessaires
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

print("\nIndice de Calinski-Harabasz (plus élevé est meilleur):")
for name, labels in models.items():
    # Pour DBSCAN, on exclut les points de bruit
    if name == 'DBSCAN' and -1 in labels:
        valid_indices = labels != -1
        if len(set(labels[valid_indices])) >= 2:
            score = calinski_harabasz_score(X_preprocessed[valid_indices], labels[valid_indices])
            print(f"{name}: {score:.4f} (excluant les points de bruit)")
        else:
            print(f"{name}: Non applicable (moins de 2 clusters sans bruit)")
    else:
        try:
            score = calinski_harabasz_score(X_preprocessed, labels)
            print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"{name}: Erreur - {e}")

print("\nIndice de Davies-Bouldin (plus bas est meilleur):")
for name, labels in models.items():
    # Pour DBSCAN, on exclut les points de bruit
    if name == 'DBSCAN' and -1 in labels:
        valid_indices = labels != -1
        if len(set(labels[valid_indices])) >= 2:
            score = davies_bouldin_score(X_preprocessed[valid_indices], labels[valid_indices])
            print(f"{name}: {score:.4f} (excluant les points de bruit)")
        else:
            print(f"{name}: Non applicable (moins de 2 clusters sans bruit)")
    else:
        try:
            score = davies_bouldin_score(X_preprocessed, labels)
            print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"{name}: Erreur - {e}")

# 2. Évaluation avec des métriques externes (si une vérité terrain est disponible)
# Convertir les catégories d'obésité en identifiants numériques pour les métriques externes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true_encoded = le.fit_transform(y_true)

print("\nÉvaluation avec des métriques externes (comparaison avec NObeyesdad):")
print("\nAdjusted Rand Index (plus élevé est meilleur):")
for name, labels in models.items():
    try:
        score = adjusted_rand_score(y_true_encoded, labels)
        print(f"{name}: {score:.4f}")
    except Exception as e:
        print(f"{name}: Erreur - {e}")

print("\nAdjusted Mutual Information (plus élevé est meilleur):")
for name, labels in models.items():
    try:
        score = adjusted_mutual_info_score(y_true_encoded, labels)
        print(f"{name}: {score:.4f}")
    except Exception as e:
        print(f"{name}: Erreur - {e}")

# 3. Visualisation comparative des clusters avec PCA et t-SNE
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)

# Création d'un DataFrame pour faciliter la visualisation
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# t-SNE (peut prendre du temps sur de grands datasets)
print("\nCalcul de t-SNE en cours...")
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_preprocessed)
print(f"t-SNE calculé en {time.time() - start_time:.2f} secondes.")

# Création d'un DataFrame pour t-SNE
tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])

# Visualisation des clusters avec PCA
plt.figure(figsize=(20, 15))
for i, (name, labels) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f'PCA - {name}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    
    # Ajouter une légende pour les clusters
    unique_labels = np.unique(labels)
    handles, _ = scatter.legend_elements()
    labels_legend = [f'Cluster {l}' if l != -1 else 'Noise' for l in unique_labels]
    plt.legend(handles, labels_legend, loc="upper right")

plt.tight_layout()
plt.show()

# Visualisation des clusters avec t-SNE
plt.figure(figsize=(20, 15))
for i, (name, labels) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f't-SNE - {name}')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.grid(True)
    
    # Ajouter une légende pour les clusters
    unique_labels = np.unique(labels)
    handles, _ = scatter.legend_elements()
    labels_legend = [f'Cluster {l}' if l != -1 else 'Noise' for l in unique_labels]
    plt.legend(handles, labels_legend, loc="upper right")

plt.tight_layout()
plt.show()

# 4. Distribution des tailles de clusters pour chaque modèle
plt.figure(figsize=(15, 10))
for i, (name, labels) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    
    # Compter les occurrences de chaque label
    unique_labels = np.unique(labels)
    counts = [np.sum(labels == label) for label in unique_labels]
    
    # Créer des étiquettes pour l'axe x
    x_labels = [f'Cluster {l}' if l != -1 else 'Noise' for l in unique_labels]
    
    # Créer un barplot
    bars = plt.bar(x_labels, counts)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{count}', ha='center', va='bottom')
    
    plt.title(f'Distribution des clusters - {name}')
    plt.xlabel('Cluster')
    plt.ylabel('Nombre d\'observations')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# 5. Tableau récapitulatif des performances
print("\nRésumé des performances:")
performance_data = []

for name, labels in models.items():
    row = {'Model': name}
    
    # Nombre de clusters (excluant le bruit pour DBSCAN)
    if name == 'DBSCAN':
        row['Clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        row['Noise Points'] = np.sum(labels == -1)
    else:
        row['Clusters'] = len(set(labels))
        row['Noise Points'] = 0
    
    # Métriques internes
    try:
        if name == 'DBSCAN' and -1 in labels:
            valid_indices = labels != -1
            if len(set(labels[valid_indices])) >= 2:
                row['Silhouette'] = silhouette_score(X_preprocessed[valid_indices], labels[valid_indices])
                row['Calinski-Harabasz'] = calinski_harabasz_score(X_preprocessed[valid_indices], labels[valid_indices])
                row['Davies-Bouldin'] = davies_bouldin_score(X_preprocessed[valid_indices], labels[valid_indices])
            else:
                row['Silhouette'] = row['Calinski-Harabasz'] = row['Davies-Bouldin'] = None
        else:
            row['Silhouette'] = silhouette_score(X_preprocessed, labels)
            row['Calinski-Harabasz'] = calinski_harabasz_score(X_preprocessed, labels)
            row['Davies-Bouldin'] = davies_bouldin_score(X_preprocessed, labels)
    except Exception:
        row['Silhouette'] = row['Calinski-Harabasz'] = row['Davies-Bouldin'] = None
    
    # Métriques externes
    try:
        row['ARI'] = adjusted_rand_score(y_true_encoded, labels)
        row['AMI'] = adjusted_mutual_info_score(y_true_encoded, labels)
    except Exception:
        row['ARI'] = row['AMI'] = None
    
    performance_data.append(row)

# Création du DataFrame de performances
performance_df = pd.DataFrame(performance_data)
print(performance_df)
performance_df.to_csv("data/performance_df.csv")

# 6. Estimation de la pureté des clusters par rapport à la variable cible 'NObeyesdad'
print("\nPureté des clusters par rapport à NObeyesdad:")

for name, labels in models.items():
    print(f"\nModèle: {name}")
    unique_labels = sorted(set(labels))
    
    for cluster in unique_labels:
        if cluster == -1 and name == 'DBSCAN':
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster}"
        
        # Sélection des indices pour ce cluster
        indices = labels == cluster
        
        # Distribution de 'NObeyesdad' dans ce cluster
        distribution = df.loc[indices, 'NObeyesdad'].value_counts(normalize=True) * 100
        
        print(f"\n{cluster_name} ({np.sum(indices)} points):")
        for category, percentage in distribution.items():
            print(f"  {category}: {percentage:.2f}%")
        
        # Calcul de la catégorie majoritaire (pureté)
        majority_category = distribution.idxmax()
        majority_percentage = distribution.max()
        print(f"  Catégorie majoritaire: {majority_category} ({majority_percentage:.2f}%)")

# 7. Conclusion générale
print("\nConclusion générale sur les performances des modèles:")
# Trouver le meilleur modèle selon chaque métrique
best_silhouette = performance_df.loc[performance_df['Silhouette'].idxmax(), 'Model'] if not performance_df['Silhouette'].isna().all() else "N/A"
best_ch = performance_df.loc[performance_df['Calinski-Harabasz'].idxmax(), 'Model'] if not performance_df['Calinski-Harabasz'].isna().all() else "N/A"
best_db = performance_df.loc[performance_df['Davies-Bouldin'].idxmin(), 'Model'] if not performance_df['Davies-Bouldin'].isna().all() else "N/A"
best_ari = performance_df.loc[performance_df['ARI'].idxmax(), 'Model'] if not performance_df['ARI'].isna().all() else "N/A"
best_ami = performance_df.loc[performance_df['AMI'].idxmax(), 'Model'] if not performance_df['AMI'].isna().all() else "N/A"

print(f"Meilleur modèle selon le score de silhouette: {best_silhouette}")
print(f"Meilleur modèle selon l'indice de Calinski-Harabasz: {best_ch}")
print(f"Meilleur modèle selon l'indice de Davies-Bouldin: {best_db}")
print(f"Meilleur modèle selon l'indice de Rand ajusté: {best_ari}")
print(f"Meilleur modèle selon l'information mutuelle ajustée: {best_ami}")

# Recommandation finale basée sur l'ensemble des métriques
print("\nRecommandation finale:")
model_scores = {}
for model in performance_df['Model']:
    model_scores[model] = 0

# Comptage simple des "victoires" par métrique
model_scores[best_silhouette] += 1
model_scores[best_ch] += 1
model_scores[best_db] += 1
model_scores[best_ari] += 1
model_scores[best_ami] += 1

best_model = max(model_scores.items(), key=lambda x: x[1])[0]
print(f"Le modèle le plus performant selon l'ensemble des métriques est: {best_model}")