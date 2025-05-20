# 🧠 Atelier Clustering : Analyse des Niveaux d'Obésité

![Banner](https://img.shields.io/badge/niveau-intermédiaire-yellow) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)

Un atelier pratique sur le clustering non supervisé appliqué à l'analyse des habitudes alimentaires et de la condition physique en lien avec l'obésité.

## 📋 Présentation

Bienvenue dans cet atelier de clustering centré sur l'analyse des données d'obésité ! Ce projet vous guide à travers l'application d'algorithmes de clustering pour découvrir des relations entre les habitudes alimentaires, l'activité physique et les niveaux d'obésité.

À travers une série d'exercices guidés, vous apprendrez à :
- Explorer et préparer les données pour le clustering
- Créer un pipeline de prétraitement avec scikit-learn
- Appliquer et évaluer différents algorithmes de clustering
- Interpréter les résultats et donner du sens aux clusters identifiés

## 🗂️ Structure du Projet

```
obesity-clustering-workshop/
│
├── data/                       # Dataset d'obésité
│
├── exercices/                  # Notebooks des exercices à compléter
│   ├── 1_dataset_exploration.py
│   ├── 2_pipeline_generation.py
│   ├── 3_kmeans_training.py
│   ├── 4_kmeans_evaluation.py
│   ├── 5_robust_model_clustering.py
│   ├── 6_robust_model_evaluation.py
│   └── 7_model_selection.py
│
├── models/                     # Dossier pour sauvegarder les modèles entraînés
│
├── solutions/                  # Solutions des exercices (à consulter après avoir essayé)
│
├── explanations_on_metrics.md  # Guide détaillé sur les métriques d'évaluation
├── explanations_on_models.md   # Explications sur les différents modèles de clustering
├── faq.md                      # Questions fréquemment posées
└── requirements.txt            # Dépendances Python
```

## 📊 Dataset

Le dataset utilisé provient de l'UCI Machine Learning Repository et contient des données sur l'estimation des niveaux d'obésité basée sur les habitudes alimentaires et la condition physique d'individus du Mexique, du Pérou et de la Colombie.

**Variables incluses** :
- Données démographiques : Genre, Âge, Taille, Poids
- Antécédents familiaux d'obésité
- Habitudes alimentaires : fréquence de consommation d'aliments à haute teneur calorique, de légumes, nombre de repas quotidiens, consommation entre les repas
- Activité physique : fréquence d'activité physique, temps passé devant des écrans
- Habitudes de vie : consommation d'eau, tabagisme, moyen de transport

La variable cible `NObeyesdad` indique la catégorie d'obésité, allant de "Poids insuffisant" à "Obésité de type III".

## 🚀 Pour commencer

### Prérequis

- Python 3.8 ou supérieur
- Connaissance de base de pandas, numpy et scikit-learn
- Familiarité avec Jupyter Notebook

### Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/username/obesity-clustering-workshop.git
cd obesity-clustering-workshop
```

2. Créez et activez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancez Jupyter Notebook :
```bash
jupyter notebook
```

5. Commencez par le premier notebook dans le dossier `exercices/` !

## 📝 Description des Exercices

### 1. Exploration des données
Analyse exploratoire du dataset pour comprendre la distribution des variables et identifier celles qui sont pertinentes pour le clustering.

### 2. Prétraitement des données
Création d'un pipeline scikit-learn pour standardiser, encoder et préparer les données pour le clustering.

### 3. Clustering avec K-means
Application de l'algorithme K-means et détermination du nombre optimal de clusters.

### 4. Évaluation des performances
Utilisation de métriques internes comme le score de silhouette, l'indice de Calinski-Harabasz et l'indice de Davies-Bouldin pour évaluer la qualité du clustering.

### 5. Algorithmes avancés
Expérimentation avec des algorithmes plus sophistiqués : Clustering hiérarchique, DBSCAN, Modèles de mélange gaussien et Clustering spectral.

### 6. Sélection du modèle
Comparaison des différentes méthodes de clustering et sélection du modèle optimal en fonction des métriques d'évaluation.

### 7. Interprétation des résultats
Analyse approfondie des caractéristiques de chaque cluster et formulation de recommandations basées sur les profils identifiés.

## 📚 Documentation Supplémentaire

- `explanations_on_metrics.md` : Guide détaillé sur les différentes métriques d'évaluation du clustering, leur formule, interprétation et cas d'utilisation.
- `explanations_on_models.md` : Explications théoriques et pratiques sur les algorithmes de clustering utilisés, leurs avantages, inconvénients et paramètres importants.
- `faq.md` : Réponses aux questions fréquemment posées et solutions aux problèmes courants.

## 🎓 Objectifs pédagogiques

À la fin de cet atelier, vous serez capable de :
- Préparer efficacement des données pour le clustering
- Choisir et paramétrer des algorithmes de clustering adaptés à différents types de données
- Évaluer la qualité des clusters avec des métriques appropriées
- Interpréter les résultats du clustering dans un contexte métier
- Traduire les insights obtenus en recommandations actionnables

## 🔍 Points forts de l'atelier

- **Approche pratique** : Exercices concrets et guidés avec des données réelles
- **Multiple algorithmes** : Comparaison de 5 algorithmes de clustering différents
- **Focus sur l'interprétation** : Aller au-delà des métriques pour donner du sens aux résultats
- **Perspective métier** : Application dans le domaine de la santé et de la prévention

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à soumettre des pull requests pour améliorer les exercices, corriger des bugs ou ajouter de nouvelles fonctionnalités.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- UCI Machine Learning Repository pour le dataset
- Tous les contributeurs qui ont participé à l'amélioration de cet atelier

---

Créé avec ❤️ pour les passionnés de data science et de santé publique.

Happy clustering! 🧩✨