# Explications pour l'atelier de clustering sur l'obésité

## Modèles de clustering utilisés

### 1. K-means

**Fonctionnement** : K-means est un algorithme de partitionnement qui divise les données en K groupes en minimisant la variance intra-cluster (somme des carrés des distances entre chaque point et le centroïde de son cluster).

**Étapes principales** :
1. Initialisation aléatoire de K centroïdes
2. Assignation de chaque point au centroïde le plus proche
3. Mise à jour des centroïdes en calculant la moyenne des points assignés
4. Répétition des étapes 2 et 3 jusqu'à convergence

**Avantages** :
- Simple à comprendre et à implémenter
- Efficace sur de grands jeux de données
- Rapide en termes de temps d'exécution

**Limites** :
- Nécessite de spécifier le nombre K de clusters à l'avance
- Sensible à l'initialisation des centroïdes
- Fonctionne mieux avec des clusters de forme sphérique et de taille similaire
- Sensible aux valeurs aberrantes

**Cas d'utilisation dans notre contexte** : K-means est utilisé comme point de départ pour identifier des profils de comportements alimentaires et d'activité physique similaires.

### 2. Clustering hiérarchique agglomératif

**Fonctionnement** : Le clustering hiérarchique construit une hiérarchie de clusters en fusionnant ou divisant des groupes. La version agglomérative commence avec chaque point comme un cluster séparé, puis fusionne progressivement les clusters les plus proches.

**Étapes principales** :
1. Chaque point forme initialement son propre cluster
2. Calcul de la matrice de distances entre tous les clusters
3. Fusion des deux clusters les plus proches
4. Mise à jour de la matrice de distances
5. Répétition des étapes 3 et 4 jusqu'à n'avoir plus qu'un seul cluster

**Stratégies de liaison** :
- **Ward** (utilisée dans l'atelier) : minimise la variance intra-cluster
- **Single** : distance minimale entre points de clusters différents
- **Complete** : distance maximale entre points de clusters différents
- **Average** : distance moyenne entre tous les points de clusters différents

**Avantages** :
- Ne nécessite pas de spécifier le nombre de clusters à l'avance
- Produit un dendrogramme qui visualise la hiérarchie des clusters
- Peut découvrir des clusters de formes et tailles variées
- Déterministe (pas d'initialisation aléatoire)

**Limites** :
- Complexité calculatoire élevée (O(n²) minimum)
- Moins adapté aux très grands jeux de données
- Difficile à interpréter lorsque les données sont de grande dimension

**Cas d'utilisation dans notre contexte** : Permet d'identifier des hiérarchies de comportements liés à l'obésité, ce qui peut aider à comprendre comment certains facteurs de risque sont reliés entre eux.

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Fonctionnement** : DBSCAN groupe les points denses et peut identifier des clusters de forme arbitraire. Il identifie également les points de bruit qui n'appartiennent à aucun cluster.

**Concepts clés** :
- **ε (epsilon)** : rayon du voisinage autour d'un point
- **MinPts** : nombre minimum de points requis dans le voisinage ε pour former un point central
- **Point central** : point ayant au moins MinPts points dans son voisinage ε
- **Point de bordure** : point appartenant au voisinage d'un point central mais n'étant pas lui-même un point central
- **Point de bruit** : point qui n'est ni central ni de bordure

**Étapes principales** :
1. Pour chaque point, identifier les voisins dans le rayon ε
2. Identifier les points centraux (ayant au moins MinPts voisins)
3. Former des clusters en connectant les points centraux qui sont voisins
4. Assigner chaque point de bordure au cluster du point central dont il est voisin
5. Les points restants sont identifiés comme bruit

**Avantages** :
- Ne nécessite pas de spécifier le nombre de clusters à l'avance
- Peut découvrir des clusters de formes arbitraires
- Identifie naturellement les valeurs aberrantes (points de bruit)
- Robuste aux données bruitées

**Limites** :
- Difficile de choisir les paramètres ε et MinPts optimaux
- Moins efficace si les clusters ont des densités très différentes
- Peut avoir des difficultés avec les données de haute dimension

**Cas d'utilisation dans notre contexte** : Particulièrement utile pour identifier des sous-groupes atypiques dans les comportements liés à l'obésité et pour isoler les cas aberrants qui pourraient biaiser d'autres analyses.

### 4. Modèle de mélange gaussien (GMM)

**Fonctionnement** : GMM modélise les données comme étant générées par un mélange de distributions gaussiennes. Chaque cluster est représenté par une distribution gaussienne différente.

**Étapes principales** :
1. Initialisation des paramètres des distributions gaussiennes
2. Application de l'algorithme EM (Expectation-Maximization) :
   - Étape E : Calculer la probabilité d'appartenance de chaque point à chaque cluster
   - Étape M : Mettre à jour les paramètres des gaussiennes (moyennes, covariances, poids)
3. Répétition jusqu'à convergence
4. Assignation de chaque point au cluster le plus probable

**Avantages** :
- Fournit des probabilités d'appartenance aux clusters (clustering "soft")
- Peut détecter des clusters de formes ellipsoïdales et de tailles variées
- Plus flexible que K-means
- Modèle probabiliste bien fondé mathématiquement

**Limites** :
- Nécessite de spécifier le nombre de composantes (clusters) à l'avance
- Sensible à l'initialisation
- Peut converger vers des maxima locaux de la vraisemblance
- Fait l'hypothèse que les clusters suivent des distributions gaussiennes

**Cas d'utilisation dans notre contexte** : Adapté pour notre dataset car les variables physiologiques et comportementales suivent souvent des distributions approximativement normales dans des sous-populations, et GMM permet de capturer les incertitudes d'appartenance.

### 5. Clustering spectral

**Fonctionnement** : Le clustering spectral utilise les vecteurs propres de la matrice de similarité des données pour réduire la dimensionnalité avant d'appliquer un algorithme de clustering simple (généralement K-means).

**Étapes principales** :
1. Construction d'une matrice de similarité entre points (souvent basée sur le noyau gaussien)
2. Calcul du laplacien normalisé de cette matrice
3. Extraction des K premiers vecteurs propres du laplacien
4. Formation d'une matrice où chaque ligne est un point dans ce nouvel espace de dimension K
5. Application de K-means sur cette matrice

**Avantages** :
- Peut identifier des clusters de formes complexes et non linéairement séparables
- Robuste aux données bruitées et aux valeurs aberrantes
- Efficace pour capturer la structure globale des données
- Basé sur une théorie mathématique solide (théorie spectrale des graphes)

**Limites** :
- Nécessite de spécifier le nombre de clusters à l'avance
- Calcul des vecteurs propres coûteux pour de grands jeux de données
- Le choix de la fonction de similarité peut être délicat
- Peut nécessiter un réglage fin des paramètres

**Cas d'utilisation dans notre contexte** : Utile pour détecter des relations non linéaires entre les variables d'obésité, alimentaires et d'activité physique, qui pourraient ne pas être capturées par des méthodes plus simples.

## Note pédagogique importante

Dans cet atelier, nous utilisons délibérément plusieurs algorithmes et métriques pour montrer que le clustering n'est pas une science exacte et qu'il est souvent nécessaire d'essayer différentes approches. Chaque méthode a ses forces et ses faiblesses, et peut révéler des aspects différents des données.

L'objectif principal n'est pas seulement d'identifier des clusters techniquement valides, mais surtout de découvrir des groupes qui ont une signification pratique et qui peuvent informer des stratégies d'intervention ciblées contre l'obésité.