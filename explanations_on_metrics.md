# Explications pour l'atelier de clustering sur l'obésité

## Métriques d'évaluation utilisées

### 1. Métriques internes (ne nécessitent pas de vérité terrain)

#### a. Score de silhouette

**Description** : Mesure à quel point un objet est similaire à son propre cluster par rapport aux autres clusters. Combine la cohésion (compacité) et la séparation.

**Calcul** : Pour chaque point i :
- a(i) = distance moyenne entre i et tous les autres points de son cluster
- b(i) = distance moyenne entre i et tous les points du cluster le plus proche (différent du sien)
- silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

La moyenne des silhouettes de tous les points donne le score de silhouette global.

**Interprétation** :
- Valeurs entre -1 et 1
- Proche de 1 : point bien classé
- Proche de 0 : point à la limite entre deux clusters
- Proche de -1 : point probablement mal classé

**Importance dans notre contexte** : Permet d'évaluer si les profils d'obésité identifiés sont bien distincts les uns des autres.

#### b. Indice de Calinski-Harabasz (indice de variance)

**Description** : Rapport entre la dispersion inter-clusters et la dispersion intra-clusters, pondéré par le nombre de clusters et d'échantillons.

**Calcul** :
CH = [trace(B) / (k-1)] / [trace(W) / (n-k)]
où :
- B est la matrice de dispersion inter-clusters
- W est la matrice de dispersion intra-clusters
- k est le nombre de clusters
- n est le nombre total de points

**Interprétation** :
- Plus la valeur est élevée, meilleure est la séparation des clusters
- Pas de limite supérieure théorique
- Utile pour comparer différentes configurations de clustering

**Importance dans notre contexte** : Aide à déterminer si les groupes d'individus identifiés sont véritablement distincts en termes de caractéristiques liées à l'obésité.

#### c. Indice de Davies-Bouldin

**Description** : Mesure la similarité moyenne entre chaque cluster et son cluster le plus similaire. La similarité est définie comme le ratio entre les dispersions intra-clusters et la distance inter-clusters.

**Calcul** :
DB = (1/k) * ∑(i=1 to k) max(j≠i) ((σi + σj) / d(ci, cj))
où :
- σi est la dispersion moyenne du cluster i
- d(ci, cj) est la distance entre les centroïdes des clusters i et j
- k est le nombre de clusters

**Interprétation** :
- Plus la valeur est faible, meilleure est la séparation des clusters
- Valeur minimale théorique de 0
- Utile pour comparer différentes configurations de clustering

**Importance dans notre contexte** : Permet d'identifier si certains groupes de comportements liés à l'obésité sont trop similaires entre eux pour être considérés comme distincts.

### 2. Métriques externes (nécessitent une vérité terrain, ici NObeyesdad)

#### a. Adjusted Rand Index (ARI)

**Description** : Mesure la similarité entre deux partitions, ajustée pour tenir compte du hasard. Compare les clusters trouvés avec les catégories connues.

**Calcul** :
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
où RI est l'Indice de Rand, calculé en comptant les paires de points qui sont:
- Dans le même cluster et la même classe (a)
- Dans des clusters différents et des classes différentes (b)
- Dans le même cluster mais des classes différentes (c)
- Dans des clusters différents mais la même classe (d)
RI = (a + b) / (a + b + c + d)

**Interprétation** :
- Valeurs entre -1 et 1
- 1 indique une correspondance parfaite
- 0 indique une correspondance due au hasard
- Valeurs négatives indiquent une correspondance inférieure au hasard

**Importance dans notre contexte** : Évalue si les clusters identifiés correspondent aux catégories d'obésité cliniquement définies, ce qui pourrait valider ou remettre en question ces classifications.

#### b. Adjusted Mutual Information (AMI)

**Description** : Mesure l'information mutuelle entre deux partitions, ajustée pour tenir compte du hasard. Quantifie la quantité d'information partagée entre les clusters et les catégories connues.

**Calcul** :
AMI = (MI - Expected_MI) / (max(MI) - Expected_MI)
où MI est l'information mutuelle entre les clusters et les catégories.

**Interprétation** :
- Valeurs entre 0 et 1
- 1 indique une correspondance parfaite
- 0 indique une correspondance due au hasard

**Importance dans notre contexte** : Indique combien d'information les clusters découverts fournissent sur les catégories d'obésité, ce qui peut aider à comprendre si des facteurs autres que l'IMC sont pertinents pour la classification.

### 3. Méthodes de détermination du nombre optimal de clusters

#### a. Méthode du coude (Elbow method)

**Description** : Examine la variance expliquée (inertie) en fonction du nombre de clusters. Le "coude" dans le graphique indique le nombre optimal de clusters.

**Interprétation** :
- Le point où l'ajout de clusters supplémentaires n'apporte plus une réduction significative de l'inertie
- Approche visuelle qui peut être subjective

**Importance dans notre contexte** : Aide à déterminer combien de profils distincts existent dans les données d'obésité sans faire d'hypothèses préalables.

#### b. Analyse du score de silhouette pour différentes valeurs de k

**Description** : Calcule le score de silhouette moyen pour différentes valeurs du nombre de clusters k.

**Interprétation** :
- Le k avec le score de silhouette moyen le plus élevé est considéré optimal
- Considère à la fois la cohésion et la séparation des clusters

**Importance dans notre contexte** : Fournit une approche plus objective pour déterminer le nombre optimal de sous-groupes dans les données d'obésité.

## Note pédagogique importante

Dans cet atelier, nous utilisons délibérément plusieurs algorithmes et métriques pour montrer que le clustering n'est pas une science exacte et qu'il est souvent nécessaire d'essayer différentes approches. Chaque méthode a ses forces et ses faiblesses, et peut révéler des aspects différents des données.

L'objectif principal n'est pas seulement d'identifier des clusters techniquement valides, mais surtout de découvrir des groupes qui ont une signification pratique et qui peuvent informer des stratégies d'intervention ciblées contre l'obésité.