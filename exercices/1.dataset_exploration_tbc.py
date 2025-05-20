import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset
df = pd.read_csv('data/obesity_data.csv', sep=';')

# Aperçu des premières lignes
print("Aperçu des données:")
print(df.head())

# Informations sur le dataset
print("\nInformations sur le dataset:")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# À compléter:
# 1. Explorer la distribution de la variable cible 'NObeyesdad'
# 2. Examiner les distributions des variables catégorielles
# 3. Créer des visualisations pour mieux comprendre les données:
#    - Distribution par genre
#    - Histogramme de l'âge
#    - Relations entre variables numériques
#    - Matrice de corrélation
# 4. Identifier les variables pertinentes pour le clustering