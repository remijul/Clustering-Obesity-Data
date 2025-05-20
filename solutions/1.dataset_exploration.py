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

# Examen de la variable cible (si applicable pour un clustering non supervisé)
print("\nDistribution de la variable d'obésité:")
print(df['NObeyesdad'].value_counts())

# Exploration des variables catégorielles
print("\nVariables catégorielles:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nDistribution de {col}:")
    print(df[col].value_counts())

# Visualisation de quelques distributions
plt.figure(figsize=(12, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribution par genre')
plt.show()

# Histogramme de l'âge
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20)
plt.title('Distribution de l\'âge')
plt.show()

# Relations entre variables numériques
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.pairplot(df[numeric_cols])
plt.suptitle('Relations entre variables numériques', y=1.02)
plt.show()

# Matrice de corrélation
plt.figure(figsize=(14, 12))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.show()

# Identification des variables d'intérêt pour le clustering
print("\nVariables potentiellement pertinentes pour le clustering:")
print("Variables numériques: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE")
print("Variables catégorielles (à encoder): Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS")