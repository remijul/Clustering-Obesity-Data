import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Chargement du dataset
df = pd.read_csv('data/obesity_data.csv', sep=';')

# Identification des colonnes par type
# Colonnes numériques
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Colonnes catégorielles à encoder avec OneHotEncoder
categorical_features_onehot = ['Gender', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Colonnes catégorielles à encoder avec OrdinalEncoder
categorical_features_ordinal = ['family_history_with_overweight']

# Création des transformateurs pour chaque type de données
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Gestion des valeurs manquantes
    ('scaler', StandardScaler())  # Standardisation
])

categorical_transformer_onehot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

categorical_transformer_ordinal = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

# Combinaison des transformateurs avec ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat_onehot', categorical_transformer_onehot, categorical_features_onehot),
        ('cat_ordinal', categorical_transformer_ordinal, categorical_features_ordinal)
    ])

# Suppression de la colonne cible pour le clustering non supervisé
# Dans un contexte de clustering, nous n'utilisons pas la variable cible
X = df.drop('NObeyesdad', axis=1)

# Application du preprocessing
X_preprocessed = preprocessor.fit_transform(X)

print(f"Forme des données après prétraitement: {X_preprocessed.shape}")
print("Le prétraitement a été appliqué avec succès!")

# Si nécessaire, on peut convertir les données transformées en DataFrame pour l'interprétation
# Mais cela nécessite de récupérer les noms des colonnes après one-hot encoding

# Affichage des premières lignes des données transformées
print("\nAperçu des données prétraitées (5 premières lignes, 10 premières features):")
print(X_preprocessed[:5, :10])

# Nous pouvons maintenant sauvegarder notre pipeline pour l'utiliser plus tard
import joblib
joblib.dump(preprocessor, 'models/obesity_preprocessor.pkl')
print("Pipeline de prétraitement sauvegardé sous 'models/obesity_preprocessor.pkl'")