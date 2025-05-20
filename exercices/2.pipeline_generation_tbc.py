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

# À compléter:
# 1. Créer des transformateurs pour chaque type de données:
#    - Pour les variables numériques: SimpleImputer + StandardScaler
#    - Pour les variables catégorielles (one-hot): SimpleImputer + OneHotEncoder
#    - Pour les variables catégorielles (ordinal): SimpleImputer + OrdinalEncoder
# 2. Combiner les transformateurs avec ColumnTransformer
# 3. Créer le jeu de données X en supprimant la variable cible
# 4. Appliquer le prétraitement et vérifier la forme des données résultantes
# 5. Sauvegarder le pipeline pour une utilisation ultérieure