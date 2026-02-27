import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from statistics import mean

# ==============================
# Lecture des données
# ==============================
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Création de la cible (Survived) et suppression de la colonne 'Survived' dans train_data
y_train = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)

# Fusion des données d'entraînement et de test pour prétraitement
data = pd.concat([train_data, test_data], sort=False)

# ==============================
# Traitement des variables catégorielles
# ==============================
categorical_features = ["Embarked", "Sex", "Cabin"]
data_encoded = pd.get_dummies(data[categorical_features])

# Ajout des features numériques
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch']
data_encoded = pd.concat([data_encoded, data[numeric_features]], axis=1)

# Imputation des valeurs manquantes pour Age avec la médiane
imputer = SimpleImputer(strategy="median")
data_encoded[['Age']] = imputer.fit_transform(data_encoded[['Age']])

# ==============================
# Feature engineering
# ==============================
# Taille de la famille
data_encoded['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# Personne seule ?
data_encoded['IsAlone'] = (data_encoded['FamilySize'] == 1).astype(int)
# Présence de cabine
data_encoded['HasCabin'] = data['Cabin'].notnull().astype(int)

# Remplissage des autres valeurs manquantes par 0
data_encoded = data_encoded.fillna(0)

# ==============================
# Séparation train/test
# ==============================
X_train = data_encoded[:len(train_data)]
X_test = data_encoded[len(train_data):]

# ==============================
# Modèle Random Forest
# ==============================
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=3)

# Validation croisée
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Mean cross-validation score:", mean(cv_scores))

# Entraînement sur l'ensemble du training set
model.fit(X_train, y_train)

# Prédictions sur le test set
predictions = model.predict(X_test)

# Création du fichier de soumission
submission = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

print("Fichier 'submission.csv' créé avec succès !")
