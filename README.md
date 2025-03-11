<img src="./Capture d’écran du 2025-03-11 16-52-04.png"/>

## Créer l'environnement virtuel avec Python 3
python3 -m venv kaggle-titanic

## Activer l'environnement
source kaggle-titanic/bin/activate

## Installer les dépendances

## Sauvegarder les dépendances dans requirements.txt
pip freeze > requirements.txt

## Lancement de jupyter
jupyter notebook --port=8891

## Ignorer les dossiers/fichiers a pas push (.gitignore)

# Titanic Survival Prediction - Random Forest Classifier

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model](#model)
- [Model Evaluation](#model-evaluation)
- [Final Submission](#final-submission)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

## Project Description
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. Using a Random Forest Classifier, we predict whether a passenger survived or not based on various features like age, sex, class, and family size. The model is trained and evaluated using cross-validation, and the final model is used to predict survival for the test dataset.

### Kaggle Score:
As of now, the best score achieved on Kaggle is **0.79665**.

## Dataset
The dataset used for this project is the Titanic dataset, which is available on Kaggle. It consists of two files:
- `train.csv` - Contains the training data with both features and the target variable (`Survived`).
- `test.csv` - Contains the test data with features only (no target variable), used for generating predictions.

## Preprocessing
The preprocessing steps are as follows:
1. **Reading the Data**: We load the training and test datasets from CSV files using `pandas`.
2. **Handling Missing Values**: 
   - We use `SimpleImputer` to fill missing values in the `Age` column with the median of that column.
   - We handle other missing values by filling them with `0`.
3. **Categorical Variables**: 
   - We use `pd.get_dummies()` to convert categorical variables like `Embarked`, `Sex`, and `Cabin` into numeric features.
4. **Feature Engineering**:
   - We create a new feature called `FamilySize` by combining the `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard) columns.
   - We create an `IsAlone` feature to identify whether the passenger is traveling alone (`FamilySize == 1`).
   - We create a `HasCabin` binary feature to indicate whether a passenger has cabin information available.

## Feature Engineering
- **Family Size**: The size of the passenger's family is derived by summing `SibSp` and `Parch` and adding `1` (for the passenger).
- **IsAlone**: This is a binary feature indicating whether a passenger is traveling alone. If `FamilySize` equals 1, then the passenger is alone.
- **HasCabin**: A binary feature to indicate whether the passenger's cabin information is missing or not.

## Model
We use the **Random Forest Classifier**, a powerful ensemble learning method based on decision trees:
- **Hyperparameters**:
  - `n_estimators = 100`: The number of trees in the forest.
  - `max_depth = 10`: The maximum depth of the trees.
  - `random_state = 3`: Ensures reproducibility.

The model is evaluated using **5-fold cross-validation** to estimate its generalization performance.

## Model Evaluation
We use `cross_val_score` to perform 5-fold cross-validation and obtain an average accuracy score. This method divides the data into 5 subsets (folds), trains the model on 4 folds, and tests it on the remaining fold, repeating the process for each fold.

The mean accuracy from cross-validation is printed to the console, allowing us to evaluate the model's performance.

```python
Mean cross-validation score:  0.8170736300295023
Kaggle website score :  0.79665
