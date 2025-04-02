import os
import pandas as pd
import numpy as np 
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemle import RandomForestClassifier
from sklearn.metric import classification_report, roc_auc_score


def train_test_data_split(X, y, test_size=0.25, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
            )
    print(f'Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test

def tune_hyperparameters(X_train, y_train):
    print("Tuning hyperparameters")

    param_grid = {
            'n_estimators': [50,100,200]
            'max_depth': [None, 10, 20]
            'min_samples_split': [2,5,10]
            'min_samples_leaf': [1,2,4]
            }

    grid_search = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs = -1

            )

    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_:.4f}')

    return grid_search.best_params_

def evaluate_model(model, X_test, y_test):

    print('Evaluating model...')

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'Classification Report\n {classification_report(y_test,y_pred)}')
    print(f'ROC AUC Score: {roc_auc:.4f}')

    return metrics

def save_model(model,path='../models/random_forest_model.pkl')
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print(f'Model loaded from {path}')
        return model
    except Exception as e:
        print(f'Error loading model: {e}') 
        raise

def get_feature_importance(mode, feature_names):
    importances = model.feature_imoprtances_
    indices = np.argsort(importances)[::-1]

    feature_importance = pd.DataFrame({
        'Feature' : [feature_names[i] for i in indices],
        'Importance' : importances[indices]
        })

    return feature_importance

if __name__ = "__main__":
    from preprocessing import load_data, preprocess_pipeline

    #load and preprocess
    path = '../data/dataset.csv'
    dataset = load_data(path)
    X, y, scaler, selected_features = preprocess_pipeline(dataset)

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #train model
    model = train_model(X_train, y_train)
    
    #evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    #save the model
    save_model(model)


    feature_importance = get_feature_importance(model, selected_features)
    print("Feature importance:")
    print(feature_importance)
