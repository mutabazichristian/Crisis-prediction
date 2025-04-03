import numpy as numpy
import matplotlib as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def load_data(path):
    try:
        dataset = pd.read_csv(path)
        print(f"Data load with shape: {dataset.shape}")
        return dataset 
    except Exception as e:
        print(f'Error loading data: {e}')
        raise

def handle_missing_values(dataset):
    #TODO
    missing_percentage = df.isnull().mean()*100
    print(f'{missing_percentage} is missing')
    return missing_percentage

def encode_categorical(dataset):
    dataset_encoded = dataset.copy()

    if 'banking_crisis' in dataset_encoded.columns:
        dataset_encoded['banking_crisis'] = dataset_encoded['banking_crisis'].replace(['no_crisis','crisis'],[0,1])

    if 'country' in dataset_encoded.columns:
        le = LabelEncoder()
        dataset['country'] = le.fit_transform(dataset['country'])
    
        country_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
        print(f"Country mapping: {country_mapping}")

    if 'country_code' in dataset_encoded.columns:
        dataset_encoded.drop(columns=['country_code'],inplace=True)

    return dataset_encoded

def scale_features(dataset,columns=None):
    if columns is None:
        columns = ['exch_usd','gdp_weighted_default','inflation_annual_cpi']
        
    dataset_scaled = dataset.copy()
    
    columns_to_scale = [col for col in columns if col in dataset_scaled.columns]

    if columns_to_scale:
        scaler = StandardScaler()
        dataset_scaled[columns_to_scale] = scaler.fit_transform(dataset_scaled[columns_to_scale])
        print(f'Scaled columns: {columns_to_scale}')

    return dataset_scaled, scaler

def feature_selection(X,y, n_features=8):
    select  = RFE(
            estimator=RandomForestClassifier(random_state=42),
            n_features_to_select =min(n_features, X.shape[1]),
            step=1
            )
    selector = selector.fit(X,y)

    selected_features = X.columns[selector.support]
    print(f'Selected features: {selected_features}')

    return X[selected_features], selected_features

def preprocess_pipeline(dataset,target_col='banking_crisis', n_features=8):
    print('Starting preprocessing pipeline...')

    handle_missing_values(dataset)

    dataset_encoded = encode_categorical(dataset)

    if target_col in dataset_encoded.columns:
        y = dataset_encoded[target_col]
        x = dataset_encoded.drop(columns=t[target_col])
    else:
        y = None
        X = dataset_encoded

    #scale
    X_scaled, scaler = scale_features(X)

    #select features
    if y is not None:
        X_selected, selected_features = feature_selection(X_scaled,n_features)
    else:
        X_selected, selected_features = X_scaled, X_scaled.columns
    print("preprocess pipeline success!")

if __name__ == "__main__":
    path = "../data/dataset.csv"
    dataset = load_data(path)
    X,y,scaler,selected_features = preprocess_pipeline(dataset)

    print(f'Preprocessed data shape: {X.shape}')


