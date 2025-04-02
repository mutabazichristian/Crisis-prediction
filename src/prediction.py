import pandas as pd
import numpy as np
import pickle
import os 

from .preprocessing import encode_categorical_variables, scale_numerical_features

class PredictionService:
    def __init__(self, model_path='../models/random_forest_model.pkl',
                 scaler_path='../models/scaler.pkl',
                 selected_features_path='../models/selected_features.pkl'):

        self.model = self._load_pickle(model_path)
        self.scaler = self._load_pickle(scaler_path)
        self.selected_features = self._load_pickle(selected_features_path)

        print(f'Prediction service initialized with {len(selected_features)} features')

    def _load_pickle(self,path):
        try:
            with open(path,'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f'Error loading {path}: {e}')
            raise

    def preprocess_input(self, data):

        if ininstance(data,dict):
        
