import pandas as pd
import numpy as np
import pickle
import os 
import logging
import joblib
import warnings

from src.preprocessing import encode_categorical, scale_features

BASE_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path=os.path.join(BASE_DIR,'..','models','banking_crisis_model.pkl'),
                 scaler_path=os.path.join(BASE_DIR,'..','models','scaler.pkl'),
                 selected_features_path=os.path.join(BASE_DIR,'..','models','selected_features.pkl')):
        
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = self._load_pickle(model_path)
            self.scaler = self._load_pickle(scaler_path)
            self.selected_features = self._load_pickle(selected_features_path)

        if all([self.model, self.scaler, self.selected_features]):
            logger.info(f'Prediction service initialized with {len(self.selected_features)} features')
        else:
            logger.error('Failed to initialize one or more components')

    def _load_pickle(self,path):
        """Load a pickled object with multiple fallback options"""
        try:
            # Ensure numpy is properly initialized
            import numpy as np
            
            methods = [
                ('joblib', lambda p: joblib.load(p)),
                ('pickle', lambda p: pickle.load(open(p, 'rb'))),
                ('joblib_memory_map', lambda p: joblib.load(p, mmap_mode='r')),
            ]
            
            last_error = None
            for method_name, loader in methods:
                try:
                    logger.info(f"Attempting to load {path} using {method_name}")
                    return loader(path)
                except Exception as e:
                    last_error = e
                    logger.warning(f"{method_name} load failed for {path}: {str(e)}")
                    continue
            
            error_msg = f'All loading methods failed for {path}. Last error: {str(last_error)}'
            logger.error(error_msg)
            return None
            
        except Exception as e:
            error_msg = f'Error in load_pickle for {path}: {str(e)}'
            logger.error(error_msg)
            return None

    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Encode categorical
            df_encoded = encode_categorical(df)
            
            # Scale numerical features
            numerical_columns = ['exch_usd', 'gdp_weighted_default', 'inflation_annual_cpi']
            df_scaled = df_encoded.copy()
            
            # Get columns that are both in the dataframe and need scaling
            columns_to_scale = [col for col in numerical_columns if col in df_scaled.columns]
            if columns_to_scale:
                # Use the scaler fitted on training data
                df_scaled[columns_to_scale] = self.scaler.transform(df_scaled[columns_to_scale])
            
            # Select only the features used by the model
            # Handle the case where some features might be missing
            missing_features = [f for f in self.selected_features if f not in df_scaled.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    df_scaled[feature] = 0  # Add missing features with default value
            
            # Ensure all required features are present and in the correct order
            result = pd.DataFrame()
            for feature in self.selected_features:
                if feature in df_scaled.columns:
                    result[feature] = df_scaled[feature]
                else:
                    result[feature] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise ValueError(f"Error preprocessing input data: {str(e)}")
    
    def predict(self, data):
        """Make prediction for a single data point or batch"""
        # Preprocess input data
        X = self.preprocess_input(data)
        
        # Make prediction
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Format results
        results = []
        for i in range(len(y_pred)):
            results.append({
                'prediction': int(y_pred[i]),
                'probability': float(y_prob[i]),
                'class': 'crisis' if y_pred[i] == 1 else 'no_crisis'
            })
        
        return results[0] if len(results) == 1 else results
    
    def batch_predict(self, data_list):
        """Make predictions for a batch of data points"""
        if not isinstance(data_list, list):
            raise ValueError("Data must be a list of dictionaries")
        
        df = pd.DataFrame(data_list)
        return self.predict(df)

def save_prediction_artifacts(model, scaler, selected_features, output_dir=os.path.join(BASE_DIR,'..','models')):
    """Save model and preprocessing artifacts for prediction"""
    try:
        import joblib
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model using both joblib and pickle for redundancy
        model_path = os.path.join(output_dir, 'banking_crisis_model.pkl')
        joblib.dump(model, model_path, compress=3)
        with open(model_path + '.backup', 'wb') as f:
            pickle.dump(model, f, protocol=4)  # Use protocol 4 for better compatibility
        
        # Save scaler using joblib
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path, compress=3)
        
        # Save selected features using joblib
        features_path = os.path.join(output_dir, 'selected_features.pkl')
        joblib.dump(selected_features, features_path, compress=3)
        
        logger.info(f"Prediction artifacts saved to {output_dir}")
        logger.info(f"Model saved with backup copy at {model_path}.backup")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path
        }
    except Exception as e:
        error_msg = f"Error saving prediction artifacts: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

if __name__ == "__main__":
    # Test prediction
    from src.preprocessing import load_data, preprocess_pipeline
    from src.model import train_model, train_test_data_split
    
    # Load and preprocess data
    path = "data/african_crises2.csv"
    dataset = load_data(path)
    X, y, scaler, selected_features = preprocess_pipeline(dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_data_split(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save prediction artifacts
    artifacts = save_prediction_artifacts(model, scaler, selected_features)
    
    # Create prediction service
    service = PredictionService(
        model_path=artifacts['model_path'],
        scaler_path=artifacts['scaler_path'],
        selected_features_path=artifacts['features_path']
    )
    
    # Test prediction on sample data
    sample_data = X_test.iloc[0].to_dict()
    prediction = service.predict(sample_data)
    print(f"Sample prediction: {prediction}")
