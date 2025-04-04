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
    def __init__(self):
        """Initialize the prediction service"""
        try:
            # Define model directory and ensure it exists
            model_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Define model paths
            self.model_path = os.path.join(model_dir, 'banking_crisis_model.pkl')
            self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
            self.features_path = os.path.join(model_dir, 'selected_features.pkl')
            
            # Initialize components
            self.model = None
            self.scaler = None
            self.selected_features = None
            self.is_ready = False
            
            # Try loading the model
            self._load_model()
            
            if not self.is_ready:
                logger.info("Attempting to retrain model...")
                self._retrain_model()
        except Exception as e:
            logger.error(f"Error initializing prediction service: {str(e)}")
            raise

    def _load_model(self):
        """Load model and its components"""
        try:
            self.model = self._load_pickle(self.model_path)
            self.scaler = self._load_pickle(self.scaler_path)
            self.selected_features = self._load_pickle(self.features_path)
            
            self.is_ready = all([self.model, self.scaler, self.selected_features])
            
            if self.is_ready:
                logger.info(f"Prediction service initialized with {len(self.selected_features)} features")
            else:
                logger.error("Failed to initialize prediction service")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_ready = False

    def _retrain_model(self):
        """Retrain the model if loading fails"""
        try:
            from src.retrain import main as retrain_model
            retrain_model()
            self._load_model()
            if self.is_ready:
                logger.info("Successfully retrained and loaded model!")
            else:
                logger.error("Failed to load model after retraining")
        except Exception as e:
            logger.error(f"Failed to retrain model: {str(e)}")
            self.is_ready = False

    def _load_pickle(self, path):
        """Load a pickle file with error handling"""
        try:
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return None
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            return None

    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Validate required columns
            required_base_columns = ['country', 'year', 'exch_usd', 'gdp_weighted_default', 'inflation_annual_cpi']
            missing_required = [col for col in required_base_columns if col not in df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            
            # Add additional columns with default values if missing
            default_columns = {
                'systemic_crisis': 0,
                'domestic_debt_in_default': 0,
                'sovereign_external_debt_default': 0,
                'currency_crises': 0,
                'inflation_crises': 0,
                'independence': 1,
                'banking_crisis': 0
            }
            
            for col, default_value in default_columns.items():
                if col not in df.columns:
                    df[col] = default_value
            
            # Sort by year to ensure correct rolling calculations
            df = df.sort_values('year')
            
            # Engineer features
            from src.preprocessing import engineer_features
            df_engineered = engineer_features(df)
            
            # Encode categorical variables
            from src.preprocessing import encode_categorical
            df_encoded = encode_categorical(df_engineered)
            
            # Convert all column names to strings
            df_encoded.columns = df_encoded.columns.astype(str)
            
            # Ensure all required features exist before scaling
            if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
                scaling_features = list(self.scaler.feature_names_in_)
                
                # Add missing features with default value 0
                for feature in scaling_features:
                    if feature not in df_encoded.columns:
                        logger.debug(f"Adding missing feature: {feature}")
                        df_encoded[feature] = 0
                
                # Ensure correct column order for scaling
                df_encoded = df_encoded.reindex(columns=scaling_features + [col for col in df_encoded.columns if col not in scaling_features])
                
                # Scale the features
                try:
                    df_encoded[scaling_features] = self.scaler.transform(df_encoded[scaling_features])
                except Exception as e:
                    logger.error(f"Error during scaling: {str(e)}")
                    raise
            
            # Select only the features used by the model
            if self.selected_features:
                result = pd.DataFrame(index=df_encoded.index)
                for feature in self.selected_features:
                    if feature in df_encoded.columns:
                        result[feature] = df_encoded[feature]
                    else:
                        logger.warning(f"Missing selected feature: {feature}, using default value 0")
                        result[feature] = 0
                
                # Verify we have all required features
                if len(result.columns) != len(self.selected_features):
                    missing = set(self.selected_features) - set(result.columns)
                    raise ValueError(f"Missing required features after preprocessing: {missing}")
                
                return result
            else:
                logger.error("No selected features available")
                raise ValueError("Model features not properly initialized")
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise ValueError(f"Error preprocessing input data: {str(e)}")
    
    def predict(self, data):
        """Make prediction for a single data point or batch"""
        if not self.is_ready:
            raise RuntimeError("Prediction service is not properly initialized. Please check the logs for details.")
            
        try:
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
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, data_list):
        """Make predictions for a batch of data points"""
        if not isinstance(data_list, list):
            raise ValueError("Data must be a list of dictionaries")
        
        df = pd.DataFrame(data_list)
        return self.predict(df)

def save_prediction_artifacts(model, scaler, selected_features, output_dir=None):
    """Save model and preprocessing artifacts for prediction"""
    try:
        # Set default output directory if none provided
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'models')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving prediction artifacts to {output_dir}")
        
        # Validate inputs
        if model is None:
            raise ValueError("Model cannot be None")
        if scaler is None:
            raise ValueError("Scaler cannot be None")
        if selected_features is None or len(selected_features) == 0:
            raise ValueError("Selected features cannot be None or empty")
        
        # Define paths
        model_path = os.path.join(output_dir, 'banking_crisis_model.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        features_path = os.path.join(output_dir, 'selected_features.pkl')
        
        # Save model with backup
        logger.info("Saving model...")
        joblib.dump(model, model_path, compress=3)
        with open(model_path + '.backup', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        # Save scaler
        logger.info("Saving scaler...")
        joblib.dump(scaler, scaler_path, compress=3)
        
        # Save selected features
        logger.info("Saving selected features...")
        joblib.dump(selected_features, features_path, compress=3)
        
        # Verify files were created
        files_to_check = [model_path, model_path + '.backup', scaler_path, features_path]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise RuntimeError(f"Failed to create file: {file_path}")
        
        logger.info("Successfully saved all prediction artifacts")
        return {
            'model_path': model_path,
            'model_backup_path': model_path + '.backup',
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
    service = PredictionService()
    
    # Test prediction on sample data
    sample_data = X_test.iloc[0].to_dict()
    prediction = service.predict(sample_data)
    print(f"Sample prediction: {prediction}")
