import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
import threading
import atexit

# Import local modules
from .preprocessing import (
    handle_missing_values, encode_categorical,
    scale_features, feature_selection, preprocess_pipeline,
    validate_input_data, engineer_features
)
from .model import ModelTrainer, parallel_backend_context
from .prediction import save_prediction_artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add thread lock for model operations
model_lock = threading.Lock()

class MLPipeline:
    def __init__(self, config_path=None):
        """Initialize ML pipeline"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = self._load_config(config_path)
        self.model_trainer = ModelTrainer(random_state=self.config["random_state"])
        self.scaler = None
        self.selected_features = None
        
        # Create required directories with absolute paths
        os.makedirs(os.path.join(self.base_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'data/train'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'data/test'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'logs'), exist_ok=True)
        
        # Try to load existing model if available
        try:
            self._load_existing_model()
        except Exception as e:
            logger.warning(f"No existing model found: {e}")
        
        logger.info("ML Pipeline initialized")

    def __del__(self):
        """Cleanup resources when the pipeline is destroyed"""
        try:
            # Cleanup any remaining parallel processing resources
            import joblib.parallel
            joblib.parallel.get_active_backend()[0]._workers.shutdown(wait=True)
        except:
            pass

    def _load_config(self, config_path):
        """Load configuration from file or use default"""
        default_config = {
            "data_path": os.path.join(self.base_dir, "data/african_crises2.csv"),
            "model_output_dir": os.path.join(self.base_dir, "models"),
            "numerical_columns": ["exch_usd", "gdp_weighted_default", "inflation_annual_cpi"],
            "n_features_to_select": 8,
            "test_size": 0.25,
            "random_state": 42,
            "model_params": {
                "max_depth": 10,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "class_weight": "balanced",
                "n_estimators": 50
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                # Merge with default config to ensure all keys exist
                merged_config = {**default_config, **config}
                return merged_config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def _load_existing_model(self):
        """Load existing model and artifacts if available"""
        with model_lock:
            model_path = os.path.join(self.config["model_output_dir"], "banking_crisis_model.pkl")
            scaler_path = os.path.join(self.config["model_output_dir"], "scaler.pkl")
            features_path = os.path.join(self.config["model_output_dir"], "selected_features.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(features_path, 'rb') as f:
                    self.selected_features = pickle.load(f)
                logger.info("Loaded existing model and artifacts")
    
    def preprocess_data(self, dataset: pd.DataFrame):
        """Preprocess the input data."""
        try:
            logger.info("Starting data preprocessing")
            
            # Validate input data
            validate_input_data(dataset)
            
            # Handle missing values
            dataset = handle_missing_values(dataset, strategy='advanced')
            
            # Engineer features
            dataset = engineer_features(dataset)
            
            # Encode categorical variables
            dataset_encoded = encode_categorical(dataset)
            logger.info("Encoded categorical variables")
            
            # Scale numerical features
            numerical_columns = self.config["numerical_columns"]
            if self.scaler is None:
                dataset_scaled, self.scaler = scale_features(dataset_encoded, numerical_columns, handle_outliers=True)
            else:
                dataset_scaled = pd.DataFrame(
                    self.scaler.transform(dataset_encoded[numerical_columns]),
                    columns=numerical_columns,
                    index=dataset_encoded.index
                )
                dataset_scaled = pd.concat([
                    dataset_encoded.drop(columns=numerical_columns),
                    dataset_scaled
                ], axis=1)
            logger.info(f"Scaled columns: {numerical_columns}")
            
            # Select features
            if self.selected_features is None:
                X = dataset_scaled.drop(columns=['banking_crisis'])
                y = dataset_scaled['banking_crisis']
                X_selected, self.selected_features, selector = feature_selection(X, y, self.config["n_features_to_select"])
                X = pd.DataFrame(X_selected, columns=self.selected_features)
            else:
                X = dataset_scaled[self.selected_features]
                y = dataset_scaled['banking_crisis']
            
            logger.info(f"Selected features: {self.selected_features}")
            logger.info("Preprocessing pipeline completed successfully!")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def run_training_pipeline(self, data_path: str) -> dict:
        """Run the complete training pipeline."""
        try:
            # Load and preprocess data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Use the correct preprocess_data method
            X, y = self.preprocess_data(df)
            logger.info("Data preprocessing completed")
            
            # Use the parallel backend context for all parallel operations
            with parallel_backend_context():
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = self.model_trainer.train_test_split(X, y)
                    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
                    logger.info("Data split completed")
                    
                    # Train model with fixed parameters
                    self.model_trainer.feature_names = self.selected_features
                    self.model = self.model_trainer.train_model(X_train, y_train)
                    logger.info("Model training completed")
                    
                    # Evaluate model
                    metrics = self.model_trainer.evaluate_model(X_test, y_test)
                    logger.info(f"Model evaluation completed. Metrics: {metrics}")
                    
                    # Save model and artifacts immediately after successful training
                    self.save_model()
                    logger.info("Model and artifacts saved successfully")
                    
                    return {
                        "status": "success",
                        "metrics": metrics,
                        "feature_importance": self.model_trainer.get_feature_importance(self.selected_features),
                        "selected_features": self.selected_features
                    }
                except Exception as e:
                    logger.error(f"Error during model training/evaluation: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Cleanup any resources
            try:
                import joblib.parallel
                joblib.parallel.get_active_backend()[0]._workers.shutdown(wait=True)
            except:
                pass

    def save_model(self):
        """Save the model and artifacts with thread safety"""
        with model_lock:
            model_path = os.path.join(self.config["model_output_dir"], "banking_crisis_model.pkl")
            scaler_path = os.path.join(self.config["model_output_dir"], "scaler.pkl")
            features_path = os.path.join(self.config["model_output_dir"], "selected_features.pkl")
            
            # Save to temporary files first
            temp_model_path = model_path + '.tmp'
            temp_scaler_path = scaler_path + '.tmp'
            temp_features_path = features_path + '.tmp'
            
            try:
                with open(temp_model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                with open(temp_scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                with open(temp_features_path, 'wb') as f:
                    pickle.dump(self.selected_features, f)
                
                # Atomic rename of temporary files
                os.replace(temp_model_path, model_path)
                os.replace(temp_scaler_path, scaler_path)
                os.replace(temp_features_path, features_path)
                
                logger.info("Model and artifacts saved successfully")
            except Exception as e:
                # Clean up temporary files if they exist
                for temp_file in [temp_model_path, temp_scaler_path, temp_features_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                raise e

if __name__ == "__main__":
    try:
        # Test pipeline
        pipeline = MLPipeline()
        result = pipeline.run_training_pipeline("data/african_crises2.csv")
        print(f"Pipeline result: {result}")
    finally:
        # Ensure cleanup of parallel processing resources
        try:
            import joblib.parallel
            joblib.parallel.get_active_backend()[0]._workers.shutdown(wait=True)
        except:
            pass
