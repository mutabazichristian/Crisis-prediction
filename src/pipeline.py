import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Import local modules
from src.preprocessing import (
    load_data, handle_missing_values, encode_categorical,
    scale_features, feature_selection 
)
from src.model import train_model, evaluate_model, get_feature_importance
from src.prediction import save_prediction_artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, config_path=None):
        """Initialize ML pipeline"""
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = None
        self.selected_features = None
        
        # Create required directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/test', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("ML Pipeline initialized")
    
    def _load_config(self, config_path):
        """Load configuration from file or use default"""
        default_config = {
            "data_path": "data/african_crises2.csv",
            "model_output_dir": "models",
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
    
    def run_training_pipeline(self, data_path=None):
        """Run complete training pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting training pipeline at {start_time}")
        
        try:
            # Use specified data path or default from config
            if data_path is None:
                data_path = self.config["data_path"]
            
            # Step 1: Load data
            logger.info(f"Loading data from {data_path}")
            df = load_data(data_path)
            
            # Step 2: Basic preprocessing
            logger.info("Preprocessing data")
            # Check for missing values
            check_missing_values(df)
            
            # Encode categorical 
            df_encoded = encode_categorical(df)
            
            # Split into features and target
            target_col = 'banking_crisis'
            y = df_encoded[target_col]
            X = df_encoded.drop(columns=[target_col])
            
            # Scale numerical features
            numerical_columns = self.config["numerical_columns"]
            X_scaled, self.scaler = scale_numerical_features(X, numerical_columns)
            
            # Feature selection
            X_selected, self.selected_features = select_features(
                X_scaled, y, self.config["n_features_to_select"]
            )
            
            # Step 3: Split data
            logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y,
                test_size=self.config["test_size"],
                random_state=self.config["random_state"],
                stratify=y
            )
            
            # Save train/test split for future reference
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_df.to_csv(f"data/train/train_{timestamp}.csv", index=False)
            test_df.to_csv(f"data/test/test_{timestamp}.csv", index=False)
            
            # Step 4: Train model
            logger.info("Training model")
            self.model = train_model(X_train, y_train, self.config["model_params"])
            
            # Step 5: Evaluate model
            logger.info("Evaluating model")
            metrics = evaluate_model(self.model, X_test, y_test)
            
            # Get feature importance
            feature_importance = get_feature_importance(self.model, self.selected_features)

        except Exception as e:
                print(f'failed to complete pipeline run with error: {e}')
