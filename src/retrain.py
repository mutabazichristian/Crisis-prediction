import os
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_data, preprocess_pipeline
from src.model import ModelTrainer
from src.prediction import save_prediction_artifacts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_path = os.path.join('data', 'dataset.csv')
        
        if not os.path.exists(data_path):
            logger.warning("No data file found, creating minimal dataset for deployment")
            data = {
                'country': ['TestCountry'],
                'year': [2024],
                'banking_crisis': ['no_crisis'],
                'inflation_annual_cpi': [2.0],
                'exch_usd': [1.0],
                'gdp_weighted_default': [0.0]
            }
            dataset = pd.DataFrame(data)
        else:
            logger.info(f"Loading data from {data_path}")
            dataset = load_data(data_path)
            
        X, y, scaler, selected_features = preprocess_pipeline(dataset)
        
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = trainer.train_test_split(X, y)
        
        # Train model
        logger.info("Training model...")
        model = trainer.train_model(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Save artifacts
        logger.info("Saving model artifacts...")
        output_dir = os.path.join('models')
        os.makedirs(output_dir, exist_ok=True)
        artifacts = save_prediction_artifacts(model, scaler, selected_features, output_dir)
        
        logger.info(f"Model training complete! Artifacts saved to {output_dir}")
        logger.info(f"Model performance metrics:\n{metrics['classification_report']}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 