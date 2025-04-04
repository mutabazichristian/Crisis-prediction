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
        path = "data/african_crises2.csv"
        dataset = load_data(path)
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
        
        # Save artifacts with new robust saving mechanism
        logger.info("Saving model artifacts...")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        artifacts = save_prediction_artifacts(model, scaler, selected_features, output_dir)
        
        logger.info(f"Model training complete! Artifacts saved to {output_dir}")
        logger.info(f"Model performance metrics:\n{metrics['classification_report']}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 