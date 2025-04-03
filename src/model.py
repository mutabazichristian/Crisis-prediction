import os
import pandas as pd
import numpy as np 
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Any
from contextlib import contextmanager

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.exceptions import NotFittedError
from joblib import parallel_backend, Parallel, delayed

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

@contextmanager
def parallel_backend_context():
    """Context manager for parallel processing"""
    try:
        with parallel_backend('threading', n_jobs=-1):
            yield
    finally:
        # Cleanup is handled automatically when the context exits
        pass

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.metrics = {}
        
    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets with stratification"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            logger.info(f'Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in train_test_split: {str(e)}")
            raise

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train the model with fixed parameters"""
        try:
            logger.info('Training model...')
            
            # Initialize model with fixed parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state
            )
            
            # Perform cross-validation before final training using threading backend
            with parallel_backend_context():
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
                logger.info(f'Cross-validation scores: mean={cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
            
            # Final training on full training set
            self.model.fit(X_train, y_train)
            
            return self.model
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        try:
            if self.model is None:
                raise NotFittedError("Model must be trained before evaluation")

            logger.info('Evaluating model...')

            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

            # Calculate various metrics
            self.metrics = {
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            self.metrics['precision_recall'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            }

            # Log the results
            logger.info(f'Classification Report:\n{classification_report(y_test, y_pred)}')
            logger.info(f'ROC AUC Score: {self.metrics["roc_auc"]:.4f}')

            return self.metrics
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate learning curves to analyze model performance"""
        try:
            with parallel_backend_context():
                train_sizes, train_scores, test_scores = learning_curve(
                    self.model, X, y, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='f1'
                )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            curves = {
                'train_sizes': train_sizes.tolist(),
                'train_mean': train_mean.tolist(),
                'train_std': train_std.tolist(),
                'test_mean': test_mean.tolist(),
                'test_std': test_std.tolist()
            }

            logger.info('Learning curves generated successfully')
            return curves
        except Exception as e:
            logger.error(f"Error generating learning curves: {str(e)}")
            raise

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Calculate and return feature importance scores"""
        try:
            if self.model is None:
                raise NotFittedError("Model must be trained before getting feature importance")

            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            feature_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })

            logger.info('Feature importance calculated successfully')
            return feature_importance
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def save_model(self, path: str = None, metadata: Dict = None) -> None:
        """Save model and associated metadata"""
        try:
            if self.model is None:
                raise NotFittedError("Model must be trained before saving")

            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(MODEL_DIR, f'banking_crisis_model_{timestamp}.pkl')

            # Prepare metadata
            metadata = metadata or {}
            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'model_parameters': self.model.get_params()
            })

            # Save model and metadata
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'metadata': metadata
                }, f)

            # Save metadata separately in JSON format
            metadata_path = path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f'Model and metadata saved successfully to {path}')
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @staticmethod
    def load_model(path: str) -> Tuple[RandomForestClassifier, Dict]:
        """Load model and metadata from file"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            model = data['model']
            metadata = data['metadata']
            
            logger.info(f'Model loaded successfully from {path}')
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        from src.preprocessing import preprocess_pipeline
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Load and preprocess data
        path = '../data/african_crises2.csv'
        dataset = pd.read_csv(path)
        X, y, scaler, selected_features, selector = preprocess_pipeline(dataset)
        
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        trainer.feature_names = selected_features
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.train_test_split(X, y)
        
        # Train model with fixed parameters
        model = trainer.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Generate learning curves
        learning_curves = trainer.plot_learning_curves(X, y)
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance(selected_features)
        print("\nFeature importance:")
        print(feature_importance)
        
        # Save model with metadata
        metadata = {
            'data_path': path,
            'preprocessing_params': {
                'n_features': len(selected_features),
                'features': selected_features
            },
            'learning_curves': learning_curves
        }
        trainer.save_model(metadata=metadata)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
