import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

def validate_input_data(df: pd.DataFrame) -> bool:
    """Validate input data for required columns and data types"""
    required_columns = ['banking_crisis', 'country', 'year']
    required_numeric_columns = ['inflation_annual_cpi', 'exch_usd', 'gdp_weighted_default']
    
    try:
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for minimum number of rows
        if len(df) < 10:
            raise ValueError("Dataset must contain at least 10 rows")
            
        # Check numeric columns
        for col in required_numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column {col} must be numeric")
                
        return True
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and transform existing ones"""
    try:
        df = df.copy()
        
        # Add year-based features
        if 'year' in df.columns:
            df['decade'] = (df['year'] // 10) * 10
            
        # Create interaction features for important numeric columns
        important_numeric_cols = ['inflation_annual_cpi', 'exch_usd', 'gdp_weighted_default']
        numeric_cols = [col for col in important_numeric_cols if col in df.columns]
        
        for col1, col2 in combinations(numeric_cols, 2):
            if col1 != 'banking_crisis' and col2 != 'banking_crisis':
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                
        # Add rolling statistics if time series data is present
        if 'year' in df.columns and len(numeric_cols) > 0:
            for col in numeric_cols:
                if col != 'banking_crisis':
                    df[f'{col}_rolling_mean'] = df.groupby('country')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std'] = df.groupby('country')[col].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
                    
        logger.info("Feature engineering completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def handle_missing_values(df: pd.DataFrame, strategy: str = 'advanced') -> pd.DataFrame:
    """Enhanced missing values handling with multiple strategies"""
    try:
        df = df.copy()
        
        if strategy == 'advanced':
            # Check missing percentage
            missing_percent = df.isnull().sum() / len(df) * 100
            
            for column in df.columns:
                missing_pct = missing_percent[column]
                
                if missing_pct > 0:
                    if missing_pct > 50:
                        logger.warning(f"Column {column} has {missing_pct:.2f}% missing values")
                        
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # For numeric columns, use interpolation where possible
                        if 'year' in df.columns:
                            df[column] = df.groupby('country')[column].interpolate(method='linear')
                        # Fallback to median for remaining NAs
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        # For categorical, use forward fill within groups if applicable
                        if 'country' in df.columns:
                            df[column] = df.groupby('country')[column].ffill().bfill()
                        # Fallback to mode
                        df[column] = df[column].fillna(df[column].mode()[0])
        else:
            # Original simple imputation strategy
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            categorical_columns = df.select_dtypes(include=['object']).columns
            df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        logger.info(f"Missing values handled successfully using {strategy} strategy")
        return df
    except Exception as e:
        logger.error(f"Error in handling missing values: {str(e)}")
        raise

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables"""
    try:
        df = df.copy()
        
        # Encode banking crisis column if it exists
        if 'banking_crisis' in df.columns:
            df['banking_crisis'] = df['banking_crisis'].replace(['no_crisis', 'crisis'], [0, 1])
        
        # Create country mapping if not exists
        if 'country' in df.columns and not df['country'].dtype.kind in 'iu':
            countries = df['country'].unique()
            country_mapping = {country: idx for idx, country in enumerate(sorted(countries))}
            df['country'] = df['country'].map(country_mapping)
            logger.info(f"Country mapping created for {len(countries)} countries")
        
        logger.info("Categorical variables encoded successfully")
        return df
    except Exception as e:
        logger.error(f"Error encoding categorical variables: {str(e)}")
        raise

def scale_features(df: pd.DataFrame, columns_to_scale: list, handle_outliers: bool = True) -> tuple:
    """Enhanced feature scaling with outlier handling"""
    try:
        df_scaled = df.copy()
        scaler = StandardScaler()
        
        if handle_outliers and columns_to_scale:
            for col in columns_to_scale:
                # Calculate IQR for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_scaled[col] = df_scaled[col].clip(lower_bound, upper_bound)
                logger.info(f"Outliers handled for column: {col}")
        
        # Scale features
        if columns_to_scale:
            scaled_features = scaler.fit_transform(df_scaled[columns_to_scale])
            df_scaled[columns_to_scale] = scaled_features
            
        logger.info(f"Features scaled successfully: {len(columns_to_scale)} columns")
        return df_scaled, scaler
    except Exception as e:
        logger.error(f"Error in feature scaling: {str(e)}")
        raise

def feature_selection(X: pd.DataFrame, y: pd.Series, n_features: int) -> tuple:
    """Select best features using SelectKBest"""
    try:
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Log feature scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        logger.info(f"Feature importance scores:\n{feature_scores}")
        
        logger.info(f"Selected {n_features} best features: {selected_features}")
        return X_selected, selected_features, selector
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise

def preprocess_pipeline(df: pd.DataFrame, target_col: str = 'banking_crisis', n_features: int = 8) -> tuple:
    """Run complete preprocessing pipeline"""
    try:
        # Validate input data
        validate_input_data(df)
        
        # Handle missing values
        df_clean = handle_missing_values(df, strategy='advanced')
        
        # Engineer features
        df_engineered = engineer_features(df_clean)
        
        # Encode categorical variables
        df_encoded = encode_categorical(df_engineered)
        
        # Split features and target
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_scaled, scaler = scale_features(X, numerical_columns, handle_outliers=True)
        
        # Select best features
        X_selected, selected_features, selector = feature_selection(X_scaled, y, n_features)
        
        logger.info("Preprocessing pipeline completed successfully")
        return X_selected, y, scaler, selected_features, selector
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        path = "../data/african_crises2.csv"
        dataset = pd.read_csv(path)
        X, y, scaler, selected_features, selector = preprocess_pipeline(dataset)
        print(f'Preprocessed data shape: {X.shape}')
        print(f'Selected features: {selected_features}')
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


