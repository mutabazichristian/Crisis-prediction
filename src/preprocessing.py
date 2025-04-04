import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from itertools import combinations
import os

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
            
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

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
    """Engineer features for the model"""
    try:
        logger.info("Starting feature engineering...")
        df = df.copy()
        
        # Convert year to decade
        df['decade'] = (df['year'] // 10) * 10
        
        # Create interaction features
        df['inflation_annual_cpi_exch_usd_interaction'] = df['inflation_annual_cpi'] * df['exch_usd']
        df['inflation_annual_cpi_gdp_weighted_default_interaction'] = df['inflation_annual_cpi'] * df['gdp_weighted_default']
        df['exch_usd_gdp_weighted_default_interaction'] = df['exch_usd'] * df['gdp_weighted_default']
        
        # Calculate rolling statistics with safe window size
        window_size = min(5, len(df))
        
        # Group by country and calculate rolling statistics
        for country in df['country'].unique():
            country_mask = df['country'] == country
            country_data = df[country_mask].sort_values('year')
            
            # Calculate rolling statistics for each feature
            for feature in ['inflation_annual_cpi', 'exch_usd', 'gdp_weighted_default']:
                if feature in df.columns:
                    # Calculate rolling mean
                    rolling_mean = country_data[feature].rolling(window=window_size, min_periods=1).mean()
                    df.loc[country_mask, f'{feature}_rolling_mean'] = rolling_mean
                    
                    # Calculate rolling std with minimum 1 value
                    rolling_std = country_data[feature].rolling(window=window_size, min_periods=1).std()
                    rolling_std = rolling_std.fillna(country_data[feature].std())  # Fill NaN with overall std
                    df.loc[country_mask, f'{feature}_rolling_std'] = rolling_std
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col}, filling with mean")
                df[col] = df[col].fillna(df[col].mean())
        
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
        
        # Define fixed country mapping
        COUNTRY_MAPPING = {
            "Algeria": 0,
            "Angola": 1,
            "Central African Republic": 2,
            "Egypt": 3,
            "Ivory Coast": 4,
            "Kenya": 5,
            "Mauritius": 6,
            "Morocco": 7,
            "Nigeria": 8,
            "South Africa": 9,
            "Tunisia": 10,
            "Zambia": 11,
            "Zimbabwe": 12
        }
        
        # Encode banking crisis column if it exists
        if 'banking_crisis' in df.columns:
            df['banking_crisis'] = df['banking_crisis'].replace(['no_crisis', 'crisis'], [0, 1])
        
        # Use fixed country mapping
        if 'country' in df.columns:
            try:
                df['country'] = df['country'].map(COUNTRY_MAPPING)
                logger.info(f"Countries encoded using fixed mapping")
            except Exception as e:
                logger.error(f"Error mapping country values: {str(e)}")
                raise ValueError(f"Invalid country values found. Valid values are: {list(COUNTRY_MAPPING.keys())}")
        
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
        # Create a copy to avoid modifying the original DataFrame
        X = X.copy()
        
        # Convert all column names to plain strings (no str_ type)
        X.columns = [str(col) for col in X.columns]
        
        # Separate numeric and categorical columns
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col != 'country']
        categorical_cols = ['country']  # We always want to keep the country column
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features available for selection")
            return X, X.columns.tolist(), None
        
        # Apply feature selection only on numeric columns
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(numeric_cols)))
        X_numeric = X[numeric_cols].copy()
        
        # Ensure numeric column names are strings
        X_numeric.columns = [str(col) for col in X_numeric.columns]
        
        X_numeric_selected = selector.fit_transform(X_numeric, y)
        selected_numeric_features = [str(col) for col in np.array(numeric_cols)[selector.get_support()]]
        
        # Log feature scores for numeric features
        feature_scores = pd.DataFrame({
            'Feature': numeric_cols,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        logger.info(f"Feature importance scores:\n{feature_scores}")
        
        # Combine selected numeric features with categorical features
        selected_features = categorical_cols + selected_numeric_features
        
        # Create the final selected features DataFrame
        X_selected = pd.DataFrame(index=X.index)
        
        # Add categorical columns first
        for col in categorical_cols:
            X_selected[str(col)] = X[col]
        
        # Add selected numeric features
        numeric_df = pd.DataFrame(
            X_numeric_selected, 
            columns=[str(f) for f in selected_numeric_features],
            index=X.index
        )
        X_selected = pd.concat([X_selected, numeric_df], axis=1)
        
        # Final check to ensure all column names are strings
        X_selected.columns = [str(col) for col in X_selected.columns]
        selected_features = [str(f) for f in selected_features]
        
        logger.info(f"Selected features: {selected_features}")
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
        
        # Convert all column names to strings early in the pipeline
        df_engineered.columns = [str(col) for col in df_engineered.columns]
        
        # Encode categorical variables
        df_encoded = encode_categorical(df_engineered)
        
        # Ensure column names are strings after encoding
        df_encoded.columns = [str(col) for col in df_encoded.columns]
        
        # Split features and target
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        # Convert all column names to strings again after split
        X.columns = [str(col) for col in X.columns]
        
        # Scale only numerical features, excluding 'country' column
        numerical_columns = [col for col in X.select_dtypes(include=[np.number]).columns if col != 'country']
        X_scaled = X.copy()
        
        if numerical_columns:
            # Ensure numerical column names are strings
            numerical_columns = [str(col) for col in numerical_columns]
            X_scaled_nums, scaler = scale_features(X, numerical_columns, handle_outliers=True)
            X_scaled[numerical_columns] = X_scaled_nums[numerical_columns]
        else:
            scaler = None
        
        # Final NaN check before feature selection
        if X_scaled.isna().any().any():
            logger.warning("Found NaN values before feature selection, filling with appropriate values")
            for col in X_scaled.columns:
                if X_scaled[col].isna().any():
                    if np.issubdtype(X_scaled[col].dtype, np.number):
                        X_scaled[col] = X_scaled[col].fillna(X_scaled[col].mean())
                    else:
                        X_scaled[col] = X_scaled[col].fillna(X_scaled[col].mode()[0])
        
        # Ensure all column names are strings before feature selection
        X_scaled.columns = [str(col) for col in X_scaled.columns]
        
        # Select best features
        X_selected, selected_features, selector = feature_selection(X_scaled, y, n_features)
        
        # Final verification of string column names
        X_selected.columns = [str(col) for col in X_selected.columns]
        selected_features = [str(f) for f in selected_features]
        
        # Verify no NaN values in final output
        assert not X_selected.isna().any().any(), "NaN values found in final preprocessed data"
        
        logger.info("Preprocessing pipeline completed successfully!")
        return X_selected, y, scaler, selected_features
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


