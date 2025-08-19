"""
Preprocessing pipeline for sklearn models to ensure consistent transformations
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path


class SklearnPreprocessingPipeline:
    """
    Handles all preprocessing steps for sklearn models including:
    - Categorical encoding
    - Cyclic feature engineering
    - Normalization
    """
    
    def __init__(self, predictor_columns, categorical_columns=None, cyclic_columns=None, 
                 encoding_method='onehot', normalization_method='min_max'):
        self.predictor_columns = predictor_columns
        self.categorical_columns = categorical_columns or []
        self.cyclic_columns = cyclic_columns or []
        self.encoding_method = encoding_method
        self.normalization_method = normalization_method
        self.pipeline = None
        self.feature_names = None
        self.encoders = {}
        self.scaler = None
        
    def _create_cyclic_features(self, df):
        """Add sine and cosine transformations for cyclic features"""
        df_encoded = df.copy()
        
        for col in self.cyclic_columns:
            if col in df_encoded.columns:
                # Detect the period based on column name or values
                if 'hour' in col.lower():
                    period = 24
                elif 'day' in col.lower() and 'week' in col.lower():
                    period = 7
                elif 'month' in col.lower():
                    period = 12
                elif 'bearing' in col.lower() or 'degree' in col.lower():
                    period = 360
                else:
                    # Try to infer from data
                    period = df_encoded[col].max() + 1
                
                # Create sine and cosine features
                df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df_encoded[col] / period)
                df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df_encoded[col] / period)
                
                # Drop original column
                df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    def _get_scaler(self):
        """Get the appropriate scaler based on normalization method"""
        if self.normalization_method == 'min_max':
            return MinMaxScaler()
        elif self.normalization_method == 'standard':
            return StandardScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            return None
    
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline"""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.predictor_columns)
        
        # Create a copy to avoid modifying original
        X_processed = X.copy()
        
        # Step 1: Add cyclic features
        if self.cyclic_columns:
            X_processed = self._create_cyclic_features(X_processed)
        
        # Step 2: Encode categorical features
        if self.categorical_columns:
            if self.encoding_method == 'onehot':
                for col in self.categorical_columns:
                    if col in X_processed.columns:
                        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(X_processed[[col]])
                        
                        # Store encoder for later use
                        self.encoders[col] = encoder
                        
                        # Create feature names
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                        
                        # Replace column with encoded features
                        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_processed.index)
                        X_processed = pd.concat([X_processed.drop(col, axis=1), encoded_df], axis=1)
        
        # Step 3: Store feature names after encoding
        self.feature_names = list(X_processed.columns)
        
        # Step 4: Fit scaler
        self.scaler = self._get_scaler()
        if self.scaler:
            self.scaler.fit(X_processed)
        
        return self
    
    def transform(self, X):
        """Transform the data using fitted preprocessors"""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.predictor_columns)
        
        # Create a copy to avoid modifying original
        X_processed = X.copy()
        
        # Step 1: Add cyclic features
        if self.cyclic_columns:
            X_processed = self._create_cyclic_features(X_processed)
        
        # Step 2: Encode categorical features using fitted encoders
        if self.categorical_columns and self.encoders:
            for col, encoder in self.encoders.items():
                if col in X_processed.columns:
                    encoded = encoder.transform(X_processed[[col]])
                    
                    # Create feature names
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    
                    # Replace column with encoded features
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_processed.index)
                    X_processed = pd.concat([X_processed.drop(col, axis=1), encoded_df], axis=1)
        
        # Ensure columns match those from fit
        if self.feature_names:
            # Add any missing columns with zeros
            for col in self.feature_names:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            
            # Select only the columns that were present during fit
            X_processed = X_processed[self.feature_names]
        
        # Step 3: Apply scaling
        if self.scaler:
            X_array = self.scaler.transform(X_processed)
            return X_array
        else:
            return X_processed.values
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_preprocessing_info(self):
        """Get information about the preprocessing steps"""
        return {
            'predictor_columns': self.predictor_columns,
            'categorical_columns': self.categorical_columns,
            'cyclic_columns': self.cyclic_columns,
            'encoding_method': self.encoding_method,
            'normalization_method': self.normalization_method,
            'feature_names_after_preprocessing': self.feature_names,
            'n_features_after_preprocessing': len(self.feature_names) if self.feature_names else len(self.predictor_columns)
        }
    
    def save(self, filepath):
        """Save the preprocessing pipeline"""
        save_dict = {
            'predictor_columns': self.predictor_columns,
            'categorical_columns': self.categorical_columns,
            'cyclic_columns': self.cyclic_columns,
            'encoding_method': self.encoding_method,
            'normalization_method': self.normalization_method,
            'feature_names': self.feature_names,
            'encoders': self.encoders,
            'scaler': self.scaler
        }
        joblib.dump(save_dict, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved preprocessing pipeline"""
        save_dict = joblib.load(filepath)
        
        pipeline = cls(
            predictor_columns=save_dict['predictor_columns'],
            categorical_columns=save_dict['categorical_columns'],
            cyclic_columns=save_dict['cyclic_columns'],
            encoding_method=save_dict['encoding_method'],
            normalization_method=save_dict['normalization_method']
        )
        
        pipeline.feature_names = save_dict['feature_names']
        pipeline.encoders = save_dict['encoders']
        pipeline.scaler = save_dict['scaler']
        
        return pipeline