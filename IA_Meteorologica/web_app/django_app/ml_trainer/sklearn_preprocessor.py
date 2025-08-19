"""
Preprocessor for sklearn models to handle feature transformations consistently
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import json
from typing import Dict, List, Any, Optional, Tuple
import os


class SklearnPreprocessor:
    """
    Handles all preprocessing for sklearn models including:
    - Categorical encoding
    - Cyclic features
    - Feature scaling
    - Column tracking
    """
    
    def __init__(self):
        self.categorical_columns = []
        self.cyclic_columns = []
        self.numeric_columns = []
        self.encoding_method = 'onehot'
        self.encoders = {}
        self.scaler = None
        self.feature_names_out = []
        self.original_columns = []
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y=None, 
            categorical_columns: List[str] = None,
            cyclic_columns: List[str] = None,
            encoding_method: str = 'onehot'):
        """
        Fit the preprocessor on training data
        """
        self.original_columns = X.columns.tolist()
        self.encoding_method = encoding_method
        
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            self.categorical_columns = []
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].nunique() < 10:
                    self.categorical_columns.append(col)
        else:
            self.categorical_columns = categorical_columns
            
        # Auto-detect cyclic columns if not provided
        if cyclic_columns is None:
            self.cyclic_columns = []
            for col in X.columns:
                if any(term in col.lower() for term in ['hour', 'day', 'month', 'bearing', 'degree', 'angle']):
                    self.cyclic_columns.append(col)
        else:
            self.cyclic_columns = cyclic_columns
            
        # Identify numeric columns (excluding categorical and cyclic)
        self.numeric_columns = [col for col in X.columns 
                               if col not in self.categorical_columns 
                               and col not in self.cyclic_columns]
        
        # Fit encoders for categorical columns
        if self.categorical_columns:
            if self.encoding_method == 'onehot':
                self.encoders['onehot'] = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                self.encoders['onehot'].fit(X[self.categorical_columns])
            elif self.encoding_method == 'ordinal':
                self.encoders['ordinal'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.encoders['ordinal'].fit(X[self.categorical_columns])
        
        self.fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using fitted preprocessors
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X_transformed = X.copy()
        
        # Apply categorical encoding
        if self.categorical_columns:
            if self.encoding_method == 'onehot' and 'onehot' in self.encoders:
                # One-hot encode each categorical column
                encoded_data = self.encoders['onehot'].transform(X_transformed[self.categorical_columns])
                feature_names = self.encoders['onehot'].get_feature_names_out(self.categorical_columns)
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X_transformed.index)
                
                # Drop original categorical columns and add encoded ones
                X_transformed = X_transformed.drop(columns=self.categorical_columns)
                X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                
            elif self.encoding_method == 'ordinal' and 'ordinal' in self.encoders:
                encoded_data = self.encoders['ordinal'].transform(X_transformed[self.categorical_columns])
                X_transformed[self.categorical_columns] = encoded_data
        
        # Apply cyclic transformations
        if self.cyclic_columns:
            for col in self.cyclic_columns:
                if col in X_transformed.columns:
                    # Assume cyclic features have a known period
                    if 'hour' in col.lower():
                        period = 24
                    elif 'month' in col.lower():
                        period = 12
                    elif 'day' in col.lower() and 'week' in col.lower():
                        period = 7
                    elif 'bearing' in col.lower() or 'degree' in col.lower():
                        period = 360
                    else:
                        period = X_transformed[col].max() + 1
                    
                    X_transformed[f'{col}_sin'] = np.sin(2 * np.pi * X_transformed[col] / period)
                    X_transformed[f'{col}_cos'] = np.cos(2 * np.pi * X_transformed[col] / period)
                    X_transformed = X_transformed.drop(columns=[col])
        
        # Store feature names for later use
        self.feature_names_out = X_transformed.columns.tolist()
        
        return X_transformed.values
        
    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> np.ndarray:
        """
        Fit and transform in one step
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)
        
    def save(self, filepath: str):
        """
        Save the preprocessor state
        """
        state = {
            'categorical_columns': self.categorical_columns,
            'cyclic_columns': self.cyclic_columns,
            'numeric_columns': self.numeric_columns,
            'encoding_method': self.encoding_method,
            'feature_names_out': self.feature_names_out,
            'original_columns': self.original_columns,
            'fitted': self.fitted
        }
        
        # Save state as JSON
        state_file = filepath.replace('.pkl', '_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f)
            
        # Save encoders
        if self.encoders:
            encoders_file = filepath.replace('.pkl', '_encoders.pkl')
            joblib.dump(self.encoders, encoders_file)
            
    def load(self, filepath: str):
        """
        Load the preprocessor state
        """
        # Load state
        state_file = filepath.replace('.pkl', '_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.categorical_columns = state['categorical_columns']
            self.cyclic_columns = state['cyclic_columns']
            self.numeric_columns = state['numeric_columns']
            self.encoding_method = state['encoding_method']
            self.feature_names_out = state['feature_names_out']
            self.original_columns = state['original_columns']
            self.fitted = state['fitted']
            
        # Load encoders
        encoders_file = filepath.replace('.pkl', '_encoders.pkl')
        if os.path.exists(encoders_file):
            self.encoders = joblib.load(encoders_file)
            
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features after transformation
        """
        return self.feature_names_out