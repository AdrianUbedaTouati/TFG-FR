"""
Training session management views
"""
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404, render
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import FileResponse, Http404
import threading

from ..models import TrainingSession, ModelDefinition
from ..serializers import TrainingSessionSerializer
from ..ml_utils import train_model
from ..utils import error_response, success_response
from ..constants import (
    STATUS_TRAINING, SUCCESS_TRAINING_STARTED,
    ERROR_TRAINING_FAILED
)
import pandas as pd
import numpy as np
import joblib
import os


class TrainingSessionListCreateView(generics.ListCreateAPIView):
    """List all training sessions or create a new one"""
    serializer_class = TrainingSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter training sessions by user"""
        if self.request.user.is_staff:
            return TrainingSession.objects.all()
        return TrainingSession.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Add user to training session"""
        serializer.save(user=self.request.user)


class TrainingSessionDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update or delete a training session"""
    serializer_class = TrainingSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter training sessions by user"""
        if self.request.user.is_staff:
            return TrainingSession.objects.all()
        return TrainingSession.objects.filter(user=self.request.user)


class TrainModelView(APIView):
    """Start training for a model"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        """Start training session"""
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=pk)
        else:
            session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
        
        # Check if already training
        if session.status == STATUS_TRAINING:
            return error_response("Model is already training")
        
        # Update status
        session.status = STATUS_TRAINING
        session.save()
        
        # Start training in background thread
        def train_async():
            try:
                train_model(session)
            except Exception as e:
                session.status = 'failed'
                session.error_message = str(e)
                session.save()
        
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
        
        return success_response(
            {'session_id': session.id},
            message=SUCCESS_TRAINING_STARTED
        )
    
    def get(self, request, pk):
        """Get training status"""
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=pk)
        else:
            session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
        
        response_data = {
            'id': session.id,
            'status': session.status,
            'created_at': session.created_at,
            'updated_at': getattr(session, 'updated_at', session.created_at),
            'model_type': session.model_type,
            'framework': getattr(session, 'framework', 'keras'),
            'dataset_name': session.dataset.name if session.dataset else None,
            'predictor_columns': session.predictor_columns,
            'target_columns': session.target_columns,
            # Progress tracking fields
            'progress': session.progress,
            'current_epoch': session.current_epoch,
            'total_epochs': session.total_epochs,
            'current_batch': session.current_batch,
            'total_batches': session.total_batches,
            'train_loss': session.train_loss,
            'val_loss': session.val_loss,
            'train_accuracy': session.train_accuracy,
            'val_accuracy': session.val_accuracy,
            'training_logs': session.training_logs,
        }
        
        # Add results if completed
        if session.status == 'completed':
            response_data.update({
                'training_history': session.training_history,
                'test_results': session.test_results,
                'model_file': session.model_file.url if session.model_file else None
            })
        elif session.status == 'failed':
            response_data['error_message'] = session.error_message
        
        return success_response(response_data)


class TrainingResultsView(LoginRequiredMixin, TemplateView):
    """Display training results"""
    template_name = 'training_results.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        session_id = self.kwargs.get('pk')
        
        # Get the training session
        if self.request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=session_id)
        else:
            session = get_object_or_404(TrainingSession, pk=session_id, user=self.request.user)
        
        context['session_id'] = session_id
        context['session'] = session
        return context


class TrainingAnalysisView(APIView):
    """Provide detailed analysis data for training results"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        """Get detailed analysis for training session"""
        print(f"=== TrainingAnalysisView GET request for session {pk} ===")
        
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=pk)
        else:
            session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
        
        print(f"Session found: ID={session.id}, Status={session.status}, Model type={session.model_type}")
        print(f"Model file: {session.model_file}")
        print(f"Dataset: {session.dataset}")
        
        if session.status != 'completed':
            print(f"Session not completed, status: {session.status}")
            return error_response("Training session is not completed")
            
        try:
            print("Starting analysis generation...")
            analysis_data = self._generate_analysis(session)
            print(f"Analysis completed successfully. Keys: {analysis_data.keys()}")
            
            # Debug: Print what we're returning
            if 'predictions_analysis' in analysis_data:
                print(f"Predictions analysis type: {type(analysis_data['predictions_analysis'])}")
                if analysis_data['predictions_analysis']:
                    print(f"Predictions analysis keys: {analysis_data['predictions_analysis'].keys() if isinstance(analysis_data['predictions_analysis'], dict) else 'Not a dict'}")
            
            return Response(analysis_data)
        except Exception as e:
            print(f"ERROR in analysis generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return error_response(f"Error generating analysis: {str(e)}")
    
    def _generate_analysis(self, session):
        """Generate comprehensive analysis based on model type"""
        print(f"_generate_analysis called for session {session.id}")
        
        analysis = {
            'session_id': session.id,
            'model_type': session.model_type,
            'status': session.status,
            'basic_info': self._get_basic_info(session),
            'predictions_analysis': None,
            'feature_importance': None,
            'confusion_matrix': None,
            'residuals_analysis': None
        }
        
        print(f"Basic analysis structure created")
        
        # Load model and make predictions for analysis
        if session.model_file:
            # Fix path if it's relative
            model_path = session.model_file.path
            if not os.path.isabs(model_path):
                # If it's a relative path, make it absolute from MEDIA_ROOT
                from django.conf import settings
                model_path = os.path.join(settings.MEDIA_ROOT, session.model_file.name)
            
            print(f"Checking model file at: {model_path}")
            if os.path.exists(model_path):
                print(f"Model file exists at: {model_path}")
                try:
                    print("Calling _generate_predictions_analysis...")
                    predictions_data = self._generate_predictions_analysis(session, model_path)
                    print(f"Predictions data returned: {type(predictions_data)}")
                    print(f"Predictions data keys: {predictions_data.keys() if isinstance(predictions_data, dict) else 'Not a dict'}")
                    
                    # Debug: Check if we got actual predictions analysis
                    if 'predictions_analysis' in predictions_data and predictions_data['predictions_analysis']:
                        print(f"SUCCESS: Got predictions analysis with {len(predictions_data['predictions_analysis'])} targets")
                        for target_key, target_data in predictions_data['predictions_analysis'].items():
                            print(f"  Target {target_key}: has scatter_data={bool(target_data.get('scatter_data'))}, has confusion_matrix={bool(target_data.get('confusion_matrix'))}")
                    else:
                        print("WARNING: No predictions analysis data generated")
                    
                    analysis.update(predictions_data)
                except Exception as e:
                    print(f"Error generating predictions analysis: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Model file not found at: {model_path}")
        else:
            print(f"No model file associated with session")
        
        # Add model-specific analysis
        if session.model_type in ['random_forest', 'decision_tree', 'xgboost']:
            print("Adding sklearn model analysis...")
            # Pass the corrected model path if we have it
            sklearn_analysis = self._analyze_sklearn_model(session, model_path if 'model_path' in locals() else None)
            print(f"Sklearn analysis keys: {sklearn_analysis.keys()}")
            analysis.update(sklearn_analysis)
        elif session.model_type in ['lstm', 'gru', 'cnn']:
            print("Adding neural model analysis...")
            analysis.update(self._analyze_neural_model(session))
        
        print(f"Final analysis keys: {analysis.keys()}")
        return analysis
    
    def _get_basic_info(self, session):
        """Get basic information about the training session"""
        return {
            'model_name': session.name,
            'model_type': session.model_type,
            'dataset_name': session.dataset.name,
            'predictor_columns': session.predictor_columns,
            'target_columns': session.target_columns,
            'train_split': session.train_split,
            'val_split': session.val_split,
            'test_split': session.test_split,
            'training_duration': self._calculate_duration(session),
            'hyperparameters': session.hyperparameters
        }
    
    def _calculate_duration(self, session):
        """Calculate training duration in seconds"""
        if session.created_at and session.updated_at:
            return (session.updated_at - session.created_at).total_seconds()
        return None
    
    def _generate_predictions_analysis(self, session, model_path=None):
        """Generate predictions vs actual analysis"""
        try:
            # Load the dataset
            df = pd.read_csv(session.dataset.file.path)
            print(f"Dataset loaded: {len(df)} rows, columns: {df.columns.tolist()}")
            
            # Check if required columns exist
            missing_predictors = [col for col in session.predictor_columns if col not in df.columns]
            missing_targets = [col for col in session.target_columns if col not in df.columns]
            
            if missing_predictors:
                print(f"Missing predictor columns: {missing_predictors}")
                return {'predictions_analysis': f'Missing predictor columns: {missing_predictors}'}
            
            if missing_targets:
                print(f"Missing target columns: {missing_targets}")
                return {'predictions_analysis': f'Missing target columns: {missing_targets}'}
            
            # Prepare data splits (similar to prepare_data in ml_utils)
            n = len(df)
            
            # Handle None splits - use defaults that match training
            train_split = session.train_split if session.train_split is not None else 0.7
            val_split = session.val_split if session.val_split is not None else 0.15
            test_split = session.test_split if session.test_split is not None else 0.15
            
            print(f"Session splits - train: {session.train_split}, val: {session.val_split}, test: {session.test_split}")
            print(f"Using splits - train: {train_split}, val: {val_split}, test: {test_split}")
            
            train_end = int(n * train_split)
            val_end = int(n * (train_split + val_split))
            
            print(f"Data splits - Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{n}")
            print(f"Dataset total size: {n}")
            
            # Get test data
            X_test = df[session.predictor_columns].iloc[val_end:].values
            y_test_raw = df[session.target_columns].iloc[val_end:]
            
            print(f"Test data shape - X: {X_test.shape}, y_test_raw: {y_test_raw.shape}")
            print(f"Target columns data types: {y_test_raw.dtypes}")
            print(f"Sample target values: {y_test_raw.head()}")
            
            # Handle categorical targets (like 'Summary')
            y_test = y_test_raw.values.copy()  # Make a copy to avoid modifying original
            target_encoders = {}
            
            # Check if model was saved with target encoders
            if model_path and os.path.exists(model_path):
                try:
                    model_data_check = joblib.load(model_path)
                    if 'target_encoders' in model_data_check:
                        print(f"Found saved target encoders in model file")
                        target_encoders = model_data_check['target_encoders']
                        print(f"Loaded target encoders for columns: {list(target_encoders.keys())}")
                        # Apply the saved encoders
                        for i, col in enumerate(session.target_columns):
                            if col in target_encoders:
                                encoder = target_encoders[col]
                                col_data = y_test_raw[col]
                                print(f"Using saved encoder for {col} with classes: {list(encoder.classes_)}")
                                y_test[:, i] = encoder.transform(col_data)
                    else:
                        print(f"No target_encoders found in model file, available keys: {list(model_data_check.keys())}")
                except Exception as e:
                    print(f"Error loading target encoders: {e}")
            
            # If no saved encoders, create new ones for categorical columns
            if not target_encoders:
                for i, col in enumerate(session.target_columns):
                    col_data = y_test_raw[col]
                    if col_data.dtype == 'object' or col_data.dtype.name == 'category':
                        print(f"Encoding categorical target column: {col}")
                        from sklearn.preprocessing import LabelEncoder
                        encoder = LabelEncoder()
                        # Use the entire dataset column to fit the encoder to get all possible classes
                        full_col_data = df[col]
                        encoder.fit(full_col_data)
                        y_test[:, i] = encoder.transform(col_data)
                        target_encoders[col] = encoder
                        print(f"Unique values in {col}: {encoder.classes_}")
                        print(f"Encoded values sample: {y_test[:5, i]}")
            
            print(f"Final test data shape - X: {X_test.shape}, y: {y_test.shape}")
            
            if X_test.shape[0] == 0:
                print("No test data available - test split too small")
                return {'predictions_analysis': 'No test data available - test split too small'}
            
            # Load model
            if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
                # Use provided model_path or default to session path
                if model_path is None:
                    model_path = session.model_file.path
                    
                print(f"Loading model from: {model_path}")
                model_data = joblib.load(model_path)
                print(f"Model data keys: {model_data.keys()}")
                
                model = model_data['model']
                preprocessing_pipeline = model_data.get('preprocessing_pipeline')
                
                print(f"Model type: {type(model)}")
                print(f"Preprocessing pipeline: {preprocessing_pipeline is not None}")
                
                # Apply preprocessing if available
                if preprocessing_pipeline:
                    X_test_df = pd.DataFrame(X_test, columns=session.predictor_columns)
                    print(f"Before preprocessing: {X_test_df.shape}")
                    X_test = preprocessing_pipeline.transform(X_test_df)
                    print(f"After preprocessing: {X_test.shape}")
                
                # Make predictions
                print(f"Making predictions with X_test shape: {X_test.shape}")
                y_pred = model.predict(X_test)
                print(f"Predictions shape: {y_pred.shape}")
                print(f"Predictions sample: {y_pred[:5] if len(y_pred) > 0 else 'Empty'}")
                
                # Ensure predictions are 2D
                if len(y_pred.shape) == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if len(y_test.shape) == 1:
                    y_test = y_test.reshape(-1, 1)
                    
                print(f"Final shapes - y_test: {y_test.shape}, y_pred: {y_pred.shape}")
                    
            else:
                # Neural networks - simplified for now
                return {'predictions_analysis': 'Neural network analysis not yet implemented'}
            
            # Generate analysis
            analysis = {}
            
            # For each target column
            for i, target_col in enumerate(session.target_columns):
                if i < y_test.shape[1] and i < y_pred.shape[1]:
                    encoder = target_encoders.get(target_col, None)
                    print(f"Analyzing target column {target_col}, encoder: {encoder}")
                    if encoder:
                        print(f"Encoder classes for {target_col}: {list(encoder.classes_)}")
                    col_analysis = self._analyze_predictions(
                        y_test[:, i], y_pred[:, i], target_col, session.model_type, encoder
                    )
                    analysis[f'target_{i}_{target_col}'] = col_analysis
                
                # If this is a classification task without encoder, try to get labels from dataset
                if col_analysis.get('task_type') == 'classification' and col_analysis.get('confusion_matrix'):
                    cm_data = col_analysis['confusion_matrix']
                    if cm_data.get('labels') and all('Class_' in str(label) for label in cm_data['labels']):
                        # Try to get actual labels from the dataset
                        try:
                            unique_values = sorted(df[target_col].unique())
                            if df[target_col].dtype == 'object':
                                # String labels - use them directly
                                cm_data['labels'] = [str(val) for val in unique_values[:len(cm_data['labels'])]]
                                print(f"Updated confusion matrix labels from dataset: {cm_data['labels']}")
                            elif len(unique_values) <= 20:
                                # Numeric but categorical - check for common patterns
                                if target_col.lower() == 'summary' or 'weather' in target_col.lower():
                                    # Try weather mapping
                                    weather_map = {
                                        0: 'Clear', 1: 'Partly Cloudy', 2: 'Mostly Cloudy',
                                        3: 'Overcast', 4: 'Foggy', 5: 'Light Rain',
                                        6: 'Rain', 7: 'Heavy Rain', 8: 'Light Snow',
                                        9: 'Snow', 10: 'Heavy Snow', 11: 'Thunderstorm'
                                    }
                                    cm_data['labels'] = [weather_map.get(int(val), f'Type_{val}') 
                                                       for val in unique_values[:len(cm_data['labels'])]]
                                    print(f"Applied weather mapping to labels: {cm_data['labels']}")
                        except Exception as e:
                            print(f"Could not update labels from dataset: {e}")
            
            return {'predictions_analysis': analysis}
            
        except Exception as e:
            print(f"Error in predictions analysis: {e}")
            return {'predictions_analysis': f'Error: {str(e)}'}
    
    def _analyze_predictions(self, y_true, y_pred, target_name, model_type, label_encoder=None):
        """Analyze predictions for a single target variable"""
        analysis = {
            'target_name': target_name,
            'n_samples': len(y_true),
            'y_true_stats': {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true))
            },
            'y_pred_stats': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            }
        }
        
        # Calculate residuals
        residuals = y_true - y_pred
        analysis['residuals'] = {
            'values': residuals.tolist()[:100],  # First 100 for visualization
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'mae': float(np.mean(np.abs(residuals)))
        }
        
        # Create scatter plot data (sample if too many points)
        n_points = min(500, len(y_true))
        indices = np.random.choice(len(y_true), n_points, replace=False)
        analysis['scatter_data'] = {
            'y_true': y_true[indices].tolist(),
            'y_pred': y_pred[indices].tolist()
        }
        
        # Determine if it's classification or regression
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))
        
        print(f"Target analysis - unique_true: {unique_true}, unique_pred: {unique_pred}")
        print(f"Model type: {model_type}")
        
        # For sklearn models, always try to create confusion matrix if reasonable number of classes
        if model_type in ['random_forest', 'decision_tree', 'xgboost']:
            if unique_true <= 20:  # Increased threshold for more flexibility
                print("Creating confusion matrix for classification task")
                analysis['task_type'] = 'classification'
                analysis['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred, label_encoder)
            else:
                print("Treating as regression task (too many unique values)")
                analysis['task_type'] = 'regression'
                # Still create a simplified confusion matrix for regression (binned)
                analysis['confusion_matrix'] = self._create_regression_confusion_matrix(y_true, y_pred)
        else:
            # Neural networks
            analysis['task_type'] = 'regression'
            
        return analysis
    
    def _calculate_confusion_matrix(self, y_true, y_pred, label_encoder=None):
        """Calculate confusion matrix for classification"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        print(f"[_calculate_confusion_matrix] Called with label_encoder: {label_encoder}")
        
        # Round predictions for classification
        y_pred_rounded = np.round(y_pred).astype(int)
        y_true_int = y_true.astype(int)
        
        cm = confusion_matrix(y_true_int, y_pred_rounded)
        
        # Get unique labels
        unique_labels = sorted(np.unique(np.concatenate([y_true_int, y_pred_rounded])))
        print(f"[_calculate_confusion_matrix] Unique labels found: {unique_labels}")
        
        # Use encoder to get real labels if available
        if label_encoder is not None:
            try:
                print(f"[_calculate_confusion_matrix] Label encoder classes: {list(label_encoder.classes_)}")
                # Get the class names for the unique labels
                label_names = []
                for label in unique_labels:
                    if label < len(label_encoder.classes_):
                        label_names.append(str(label_encoder.classes_[label]))
                    else:
                        label_names.append(f'Unknown_{label}')
                print(f"[_calculate_confusion_matrix] Using encoded labels: {label_names}")
            except Exception as e:
                print(f"[_calculate_confusion_matrix] Error getting label names: {e}")
                import traceback
                traceback.print_exc()
                label_names = [f'Class_{label}' for label in unique_labels]
        else:
            print(f"[_calculate_confusion_matrix] No label encoder provided")
            # Try to infer meaningful labels based on the data
            if len(unique_labels) <= 12:  # Common weather-related classifications
                # Check if these might be weather conditions (0-11 mapping)
                weather_conditions = {
                    0: 'Clear',
                    1: 'Partly Cloudy', 
                    2: 'Mostly Cloudy',
                    3: 'Overcast',
                    4: 'Foggy',
                    5: 'Light Rain',
                    6: 'Rain', 
                    7: 'Heavy Rain',
                    8: 'Light Snow',
                    9: 'Snow',
                    10: 'Heavy Snow',
                    11: 'Thunderstorm'
                }
                
                # Check if labels match weather pattern
                if max(unique_labels) < len(weather_conditions):
                    label_names = [weather_conditions.get(label, f'Condition_{label}') for label in unique_labels]
                    print(f"[_calculate_confusion_matrix] Using inferred weather labels: {label_names}")
                else:
                    label_names = [f'Class_{label}' for label in unique_labels]
            else:
                label_names = [f'Class_{label}' for label in unique_labels]
        
        result = {
            'matrix': cm.tolist(),
            'labels': label_names
        }
        print(f"[_calculate_confusion_matrix] Returning result: {result}")
        return result
    
    def _create_regression_confusion_matrix(self, y_true, y_pred):
        """Create a binned confusion matrix for regression tasks"""
        try:
            # Create bins for continuous values
            n_bins = 5
            
            # Create bins based on true values range
            y_min, y_max = np.min(y_true), np.max(y_true)
            bin_edges = np.linspace(y_min, y_max, n_bins + 1)
            
            # Digitize values into bins
            y_true_binned = np.digitize(y_true, bin_edges) - 1
            y_pred_binned = np.digitize(y_pred, bin_edges) - 1
            
            # Ensure bins are within range
            y_true_binned = np.clip(y_true_binned, 0, n_bins - 1)
            y_pred_binned = np.clip(y_pred_binned, 0, n_bins - 1)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_binned, y_pred_binned, labels=range(n_bins))
            
            # Create bin labels
            bin_labels = []
            for i in range(n_bins):
                start = bin_edges[i]
                end = bin_edges[i + 1]
                bin_labels.append(f"{start:.2f}-{end:.2f}")
            
            return {
                'matrix': cm.tolist(),
                'labels': bin_labels,
                'type': 'regression_binned'
            }
        except Exception as e:
            print(f"Error creating regression confusion matrix: {e}")
            return None
    
    def _analyze_sklearn_model(self, session, model_path=None):
        """Analyze sklearn models (Random Forest, Decision Tree, XGBoost)"""
        analysis = {}
        
        try:
            # Use provided model_path or default to session path
            if model_path is None:
                model_path = session.model_file.path
                # Fix path if needed
                if not os.path.isabs(model_path):
                    from django.conf import settings
                    model_path = os.path.join(settings.MEDIA_ROOT, session.model_file.name)
                    
            print(f"Loading sklearn model from: {model_path}")
            model_data = joblib.load(model_path)
            model = model_data['model']
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = session.predictor_columns
                preprocessing_info = session.preprocessing_info
                
                # Use processed feature names if available
                if preprocessing_info and 'feature_names_after_preprocessing' in preprocessing_info:
                    feature_names = preprocessing_info['feature_names_after_preprocessing']
                
                # Ensure we don't exceed the available feature names
                importances = model.feature_importances_
                n_features = min(len(importances), len(feature_names))
                
                feature_importance = [
                    {
                        'feature': feature_names[i],
                        'importance': float(importances[i])
                    }
                    for i in range(n_features)
                ]
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                analysis['feature_importance'] = feature_importance[:20]  # Top 20
            
            # Model-specific metrics
            if session.model_type == 'random_forest':
                analysis['model_specific'] = {
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'oob_score': getattr(model, 'oob_score_', None)
                }
            elif session.model_type == 'decision_tree':
                analysis['model_specific'] = {
                    'max_depth': model.max_depth,
                    'n_leaves': model.get_n_leaves(),
                    'tree_depth': model.tree_.max_depth
                }
                
        except Exception as e:
            analysis['sklearn_analysis_error'] = str(e)
            
        return analysis
    
    def _analyze_neural_model(self, session):
        """Analyze neural network models"""
        analysis = {}
        
        # Extract information from training history
        if session.training_history:
            history = session.training_history
            analysis['training_curves'] = {
                'epochs': list(range(1, len(history.get('loss', [])) + 1)),
                'loss': history.get('loss', []),
                'val_loss': history.get('val_loss', []),
                'accuracy': history.get('accuracy', []),
                'val_accuracy': history.get('val_accuracy', [])
            }
            
            # Find best epoch
            if 'val_loss' in history and history['val_loss']:
                best_epoch = np.argmin(history['val_loss']) + 1
                analysis['best_epoch'] = {
                    'epoch': int(best_epoch),
                    'val_loss': float(history['val_loss'][best_epoch - 1])
                }
        
        return analysis


class TrainingAnalysisTestView(APIView):
    """Simple test view for debugging"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        """Simple test endpoint"""
        print(f"=== TEST VIEW CALLED for session {pk} ===")
        
        try:
            # Get session
            if request.user.is_staff:
                session = get_object_or_404(TrainingSession, pk=pk)
            else:
                session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
            
            test_data = {
                'message': 'Test endpoint working',
                'session_id': session.id,
                'model_type': session.model_type,
                'status': session.status,
                'model_file_exists': bool(session.model_file and session.model_file.name),
                'predictor_columns': session.predictor_columns,
                'target_columns': session.target_columns,
                'predictions_analysis': {
                    'target_0_test': {
                        'scatter_data': {
                            'y_true': [1, 2, 3, 4, 5],
                            'y_pred': [1.1, 2.2, 2.8, 4.1, 4.9]
                        },
                        'confusion_matrix': {
                            'matrix': [[10, 2], [1, 15]],
                            'labels': ['Class_0', 'Class_1']
                        },
                        'residuals': {
                            'values': [0.1, -0.2, 0.2, -0.1, 0.1]
                        }
                    }
                },
                'feature_importance': [
                    {'feature': 'feature_1', 'importance': 0.3},
                    {'feature': 'feature_2', 'importance': 0.2},
                    {'feature': 'feature_3', 'importance': 0.5}
                ]
            }
            
            print(f"Returning test data: {test_data}")
            return Response(test_data)
            
        except Exception as e:
            print(f"Error in test view: {e}")
            import traceback
            traceback.print_exc()


class DownloadModelView(APIView):
    """Download trained model file"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        """Download the model file for a training session"""
        # Get the training session
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=pk)
        else:
            session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
        
        # Check if session is completed and has a model file
        if session.status != 'completed':
            raise Http404("Training session is not completed")
            
        if not session.model_file:
            raise Http404("No model file available")
        
        # Get the model file path
        if hasattr(session.model_file, 'path'):
            model_path = session.model_file.path
        else:
            # If model_file is stored as a string path
            from django.conf import settings
            model_path = os.path.join(settings.MEDIA_ROOT, str(session.model_file))
        
        # Check if file/directory exists
        if not os.path.exists(model_path):
            raise Http404("Model file not found on disk")
        
        # Handle different model types
        if os.path.isdir(model_path):
            # TensorFlow models are saved as directories, create a zip file
            import zipfile
            import tempfile
            
            # Create temporary zip file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                        zipf.write(file_path, arcname)
            
            filename = f"model_{session.model_type}_{session.id}.zip"
            response = FileResponse(
                open(temp_file.name, 'rb'),
                as_attachment=True,
                filename=filename
            )
            # Clean up temp file after response
            os.unlink(temp_file.name)
            return response
        else:
            # Regular file (RandomForest, PyTorch, etc.)
            if session.model_type == 'random_forest':
                extension = '.pkl'
            elif session.framework == 'pytorch':
                extension = '.pth'
            else:
                extension = os.path.splitext(model_path)[1]
            
            filename = f"model_{session.model_type}_{session.id}{extension}"
            
            # Return file response
            response = FileResponse(
                open(model_path, 'rb'),
                as_attachment=True,
                filename=filename
            )
            return response