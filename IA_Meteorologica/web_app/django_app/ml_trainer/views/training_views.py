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
        # Verificar permisos
        if request.user.is_staff:
            session = get_object_or_404(TrainingSession, pk=pk)
        else:
            session = get_object_or_404(TrainingSession, pk=pk, user=request.user)
            
        if session.status != 'completed':
            return error_response("Training session is not completed")
            
        try:
            analysis_data = self._generate_analysis(session)
            return Response(analysis_data)
        except Exception as e:
            return error_response(f"Error generating analysis: {str(e)}")
    
    def _generate_analysis(self, session):
        """Generate comprehensive analysis based on model type"""
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
        
        # Load model and make predictions for analysis
        if session.model_file and os.path.exists(session.model_file.path):
            try:
                predictions_data = self._generate_predictions_analysis(session)
                analysis.update(predictions_data)
            except Exception as e:
                print(f"Error generating predictions analysis: {e}")
        
        # Add model-specific analysis
        if session.model_type in ['random_forest', 'decision_tree', 'xgboost']:
            analysis.update(self._analyze_sklearn_model(session))
        elif session.model_type in ['lstm', 'gru', 'cnn']:
            analysis.update(self._analyze_neural_model(session))
            
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
    
    def _generate_predictions_analysis(self, session):
        """Generate predictions vs actual analysis"""
        try:
            # Load the dataset
            df = pd.read_csv(session.dataset.file.path)
            
            # Prepare data splits (similar to prepare_data in ml_utils)
            n = len(df)
            train_end = int(n * session.train_split)
            val_end = int(n * (session.train_split + session.val_split))
            
            # Get test data
            X_test = df[session.predictor_columns].iloc[val_end:].values
            y_test = df[session.target_columns].iloc[val_end:].values
            
            # Load model
            if session.model_type in ['decision_tree', 'random_forest', 'xgboost']:
                model_data = joblib.load(session.model_file.path)
                model = model_data['model']
                preprocessing_pipeline = model_data.get('preprocessing_pipeline')
                
                # Apply preprocessing if available
                if preprocessing_pipeline:
                    X_test_df = pd.DataFrame(X_test, columns=session.predictor_columns)
                    X_test = preprocessing_pipeline.transform(X_test_df)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Ensure predictions are 2D
                if len(y_pred.shape) == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if len(y_test.shape) == 1:
                    y_test = y_test.reshape(-1, 1)
                    
            else:
                # Neural networks - simplified for now
                return {'predictions_analysis': 'Neural network analysis not yet implemented'}
            
            # Generate analysis
            analysis = {}
            
            # For each target column
            for i, target_col in enumerate(session.target_columns):
                if i < y_test.shape[1] and i < y_pred.shape[1]:
                    col_analysis = self._analyze_predictions(
                        y_test[:, i], y_pred[:, i], target_col, session.model_type
                    )
                    analysis[f'target_{i}_{target_col}'] = col_analysis
            
            return {'predictions_analysis': analysis}
            
        except Exception as e:
            print(f"Error in predictions analysis: {e}")
            return {'predictions_analysis': f'Error: {str(e)}'}
    
    def _analyze_predictions(self, y_true, y_pred, target_name, model_type):
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
        if unique_true <= 10 and model_type in ['random_forest', 'decision_tree', 'xgboost']:
            # Likely classification
            analysis['task_type'] = 'classification'
            analysis['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred)
        else:
            # Regression
            analysis['task_type'] = 'regression'
            
        return analysis
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix for classification"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Round predictions for classification
        y_pred_rounded = np.round(y_pred).astype(int)
        y_true_int = y_true.astype(int)
        
        cm = confusion_matrix(y_true_int, y_pred_rounded)
        
        return {
            'matrix': cm.tolist(),
            'labels': sorted(np.unique(np.concatenate([y_true_int, y_pred_rounded]))).tolist()
        }
    
    def _analyze_sklearn_model(self, session):
        """Analyze sklearn models (Random Forest, Decision Tree, XGBoost)"""
        analysis = {}
        
        try:
            model_data = joblib.load(session.model_file.path)
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