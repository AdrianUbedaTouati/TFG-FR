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