"""
Training session management views
"""
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
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
            'model_type': session.model_type,
            'framework': getattr(session, 'framework', 'keras')
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