"""
Custom callbacks for tracking training progress in real-time
"""
import tensorflow as tf
from tensorflow import keras
import time
from datetime import datetime


class SessionProgressCallback(keras.callbacks.Callback):
    """
    Custom callback to update TrainingSession with real-time progress
    """
    
    def __init__(self, session, total_epochs, total_batches_per_epoch):
        super().__init__()
        self.session = session
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        self.start_time = time.time()
        
        # Initialize session
        self.session.total_epochs = total_epochs
        self.session.save()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.session.current_epoch = epoch + 1
        self.session.save()
        self._add_log(f"Époque {epoch + 1}/{self.total_epochs} démarrée")
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        # Update batch progress
        self.session.current_batch = batch + 1
        
        # Calculate overall progress
        epoch_progress = (self.session.current_epoch - 1) / self.total_epochs
        batch_progress = (batch + 1) / (self.total_batches_per_epoch * self.total_epochs)
        self.session.progress = epoch_progress + batch_progress
        
        # Update loss if available
        if logs:
            if 'loss' in logs:
                self.session.train_loss = float(logs['loss'])
            if 'accuracy' in logs:
                self.session.train_accuracy = float(logs['accuracy'])
        
        # Save every 10 batches to avoid too many DB writes
        if batch % 10 == 0:
            self.session.save()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        # Update metrics
        if logs:
            if 'loss' in logs:
                self.session.train_loss = float(logs['loss'])
            if 'val_loss' in logs:
                self.session.val_loss = float(logs['val_loss'])
            if 'accuracy' in logs:
                self.session.train_accuracy = float(logs['accuracy'])
            if 'val_accuracy' in logs:
                self.session.val_accuracy = float(logs['val_accuracy'])
        
        # Update progress
        self.session.progress = (epoch + 1) / self.total_epochs
        
        # Add log entry
        metrics_str = []
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics_str.append(f"{key}: {value:.4f}")
        
        self._add_log(f"Époque {epoch + 1}/{self.total_epochs} terminée - {', '.join(metrics_str)}")
        
        # Save to database
        self.session.save()
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self._add_log("Démarrage de l'entraînement du modèle...")
        self.session.current_epoch = 0
        self.session.current_batch = 0
        self.session.progress = 0.0
        self.session.save()
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self._add_log(f"Entraînement terminé en {minutes}m {seconds}s")
        self.session.progress = 1.0
        self.session.save()
    
    def _add_log(self, message):
        """Add a log entry to the session"""
        if not isinstance(self.session.training_logs, list):
            self.session.training_logs = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        self.session.training_logs.append(log_entry)
        
        # Keep only last 100 logs to avoid too much data
        if len(self.session.training_logs) > 100:
            self.session.training_logs = self.session.training_logs[-100:]


class SklearnProgressCallback:
    """
    Progress tracking for sklearn models during training
    """
    
    def __init__(self, session, estimator_name='model'):
        self.session = session
        self.estimator_name = estimator_name
        self.start_time = time.time()
        
    def on_train_begin(self):
        """Called at the beginning of training"""
        self._add_log(f"Démarrage de l'entraînement {self.estimator_name}...")
        self.session.progress = 0.0
        self.session.save()
    
    def on_train_end(self):
        """Called at the end of training"""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self._add_log(f"Entraînement {self.estimator_name} terminé en {minutes}m {seconds}s")
        self.session.progress = 1.0
        self.session.save()
    
    def update_progress(self, progress, message=None):
        """Update progress manually"""
        self.session.progress = progress
        if message:
            self._add_log(message)
        # Save specific fields to avoid overwriting concurrent updates
        fields_to_update = ['progress', 'training_logs'] if message else ['progress']
        self.session.save(update_fields=fields_to_update)
    
    def update_message(self, message):
        """Update only the message without changing progress"""
        self._add_log(message)
        # Save without updating modified timestamp to avoid race conditions
        self.session.save(update_fields=['training_logs'])
    
    def log_message(self, message, save_immediately=True):
        """Add a log message without updating progress"""
        self._add_log(message)
        if save_immediately:
            self.session.save()
    
    def _add_log(self, message):
        """Add a log entry to the session"""
        if not isinstance(self.session.training_logs, list):
            self.session.training_logs = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        self.session.training_logs.append(log_entry)
        
        # Keep only last 100 logs
        if len(self.session.training_logs) > 100:
            self.session.training_logs = self.session.training_logs[-100:]