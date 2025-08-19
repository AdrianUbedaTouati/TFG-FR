/**
 * Enhanced training progress for different model types
 */

// Model-specific configurations
const MODEL_CONFIGS = {
    random_forest: {
        showEpochs: false,
        metrics: ['n_estimators', 'oob_score', 'feature_importance'],
        logDetails: true,
        progressSteps: [
            'Préparation des données',
            'Encodage des variables catégoriques',
            'Création des caractéristiques cycliques',
            'Construction de la forêt',
            'Entraînement des arbres',
            'Calcul des scores OOB',
            'Évaluation sur validation',
            'Calcul de l\'importance des variables'
        ]
    },
    xgboost: {
        showEpochs: true,
        epochLabel: 'Round',
        metrics: ['train-rmse', 'eval-rmse', 'best_iteration'],
        logDetails: true,
        progressSteps: [
            'Préparation des données',
            'Configuration des paramètres',
            'Début du boosting',
            'Optimisation des arbres',
            'Early stopping check',
            'Finalisation du modèle'
        ]
    },
    decision_tree: {
        showEpochs: false,
        metrics: ['max_depth', 'n_leaves', 'feature_importance'],
        logDetails: true,
        progressSteps: [
            'Préparation des données',
            'Construction de l\'arbre',
            'Recherche des meilleures divisions',
            'Élagage de l\'arbre',
            'Évaluation finale'
        ]
    }
};

// Enhanced log entry with timestamp and styling
function addDetailedLogEntry(message, level = 'info') {
    const logContainer = document.getElementById('training-logs');
    const timestamp = new Date().toLocaleTimeString('fr-FR');
    
    const levelClasses = {
        'info': 'text-info',
        'success': 'text-success',
        'warning': 'text-warning',
        'error': 'text-danger',
        'progress': 'text-primary'
    };
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${levelClasses[level] || 'text-muted'}`;
    entry.innerHTML = `[${timestamp}] ${message}`;
    
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Update UI based on model type
function updateUIForModelType(modelType) {
    const config = MODEL_CONFIGS[modelType] || {};
    
    // Show/hide epoch-related elements
    const epochElements = document.querySelectorAll('.epoch-related');
    epochElements.forEach(el => {
        el.style.display = config.showEpochs ? 'block' : 'none';
    });
    
    // Update metrics display
    if (!config.showEpochs) {
        document.getElementById('metrics-container').style.display = 'none';
        document.getElementById('charts-container').style.display = 'none';
    }
    
    // Add model-specific info
    addDetailedLogEntry(`Mode d'entraînement: ${modelType.toUpperCase()}`, 'info');
    
    if (config.progressSteps) {
        addDetailedLogEntry('Étapes d\'entraînement prévues:', 'info');
        config.progressSteps.forEach((step, idx) => {
            setTimeout(() => {
                addDetailedLogEntry(`${idx + 1}. ${step}`, 'progress');
            }, 100 * idx);
        });
    }
}

// Enhanced progress update for sklearn models
function updateSklearnProgress(data) {
    const progress = data.progress || 0;
    const message = data.message || '';
    const details = data.details || {};
    
    // Update progress bar
    const progressBar = document.getElementById('training-progress');
    const progressText = document.getElementById('progress-text');
    
    progressBar.style.width = `${progress * 100}%`;
    progressText.textContent = `${Math.round(progress * 100)}%`;
    
    // Add detailed log entry
    if (message) {
        addDetailedLogEntry(message, 'progress');
    }
    
    // Show specific details for Random Forest
    if (data.model_type === 'random_forest' && details) {
        if (details.n_estimators) {
            addDetailedLogEntry(`Nombre d'arbres: ${details.n_estimators}`, 'info');
        }
        if (details.current_tree !== undefined) {
            addDetailedLogEntry(`Entraînement de l'arbre ${details.current_tree + 1}/${details.n_estimators}`, 'progress');
        }
        if (details.oob_score !== undefined) {
            addDetailedLogEntry(`Score OOB actuel: ${details.oob_score.toFixed(4)}`, 'success');
        }
        if (details.features_processed) {
            addDetailedLogEntry(`Caractéristiques traitées: ${details.features_processed}`, 'info');
        }
    }
    
    // Show XGBoost specific details
    if (data.model_type === 'xgboost' && details) {
        if (details.iteration !== undefined) {
            addDetailedLogEntry(`Boosting round ${details.iteration}`, 'progress');
        }
        if (details.eval_metric) {
            addDetailedLogEntry(`${details.eval_metric}: ${details.eval_value?.toFixed(6)}`, 'info');
        }
    }
}

// Override the original updateTrainingStatus function
const originalUpdateTrainingStatus = window.updateTrainingStatus;
window.updateTrainingStatus = function(data) {
    // Call original function first
    if (originalUpdateTrainingStatus) {
        originalUpdateTrainingStatus(data);
    }
    
    // Add enhanced logging for sklearn models
    if (data.framework === 'sklearn') {
        updateSklearnProgress(data);
    }
    
    // Update UI based on model type if not already done
    if (data.model_type && !window.modelTypeSet) {
        updateUIForModelType(data.model_type);
        window.modelTypeSet = true;
    }
}

// Enhanced polling function
function enhancedPolling() {
    fetch(`/api/training/${sessionId}/status/`)
        .then(response => response.json())
        .then(data => {
            // Show all log entries from backend
            if (data.log_entries && Array.isArray(data.log_entries)) {
                data.log_entries.forEach(entry => {
                    if (!window.processedLogs.has(entry.id || entry.message)) {
                        addDetailedLogEntry(entry.message, entry.level || 'info');
                        window.processedLogs.add(entry.id || entry.message);
                    }
                });
            }
            
            // Update status with enhanced info
            updateTrainingStatus(data);
            
            // Stop polling if training is complete
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(pollingInterval);
            }
        })
        .catch(error => {
            console.error('Erreur lors de la récupération du statut:', error);
            addDetailedLogEntry('Erreur de connexion au serveur', 'error');
        });
}

// Initialize enhanced features
document.addEventListener('DOMContentLoaded', function() {
    window.processedLogs = new Set();
    window.modelTypeSet = false;
    
    // Start enhanced polling
    if (window.pollingInterval) {
        clearInterval(window.pollingInterval);
    }
    window.pollingInterval = setInterval(enhancedPolling, 1000);
    
    // Initial poll
    enhancedPolling();
});