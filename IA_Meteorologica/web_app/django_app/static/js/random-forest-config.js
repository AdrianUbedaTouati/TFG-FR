/**
 * Random Forest Configuration Manager
 * Handles all UI logic and validation for Random Forest models
 */

// Configuration constants
const RF_PRESETS = {
    rapido: {
        n_estimators: 100,
        max_depth_enabled: false,
        max_depth: null,
        max_features: 'sqrt',
        min_samples_leaf: 1,
        bootstrap: true,
        oob_score: false
    },
    balanceado: {
        n_estimators: 300,
        max_depth_enabled: false,
        max_depth: null,
        max_features: 'auto',
        min_samples_leaf: 1,
        bootstrap: true,
        oob_score: false
    },
    preciso: {
        n_estimators: 1000,
        max_depth_enabled: false,
        max_depth: null,
        max_features: 'auto',
        min_samples_leaf: 1,
        bootstrap: true,
        oob_score: true
    }
};

// Problem type configurations
const PROBLEM_TYPE_CONFIG = {
    classification: {
        defaultCriterion: 'gini',
        criteriaOptions: ['gini', 'entropy', 'log_loss'],
        defaultMaxFeatures: 'sqrt',
        enableClassWeight: true,
        metrics: ['accuracy', 'f1', 'precision', 'recall', 'auc']
    },
    regression: {
        defaultCriterion: 'squared_error',
        criteriaOptions: ['squared_error', 'absolute_error', 'poisson'],
        defaultMaxFeatures: 1.0,
        enableClassWeight: false,
        metrics: ['mse', 'mae', 'rmse', 'r2']
    }
};

/**
 * Initialize Random Forest configuration
 */
function initializeRandomForestConfig() {
    // Set default values
    document.getElementById('rfPreset').value = 'balanceado';
    document.getElementById('rfProblemType').value = 'auto';
    
    // Add event listeners
    setupEventListeners();
    
    // Initialize with default preset
    applyRandomForestPreset();
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Target column changes
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('target-checkbox')) {
            handleTargetColumnChange();
        }
    });
    
    // Validation method changes
    document.getElementById('rfValidationMethod')?.addEventListener('change', handleValidationMethodChange);
    
    // Problem type changes
    document.getElementById('rfProblemType')?.addEventListener('change', updateRandomForestOptions);
}

/**
 * Handle target column selection changes
 */
function handleTargetColumnChange() {
    const modelType = document.getElementById('modelType').value;
    if (modelType !== 'random_forest') return;
    
    const targetCheckboxes = document.querySelectorAll('.target-checkbox:checked');
    const predictorCheckboxes = document.querySelectorAll('.predictor-checkbox');
    
    // Random Forest now supports multi-output (multiple targets)
    // No need to enforce single target anymore
    
    // Remove selected targets from predictors
    if (targetCheckboxes.length >= 1) {
        const targetColumns = Array.from(targetCheckboxes).map(cb => cb.value);
        
        predictorCheckboxes.forEach(cb => {
            if (targetColumns.includes(cb.value)) {
                cb.checked = false;
                cb.disabled = true;
                cb.parentElement.style.opacity = '0.5';
            } else {
                cb.disabled = false;
                cb.parentElement.style.opacity = '1';
            }
        });
        
        // Auto-detect problem type if set to auto
        // For multi-output, use the first target column for detection
        if (document.getElementById('rfProblemType').value === 'auto' && targetCheckboxes.length > 0) {
            detectProblemType(targetCheckboxes[0].value);
        }
        
        // Show info message for multi-output
        if (targetCheckboxes.length > 1) {
            showNotification('Multi-sortie activée', 
                `Random Forest va prédire ${targetCheckboxes.length} variables simultanément`, 
                'info');
        }
    }
}

/**
 * Auto-detect problem type based on target column
 */
async function detectProblemType(targetColumn) {
    const datasetId = document.getElementById('dataset').value;
    if (!datasetId || !targetColumn) return;
    
    try {
        const response = await fetch(`/api/datasets/${datasetId}/column-info/${targetColumn}/`);
        const data = await response.json();
        
        let problemType = 'regression';
        const uniqueValues = data.unique_values || data.n_unique;
        const totalValues = data.total_values || data.count;
        
        // Classification heuristics
        if (uniqueValues <= 2) {
            problemType = 'classification';
            console.log('Binary classification detected');
        } else if (uniqueValues < 20 && uniqueValues < totalValues * 0.05) {
            problemType = 'classification';
            console.log('Multi-class classification detected');
        } else if (data.dtype === 'object' || data.dtype === 'category') {
            problemType = 'classification';
            console.log('Categorical target detected');
        }
        
        document.getElementById('rfProblemType').value = problemType;
        updateRandomForestOptions();
        
        showNotification('Détection automatique', 
            `Type de problème détecté: ${problemType === 'classification' ? 'Classification' : 'Régression'}`, 
            'info');
    } catch (error) {
        console.error('Error detecting problem type:', error);
    }
}

/**
 * Apply selected preset configuration
 */
function applyRandomForestPreset() {
    const preset = document.getElementById('rfPreset').value;
    const config = RF_PRESETS[preset];
    
    if (config) {
        // Apply configuration
        document.getElementById('rfNEstimators').value = config.n_estimators;
        document.getElementById('rfNEstimatorsValue').textContent = config.n_estimators;
        
        document.getElementById('rfMaxDepthEnabled').checked = config.max_depth_enabled;
        document.getElementById('rfMaxDepth').disabled = !config.max_depth_enabled;
        if (config.max_depth) {
            document.getElementById('rfMaxDepth').value = config.max_depth;
        }
        
        // Handle max_features
        if (config.max_features === 'auto') {
            // Will be set based on problem type
            updateMaxFeaturesDefault();
        } else {
            document.getElementById('rfMaxFeatures').value = config.max_features;
        }
        
        document.getElementById('rfMinSamplesLeaf').value = config.min_samples_leaf;
        document.getElementById('rfBootstrap').checked = config.bootstrap;
        document.getElementById('rfOobScore').checked = config.oob_score;
        
        // Update dependent options
        toggleBootstrapOptions();
        toggleCustomMaxFeatures();
    }
}

/**
 * Update Random Forest options based on problem type
 */
function updateRandomForestOptions() {
    const problemType = document.getElementById('rfProblemType').value;
    
    if (problemType === 'auto') {
        // Keep current settings, will be determined at training time
        return;
    }
    
    const config = PROBLEM_TYPE_CONFIG[problemType];
    
    // Show/hide classification-specific options
    const classificationSettings = document.getElementById('rfClassificationSettings');
    if (classificationSettings) {
        classificationSettings.style.display = config.enableClassWeight ? 'block' : 'none';
    }
    
    // Update criterion options
    updateCriterionOptions(problemType, config);
    
    // Update max_features default
    updateMaxFeaturesDefault();
    
    // Update validation options
    updateValidationOptions(problemType);
}

/**
 * Update criterion dropdown based on problem type
 */
function updateCriterionOptions(problemType, config) {
    const criterionSelect = document.getElementById('rfCriterion');
    criterionSelect.innerHTML = '<option value="auto">Automatique (recommandé)</option>';
    
    config.criteriaOptions.forEach(criterion => {
        const option = document.createElement('option');
        option.value = criterion;
        option.textContent = getCriterionLabel(criterion, problemType);
        criterionSelect.appendChild(option);
    });
    
    // Set default if not auto
    if (criterionSelect.value !== 'auto') {
        criterionSelect.value = config.defaultCriterion;
    }
}

/**
 * Get human-readable label for criterion
 */
function getCriterionLabel(criterion, problemType) {
    const labels = {
        // Classification
        'gini': 'Gini (par défaut)',
        'entropy': 'Entropie',
        'log_loss': 'Log Loss',
        // Regression
        'squared_error': 'Erreur quadratique (MSE)',
        'absolute_error': 'Erreur absolue (MAE)',
        'poisson': 'Poisson (pour comptes)'
    };
    return labels[criterion] || criterion;
}

/**
 * Update max_features default based on problem type
 */
function updateMaxFeaturesDefault() {
    const problemType = document.getElementById('rfProblemType').value;
    const maxFeatures = document.getElementById('rfMaxFeatures');
    
    if (maxFeatures.value === 'auto' || !maxFeatures.value) {
        if (problemType === 'classification') {
            maxFeatures.value = 'sqrt';
        } else if (problemType === 'regression') {
            maxFeatures.value = '1.0';
        }
    }
}

/**
 * Update validation options based on problem type
 */
function updateValidationOptions(problemType) {
    const validationMethod = document.getElementById('rfValidationMethod');
    const currentValue = validationMethod.value;
    
    validationMethod.innerHTML = '';
    
    // Common options
    const options = [
        { value: 'holdout', label: 'Hold-out 80/20' },
        { value: 'cv', label: 'Validation croisée (5 folds)' },
        { value: 'oob', label: 'Out-of-Bag (OOB)' }
    ];
    
    // Add stratified option for classification
    if (problemType === 'classification') {
        options.splice(1, 0, { value: 'stratified', label: 'Hold-out stratifié 80/20' });
        options[2] = { value: 'stratified_cv', label: 'Validation croisée stratifiée (5 folds)' };
    }
    
    // Add time series option if temporal columns detected
    if (hasTemporalColumns()) {
        options.push({ value: 'time_series', label: 'Time Series Split' });
    }
    
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.label;
        validationMethod.appendChild(option);
    });
    
    // Restore previous value if possible
    if (Array.from(validationMethod.options).some(opt => opt.value === currentValue)) {
        validationMethod.value = currentValue;
    }
    
    handleValidationMethodChange();
}

/**
 * Check if dataset has temporal columns
 */
function hasTemporalColumns() {
    // Check predictor columns for temporal indicators
    const predictorCheckboxes = document.querySelectorAll('.predictor-checkbox');
    return Array.from(predictorCheckboxes).some(cb => {
        const colName = cb.value.toLowerCase();
        return colName.includes('date') || colName.includes('time') || 
               colName.includes('year') || colName.includes('month');
    });
}

/**
 * Handle validation method changes
 */
function handleValidationMethodChange() {
    const method = document.getElementById('rfValidationMethod').value;
    const oobCheckbox = document.getElementById('rfOobScore');
    const bootstrapCheckbox = document.getElementById('rfBootstrap');
    
    if (method === 'oob') {
        // Force bootstrap and OOB to be enabled
        bootstrapCheckbox.checked = true;
        oobCheckbox.checked = true;
        bootstrapCheckbox.disabled = true;
        showNotification('Info', 'Bootstrap et OOB score activés pour la validation OOB', 'info');
    } else {
        // Re-enable bootstrap control
        bootstrapCheckbox.disabled = false;
        toggleBootstrapOptions();
    }
}

/**
 * Toggle Random Forest max depth
 */
function toggleRandomForestMaxDepth() {
    const enabled = document.getElementById('rfMaxDepthEnabled').checked;
    document.getElementById('rfMaxDepth').disabled = !enabled;
}

/**
 * Toggle custom max features input
 */
function toggleCustomMaxFeatures() {
    const maxFeatures = document.getElementById('rfMaxFeatures').value;
    const isCustom = maxFeatures === 'custom';
    
    document.getElementById('rfMaxFeaturesFraction').style.display = isCustom ? 'block' : 'none';
    document.getElementById('rfMaxFeaturesFractionContainer').style.display = isCustom ? 'flex' : 'none';
}

/**
 * Toggle bootstrap-dependent options
 */
function toggleBootstrapOptions() {
    const bootstrap = document.getElementById('rfBootstrap').checked;
    const oobContainer = document.getElementById('rfOobScoreContainer');
    const oobCheckbox = document.getElementById('rfOobScore');
    
    if (!bootstrap) {
        oobCheckbox.checked = false;
        oobContainer.style.opacity = '0.5';
        oobCheckbox.disabled = true;
    } else {
        oobContainer.style.opacity = '1';
        oobCheckbox.disabled = false;
    }
}

/**
 * Get Random Forest configuration
 */
function getRandomForestConfig() {
    const problemType = document.getElementById('rfProblemType').value;
    
    const config = {
        preset: document.getElementById('rfPreset').value,
        problem_type: problemType,
        n_estimators: parseInt(document.getElementById('rfNEstimators').value),
        max_depth_enabled: document.getElementById('rfMaxDepthEnabled').checked,
        max_depth: document.getElementById('rfMaxDepthEnabled').checked ? 
            parseInt(document.getElementById('rfMaxDepth').value) : null,
        max_features: document.getElementById('rfMaxFeatures').value === 'custom' ? 
            parseFloat(document.getElementById('rfMaxFeaturesFraction').value) : 
            document.getElementById('rfMaxFeatures').value,
        criterion: document.getElementById('rfCriterion').value,
        validation_method: document.getElementById('rfValidationMethod').value,
        
        // Advanced options
        min_samples_split: parseInt(document.getElementById('rfMinSamplesSplit').value),
        min_samples_leaf: parseInt(document.getElementById('rfMinSamplesLeaf').value),
        min_weight_fraction_leaf: parseFloat(document.getElementById('rfMinWeightFractionLeaf').value),
        min_impurity_decrease: parseFloat(document.getElementById('rfMinImpurityDecrease').value),
        bootstrap: document.getElementById('rfBootstrap').checked,
        oob_score: document.getElementById('rfOobScore').checked,
        n_jobs: parseInt(document.getElementById('rfNJobs').value),
        random_state: document.getElementById('rfRandomState').value ? 
            parseInt(document.getElementById('rfRandomState').value) : null,
        
    };
    
    // Classification-specific options
    if (problemType === 'classification' || problemType === 'auto') {
        const classWeightSelect = document.getElementById('rfClassWeight');
        if (classWeightSelect) {
            config.class_weight = classWeightSelect.value || null;
        }
        config.decision_threshold = parseFloat(document.getElementById('rfDecisionThreshold')?.value ?? 0.5);
    }
    
    return config;
}

/**
 * Validate Random Forest configuration
 */
function validateRandomForestConfig() {
    const errors = [];
    const warnings = [];
    
    // Check target selection
    const targetCheckboxes = document.querySelectorAll('.target-checkbox:checked');
    if (targetCheckboxes.length === 0) {
        errors.push('Sélectionnez au moins une variable cible');
    }
    
    // Check predictor selection
    const predictorCheckboxes = document.querySelectorAll('.predictor-checkbox:checked');
    if (predictorCheckboxes.length === 0) {
        errors.push('Sélectionnez au moins une variable prédictive');
    }
    
    // Check for overlap
    if (targetCheckboxes.length === 1 && predictorCheckboxes.length > 0) {
        const targetCol = targetCheckboxes[0].value;
        const predictorCols = Array.from(predictorCheckboxes).map(cb => cb.value);
        if (predictorCols.includes(targetCol)) {
            errors.push('Une variable ne peut pas être à la fois cible et prédictive');
        }
    }
    
    // Sample size warnings
    const datasetInfo = getDatasetInfo();
    if (datasetInfo && datasetInfo.row_count < 100) {
        warnings.push('Dataset très petit (<100 échantillons). Les résultats peuvent être instables.');
    }
    
    // OOB vs validation method
    const validationMethod = document.getElementById('rfValidationMethod').value;
    const oobScore = document.getElementById('rfOobScore').checked;
    if (validationMethod === 'oob' && !oobScore) {
        errors.push('La validation OOB nécessite que le score OOB soit activé');
    }
    
    return { errors, warnings };
}

/**
 * Get dataset information (mock function - replace with actual API call)
 */
function getDatasetInfo() {
    // This should be replaced with actual dataset info from API
    return {
        row_count: 1000,
        column_count: 20
    };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('randomForestConfiguration')) {
        initializeRandomForestConfig();
    }
});