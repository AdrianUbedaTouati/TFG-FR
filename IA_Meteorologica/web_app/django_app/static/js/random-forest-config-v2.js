/**
 * Random Forest Configuration Manager V2
 * Sistema robusto de persistencia y edición
 */

// Estado global para mantener configuraciones del usuario
const RFUserState = {
    // Valores modificados por el usuario (no sobrescribir con presets)
    userModified: {
        n_estimators: false,
        criterion: false,
        max_depth: false,
        max_features: false,
        min_samples_split: false,
        min_samples_leaf: false,
        // Agregar más según necesidad
    },
    
    // Valores actuales
    currentValues: {},
    
    // Marcar un valor como modificado por el usuario
    markAsUserModified: function(field) {
        this.userModified[field] = true;
    },
    
    // Resetear estado de modificación
    resetModificationState: function() {
        Object.keys(this.userModified).forEach(key => {
            this.userModified[key] = false;
        });
    },
    
    // Guardar valor actual
    saveValue: function(field, value) {
        this.currentValues[field] = value;
    },
    
    // Obtener valor actual
    getValue: function(field) {
        return this.currentValues[field];
    },
    
    // Verificar si un campo fue modificado por el usuario
    isUserModified: function(field) {
        return this.userModified[field] === true;
    }
};

// Configuration constants
const RF_PRESETS = {
    rapido: {
        n_estimators: 100,
        max_depth_enabled: false,
        max_depth: null,
        max_features: 'sqrt',
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_weight_fraction_leaf: 0,
        min_impurity_decrease: 0,
        bootstrap: true,
        oob_score: false,
        n_jobs: -1,
        random_state: 42
    },
    balanceado: {
        n_estimators: 300,
        max_depth_enabled: false,
        max_depth: null,
        max_features: 'auto',
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_weight_fraction_leaf: 0,
        min_impurity_decrease: 0,
        bootstrap: true,
        oob_score: false,
        n_jobs: -1,
        random_state: 42
    },
    preciso: {
        n_estimators: 1000,
        max_depth_enabled: true,
        max_depth: 20,
        max_features: 'auto',
        min_samples_split: 5,
        min_samples_leaf: 2,
        min_weight_fraction_leaf: 0,
        min_impurity_decrease: 0,
        bootstrap: true,
        oob_score: true,
        n_jobs: -1,
        random_state: 42
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
    // Reset modification state for new model
    RFUserState.resetModificationState();
    
    // Set default values
    document.getElementById('rfPreset').value = 'balanceado';
    document.getElementById('rfProblemType').value = 'auto';
    
    // Add event listeners with state tracking
    setupEventListenersWithTracking();
    
    // Initialize with default preset (sin sobrescribir valores modificados)
    applyRandomForestPreset();
}

/**
 * Setup all event listeners with state tracking
 */
function setupEventListenersWithTracking() {
    // Track user modifications
    
    // Número de árboles
    const nEstimatorsInput = document.getElementById('rfNEstimators');
    if (nEstimatorsInput) {
        nEstimatorsInput.addEventListener('input', function() {
            RFUserState.markAsUserModified('n_estimators');
            RFUserState.saveValue('n_estimators', this.value);
            document.getElementById('rfNEstimatorsValue').textContent = this.value;
        });
    }
    
    // Criterio
    const criterionSelect = document.getElementById('rfCriterion');
    if (criterionSelect) {
        criterionSelect.addEventListener('change', function() {
            if (this.value !== 'auto') {
                RFUserState.markAsUserModified('criterion');
                RFUserState.saveValue('criterion', this.value);
            }
        });
    }
    
    // Max depth
    const maxDepthInput = document.getElementById('rfMaxDepth');
    if (maxDepthInput) {
        maxDepthInput.addEventListener('input', function() {
            RFUserState.markAsUserModified('max_depth');
            RFUserState.saveValue('max_depth', this.value);
        });
    }
    
    // Max features
    const maxFeaturesSelect = document.getElementById('rfMaxFeatures');
    if (maxFeaturesSelect) {
        maxFeaturesSelect.addEventListener('change', function() {
            RFUserState.markAsUserModified('max_features');
            RFUserState.saveValue('max_features', this.value);
        });
    }
    
    // Min samples split
    const minSamplesSplitInput = document.getElementById('rfMinSamplesSplit');
    if (minSamplesSplitInput) {
        minSamplesSplitInput.addEventListener('input', function() {
            RFUserState.markAsUserModified('min_samples_split');
            RFUserState.saveValue('min_samples_split', this.value);
        });
    }
    
    // Min samples leaf
    const minSamplesLeafInput = document.getElementById('rfMinSamplesLeaf');
    if (minSamplesLeafInput) {
        minSamplesLeafInput.addEventListener('input', function() {
            RFUserState.markAsUserModified('min_samples_leaf');
            RFUserState.saveValue('min_samples_leaf', this.value);
        });
    }
    
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
 * Apply selected preset configuration
 */
function applyRandomForestPreset() {
    const preset = document.getElementById('rfPreset').value;
    const config = RF_PRESETS[preset];
    
    if (config) {
        // Aplicar TODOS los valores del preset
        document.getElementById('rfNEstimators').value = config.n_estimators;
        document.getElementById('rfNEstimatorsValue').textContent = config.n_estimators;
        
        // Max depth
        document.getElementById('rfMaxDepthEnabled').checked = config.max_depth_enabled;
        document.getElementById('rfMaxDepth').disabled = !config.max_depth_enabled;
        if (config.max_depth) {
            document.getElementById('rfMaxDepth').value = config.max_depth;
        }
        
        // Max features
        if (config.max_features === 'auto') {
            updateMaxFeaturesDefault();
        } else {
            document.getElementById('rfMaxFeatures').value = config.max_features;
        }
        
        // Parámetros de división
        document.getElementById('rfMinSamplesSplit').value = config.min_samples_split;
        document.getElementById('rfMinSamplesLeaf').value = config.min_samples_leaf;
        document.getElementById('rfMinWeightFractionLeaf').value = config.min_weight_fraction_leaf;
        document.getElementById('rfMinImpurityDecrease').value = config.min_impurity_decrease;
        
        // Bootstrap y OOB
        document.getElementById('rfBootstrap').checked = config.bootstrap;
        document.getElementById('rfOobScore').checked = config.oob_score;
        
        // Performance
        document.getElementById('rfNJobs').value = config.n_jobs;
        if (config.random_state) {
            document.getElementById('rfRandomState').value = config.random_state;
        }
        
        // Update dependent options
        toggleBootstrapOptions();
        toggleCustomMaxFeatures();
        toggleRandomForestMaxDepth();
        
        console.log(`Preset "${preset}" aplicado con éxito`);
    }
}

/**
 * Update Random Forest options based on problem type (mejorado)
 */
function updateRandomForestOptions() {
    const problemType = document.getElementById('rfProblemType').value;
    
    if (problemType === 'auto') {
        return;
    }
    
    const config = PROBLEM_TYPE_CONFIG[problemType];
    
    // Show/hide classification-specific options
    const classificationSettings = document.getElementById('rfClassificationSettings');
    if (classificationSettings) {
        classificationSettings.style.display = config.enableClassWeight ? 'block' : 'none';
    }
    
    // Update criterion options (preservando selección del usuario)
    updateCriterionOptionsPreserving(problemType, config);
    
    // Update max_features default (solo si no fue modificado)
    if (!RFUserState.isUserModified('max_features')) {
        updateMaxFeaturesDefault();
    }
    
    // Update validation options
    updateValidationOptions(problemType);
}

/**
 * Update criterion dropdown preserving user selection
 */
function updateCriterionOptionsPreserving(problemType, config) {
    const criterionSelect = document.getElementById('rfCriterion');
    const currentValue = criterionSelect.value;
    const userModifiedValue = RFUserState.getValue('criterion');
    
    // Guardar el valor actual antes de reconstruir
    let valueToRestore = currentValue;
    
    // Si hay un valor modificado por el usuario, usarlo
    if (RFUserState.isUserModified('criterion') && userModifiedValue) {
        valueToRestore = userModifiedValue;
    }
    
    // Rebuild options
    criterionSelect.innerHTML = '<option value="auto">Automatique (recommandé)</option>';
    
    config.criteriaOptions.forEach(criterion => {
        const option = document.createElement('option');
        option.value = criterion;
        option.textContent = getCriterionLabel(criterion, problemType);
        criterionSelect.appendChild(option);
    });
    
    // Intentar restaurar el valor
    if (valueToRestore && valueToRestore !== 'auto') {
        // Verificar si el valor es válido para el tipo de problema actual
        if (config.criteriaOptions.includes(valueToRestore)) {
            criterionSelect.value = valueToRestore;
        } else {
            // Si no es válido pero existe, mantener 'auto' sin marcar como modificado
            criterionSelect.value = 'auto';
            if (RFUserState.isUserModified('criterion')) {
                RFUserState.userModified.criterion = false;
            }
        }
    } else {
        // Mantener auto si ese era el valor
        criterionSelect.value = 'auto';
    }
}

/**
 * Handle target column selection changes
 */
function handleTargetColumnChange() {
    const modelType = document.getElementById('modelType').value;
    if (modelType !== 'random_forest') return;
    
    const targetCheckboxes = document.querySelectorAll('.target-checkbox:checked');
    const predictorCheckboxes = document.querySelectorAll('.predictor-checkbox');
    
    // Random Forest supports multi-output
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
        } else if (uniqueValues < 20 && uniqueValues < totalValues * 0.05) {
            problemType = 'classification';
        } else if (data.dtype === 'object' || data.dtype === 'category') {
            problemType = 'classification';
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
 * Load Random Forest configuration for editing (mejorado)
 */
function loadRandomForestConfiguration(hyperparams) {
    if (!hyperparams) return;
    
    console.log('Loading Random Forest configuration:', hyperparams);
    
    // Reset user modification state when loading existing model
    RFUserState.resetModificationState();
    
    // Load all values
    setValueWithRangeUpdate('rfPreset', hyperparams.preset || 'balanceado');
    setValueWithRangeUpdate('rfProblemType', hyperparams.problem_type || 'auto');
    setValueWithRangeUpdate('rfNEstimators', hyperparams.n_estimators || 300, 'rfNEstimatorsValue');
    
    // Max depth
    setCheckboxValue('rfMaxDepthEnabled', hyperparams.max_depth_enabled || false, toggleRandomForestMaxDepth);
    if (hyperparams.max_depth) {
        setValueWithRangeUpdate('rfMaxDepth', hyperparams.max_depth);
    }
    
    // Max features
    if (hyperparams.max_features) {
        if (typeof hyperparams.max_features === 'number') {
            setValueWithRangeUpdate('rfMaxFeatures', 'custom');
            setValueWithRangeUpdate('rfMaxFeaturesFraction', hyperparams.max_features, 'rfMaxFeaturesFractionValue');
        } else {
            setValueWithRangeUpdate('rfMaxFeatures', hyperparams.max_features);
        }
        toggleCustomMaxFeatures();
    }
    
    // Other parameters - NO usar valores por defecto para criterion
    if (hyperparams.criterion !== undefined) {
        setValueWithRangeUpdate('rfCriterion', hyperparams.criterion);
    }
    setValueWithRangeUpdate('rfMinSamplesSplit', hyperparams.min_samples_split || 2);
    setValueWithRangeUpdate('rfMinSamplesLeaf', hyperparams.min_samples_leaf || 1);
    setValueWithRangeUpdate('rfMinWeightFractionLeaf', hyperparams.min_weight_fraction_leaf || 0);
    setValueWithRangeUpdate('rfMinImpurityDecrease', hyperparams.min_impurity_decrease || 0);
    
    // Bootstrap and OOB
    setCheckboxValue('rfBootstrap', hyperparams.bootstrap !== false, toggleBootstrapOptions);
    setCheckboxValue('rfOobScore', hyperparams.oob_score || false);
    
    // Performance
    setValueWithRangeUpdate('rfNJobs', hyperparams.n_jobs || -1);
    if (hyperparams.random_state !== null && hyperparams.random_state !== undefined) {
        setValueWithRangeUpdate('rfRandomState', hyperparams.random_state);
    }
    
    // Classification settings
    if (hyperparams.class_weight) {
        setValueWithRangeUpdate('rfClassWeight', hyperparams.class_weight);
    }
    if (hyperparams.decision_threshold !== undefined) {
        setValueWithRangeUpdate('rfDecisionThreshold', hyperparams.decision_threshold, 'rfDecisionThresholdValue');
    }
    
    // Validation method
    setValueWithRangeUpdate('rfValidationMethod', hyperparams.validation_method || 'holdout');
    
    // Update UI based on problem type
    updateRandomForestOptions();
    
    // IMPORTANTE: Después de actualizar opciones, restaurar el criterio si existe
    if (hyperparams.criterion && hyperparams.criterion !== 'auto') {
        setTimeout(() => {
            const criterionSelect = document.getElementById('rfCriterion');
            if (criterionSelect) {
                // Verificar si la opción existe
                const optionExists = Array.from(criterionSelect.options).some(opt => opt.value === hyperparams.criterion);
                if (optionExists) {
                    criterionSelect.value = hyperparams.criterion;
                    console.log('Criterion restored to:', hyperparams.criterion);
                } else {
                    console.warn('Criterion not valid for current problem type:', hyperparams.criterion);
                }
            }
        }, 100);
    }
    
    // Guardar valores cargados en el estado
    RFUserState.currentValues = { ...hyperparams };
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
 * Get dataset information
 */
function getDatasetInfo() {
    // This should be replaced with actual dataset info from API
    return {
        row_count: 1000,
        column_count: 20
    };
}

// Helper functions
function updateRangeValue(rangeId, displayId) {
    const range = document.getElementById(rangeId);
    const display = document.getElementById(displayId);
    if (range && display) {
        display.textContent = range.value;
    }
}

function setValueWithRangeUpdate(elementId, value, rangeDisplayId) {
    const element = document.getElementById(elementId);
    if (element && value !== undefined && value !== null) {
        element.value = value;
        if (rangeDisplayId) {
            updateRangeValue(elementId, rangeDisplayId);
        }
    }
}

function setCheckboxValue(elementId, value, callback) {
    const element = document.getElementById(elementId);
    if (element && value !== undefined) {
        element.checked = value;
        if (callback && typeof callback === 'function') {
            callback();
        }
    }
}

/**
 * Reset to default preset (balanceado)
 */
function resetRandomForestPreset() {
    // Reset modificaciones del usuario
    RFUserState.resetModificationState();
    
    // Establecer preset balanceado
    document.getElementById('rfPreset').value = 'balanceado';
    applyRandomForestPreset();
    
    showNotification('Restablecido', 'Configuración predefinida restablecida a "Balanceado"', 'info');
}

/**
 * Optimize configuration for current problem type
 */
function resetRandomForestProblemType() {
    const problemType = document.getElementById('rfProblemType').value;
    
    if (problemType === 'auto') {
        showNotification('Información', 'Seleccione primero un tipo de problema específico (Clasificación o Regresión)', 'warning');
        return;
    }
    
    // Configuraciones óptimas para cada tipo
    const optimalConfigs = {
        classification: {
            n_estimators: 500,
            max_depth_enabled: true,
            max_depth: 20,
            max_features: 'sqrt',
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: 'gini',
            bootstrap: true,
            oob_score: true,
            class_weight: 'balanced',
            validation_method: 'stratified_cv'
        },
        regression: {
            n_estimators: 500,
            max_depth_enabled: true,
            max_depth: 25,
            max_features: 1.0,
            min_samples_split: 5,
            min_samples_leaf: 2,
            criterion: 'squared_error',
            bootstrap: true,
            oob_score: true,
            validation_method: 'cv'
        }
    };
    
    const config = optimalConfigs[problemType];
    if (!config) return;
    
    // Aplicar configuración óptima
    document.getElementById('rfNEstimators').value = config.n_estimators;
    document.getElementById('rfNEstimatorsValue').textContent = config.n_estimators;
    
    document.getElementById('rfMaxDepthEnabled').checked = config.max_depth_enabled;
    document.getElementById('rfMaxDepth').disabled = !config.max_depth_enabled;
    document.getElementById('rfMaxDepth').value = config.max_depth;
    
    // Manejar max_features correctamente
    if (typeof config.max_features === 'number') {
        // Si es un número, usar la opción "todas las características"
        document.getElementById('rfMaxFeatures').value = '1.0';
    } else {
        document.getElementById('rfMaxFeatures').value = config.max_features;
    }
    
    document.getElementById('rfMinSamplesSplit').value = config.min_samples_split;
    document.getElementById('rfMinSamplesLeaf').value = config.min_samples_leaf;
    
    // Primero actualizar las opciones según el tipo de problema
    updateRandomForestOptions();
    
    // Luego establecer el criterio después de que las opciones estén disponibles
    setTimeout(() => {
        document.getElementById('rfCriterion').value = config.criterion;
    }, 100);
    
    document.getElementById('rfBootstrap').checked = config.bootstrap;
    document.getElementById('rfOobScore').checked = config.oob_score;
    document.getElementById('rfValidationMethod').value = config.validation_method;
    
    // Para clasificación, configurar class weight
    if (problemType === 'classification') {
        const classWeightSelect = document.getElementById('rfClassWeight');
        if (classWeightSelect) {
            classWeightSelect.value = config.class_weight;
        }
    }
    
    // Actualizar opciones dependientes
    toggleBootstrapOptions();
    toggleCustomMaxFeatures();
    toggleRandomForestMaxDepth();
    handleValidationMethodChange();
    
    // Marcar todos los valores como NO modificados por el usuario
    // para que puedan ser actualizados por futuros presets si se desea
    RFUserState.resetModificationState();
    
    const message = problemType === 'classification' 
        ? 'Configuración optimizada para Clasificación aplicada' 
        : 'Configuración optimizada para Regresión aplicada';
    
    showNotification('Optimizado', message, 'success');
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('randomForestConfiguration')) {
        initializeRandomForestConfig();
    }
});