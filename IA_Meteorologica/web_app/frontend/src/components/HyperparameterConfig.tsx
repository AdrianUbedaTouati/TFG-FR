import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  TextField,
  Grid,
  Slider,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Chip,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

interface Hyperparameter {
  name: string;
  label: string;
  type: 'number' | 'select' | 'slider' | 'checkbox' | 'conditional' | 'preset';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: any; label: string }[];
  description: string;
  category?: 'basic' | 'advanced';
  tooltip?: string;
  dependsOn?: string;
  dependsOnValue?: any;
  problemType?: 'classification' | 'regression' | 'both';
}

const hyperparameterConfigs: Record<string, Hyperparameter[]> = {
  lstm: [
    { name: 'units', label: 'Unidades LSTM', type: 'number', default: 50, min: 10, max: 500, description: 'Número de unidades en cada capa LSTM' },
    { name: 'layers', label: 'Número de capas', type: 'number', default: 2, min: 1, max: 5, description: 'Número de capas LSTM apiladas' },
    { name: 'dropout', label: 'Dropout', type: 'slider', default: 0.2, min: 0, max: 0.5, step: 0.1, description: 'Tasa de dropout para regularización' },
    { name: 'epochs', label: 'Épocas', type: 'number', default: 50, min: 10, max: 500, description: 'Número de épocas de entrenamiento' },
    { name: 'batch_size', label: 'Tamaño del batch', type: 'select', default: 32, options: [
      { value: 16, label: '16' },
      { value: 32, label: '32' },
      { value: 64, label: '64' },
      { value: 128, label: '128' }
    ], description: 'Tamaño del batch para entrenamiento' },
    { name: 'learning_rate', label: 'Tasa de aprendizaje', type: 'slider', default: 0.001, min: 0.0001, max: 0.1, step: 0.0001, description: 'Tasa de aprendizaje del optimizador' }
  ],
  cnn: [
    { name: 'filters', label: 'Filtros', type: 'number', default: 64, min: 16, max: 256, description: 'Número de filtros convolucionales' },
    { name: 'kernel_size', label: 'Tamaño del kernel', type: 'select', default: 3, options: [
      { value: 3, label: '3x3' },
      { value: 5, label: '5x5' },
      { value: 7, label: '7x7' }
    ], description: 'Tamaño del kernel convolucional' },
    { name: 'pool_size', label: 'Tamaño del pool', type: 'select', default: 2, options: [
      { value: 2, label: '2x2' },
      { value: 3, label: '3x3' }
    ], description: 'Tamaño del max pooling' },
    { name: 'dense_units', label: 'Unidades densas', type: 'number', default: 100, min: 10, max: 500, description: 'Unidades en la capa densa' },
    { name: 'epochs', label: 'Épocas', type: 'number', default: 50, min: 10, max: 500, description: 'Número de épocas de entrenamiento' },
    { name: 'batch_size', label: 'Tamaño del batch', type: 'select', default: 32, options: [
      { value: 16, label: '16' },
      { value: 32, label: '32' },
      { value: 64, label: '64' },
      { value: 128, label: '128' }
    ], description: 'Tamaño del batch para entrenamiento' }
  ],
  decision_tree: [
    { name: 'max_depth', label: 'Profundidad máxima', type: 'number', default: 10, min: 1, max: 50, description: 'Profundidad máxima del árbol' },
    { name: 'min_samples_split', label: 'Min muestras para dividir', type: 'number', default: 2, min: 2, max: 20, description: 'Mínimo de muestras para dividir un nodo' },
    { name: 'min_samples_leaf', label: 'Min muestras en hoja', type: 'number', default: 1, min: 1, max: 20, description: 'Mínimo de muestras en un nodo hoja' },
    { name: 'max_features', label: 'Max características', type: 'select', default: 'auto', options: [
      { value: 'auto', label: 'Auto' },
      { value: 'sqrt', label: 'Raíz cuadrada' },
      { value: 'log2', label: 'Log2' }
    ], description: 'Número máximo de características a considerar' }
  ],
  transformer: [
    { name: 'd_model', label: 'Dimensión del modelo', type: 'number', default: 128, min: 32, max: 512, description: 'Dimensión de embeddings' },
    { name: 'nhead', label: 'Cabezas de atención', type: 'select', default: 8, options: [
      { value: 4, label: '4' },
      { value: 8, label: '8' },
      { value: 16, label: '16' }
    ], description: 'Número de cabezas de atención' },
    { name: 'num_layers', label: 'Número de capas', type: 'number', default: 3, min: 1, max: 10, description: 'Número de capas del transformer' },
    { name: 'dropout', label: 'Dropout', type: 'slider', default: 0.1, min: 0, max: 0.5, step: 0.1, description: 'Tasa de dropout' },
    { name: 'epochs', label: 'Épocas', type: 'number', default: 50, min: 10, max: 500, description: 'Número de épocas de entrenamiento' },
    { name: 'batch_size', label: 'Tamaño del batch', type: 'select', default: 32, options: [
      { value: 16, label: '16' },
      { value: 32, label: '32' },
      { value: 64, label: '64' }
    ], description: 'Tamaño del batch para entrenamiento' }
  ],
  random_forest: [
    // Preset selector
    { name: 'preset', label: 'Configuración predefinida', type: 'select', default: 'balanceado', category: 'basic',
      options: [
        { value: 'rapido', label: 'Rápido' },
        { value: 'balanceado', label: 'Balanceado' },
        { value: 'preciso', label: 'Preciso (lento)' }
      ], 
      description: 'Configuración predefinida que ajusta valores por defecto',
      tooltip: 'Rápido: menos árboles, más rápido. Balanceado: buen compromiso. Preciso: más árboles, mejor calidad.'
    },
    
    // Problem type selector
    { name: 'problem_type', label: 'Tipo de problema', type: 'select', default: 'regression', category: 'basic',
      options: [
        { value: 'classification', label: 'Clasificación' },
        { value: 'regression', label: 'Regresión' }
      ],
      description: 'Tipo de problema que vas a resolver',
      tooltip: 'Clasificación para predecir categorías, Regresión para valores numéricos'
    },

    // 🟢 Opciones sencillas
    { name: 'n_estimators', label: 'Número de árboles', type: 'slider', default: 300, min: 50, max: 1000, step: 50, category: 'basic',
      description: 'Número de árboles en el bosque',
      tooltip: 'Más árboles = mejor estabilidad, pero más tiempo de entrenamiento'
    },
    
    { name: 'max_depth_enabled', label: 'Limitar profundidad', type: 'checkbox', default: false, category: 'basic',
      description: 'Activar para limitar la profundidad máxima de los árboles',
      tooltip: 'Limitar la profundidad reduce sobreajuste y acelera el entrenamiento'
    },
    
    { name: 'max_depth', label: 'Profundidad máxima', type: 'number', default: 10, min: 2, max: 50, category: 'basic',
      dependsOn: 'max_depth_enabled', dependsOnValue: true,
      description: 'Profundidad máxima de cada árbol',
      tooltip: 'Solo aplica si "Limitar profundidad" está activado'
    },

    { name: 'max_features', label: 'Características por división', type: 'conditional', default: 'sqrt', category: 'basic',
      problemType: 'both',
      options: [
        { value: 'sqrt', label: 'sqrt (recomendado para clasificación)' },
        { value: 'log2', label: 'log2' },
        { value: '1.0', label: '1.0 (todas - recomendado para regresión)' },
        { value: 'custom', label: 'Fracción personalizada' }
      ],
      description: 'Número de características a considerar en cada división',
      tooltip: 'sqrt para clasificación, 1.0 (todas) para regresión'
    },

    { name: 'max_features_fraction', label: 'Fracción de características', type: 'slider', default: 0.5, min: 0.1, max: 1.0, step: 0.1, category: 'basic',
      dependsOn: 'max_features', dependsOnValue: 'custom',
      description: 'Fracción de características a usar (0.1 - 1.0)',
      tooltip: 'Solo aplica si seleccionaste "Fracción personalizada"'
    },

    { name: 'criterion', label: 'Criterio de división', type: 'select', default: 'auto', category: 'basic',
      options: [
        { value: 'auto', label: 'Automático (recomendado)' },
        { value: 'gini', label: 'Gini (clasificación)' },
        { value: 'entropy', label: 'Entropía (clasificación)' },
        { value: 'squared_error', label: 'Error cuadrático (regresión)' }
      ],
      description: 'Criterio para evaluar la calidad de las divisiones',
      tooltip: 'Automático selecciona el mejor según el tipo de problema'
    },

    { name: 'class_weight_balanced', label: 'Balancear clases automáticamente', type: 'checkbox', default: false, category: 'basic',
      problemType: 'classification',
      description: 'Ajustar automáticamente los pesos de las clases',
      tooltip: 'Útil cuando tienes clases desbalanceadas en clasificación'
    },

    { name: 'validation_method', label: 'Método de validación', type: 'select', default: 'holdout', category: 'basic',
      options: [
        { value: 'holdout', label: 'Hold-out 80/20' },
        { value: 'cv', label: 'Validación cruzada k-fold (k=5)' }
      ],
      description: 'Método para evaluar el modelo durante entrenamiento',
      tooltip: 'Solo para evaluación; no afecta al modelo final'
    },

    // ⚙️ Opciones avanzadas - Estructura de árbol
    { name: 'min_samples_split', label: 'Min muestras para dividir', type: 'number', default: 2, min: 2, max: 50, category: 'advanced',
      description: 'Mínimo de muestras necesarias para dividir un nodo interno',
      tooltip: 'Valores más altos previenen sobreajuste'
    },

    { name: 'min_samples_leaf', label: 'Min muestras en hoja', type: 'number', default: 1, min: 1, max: 50, category: 'advanced',
      description: 'Mínimo de muestras requeridas en un nodo hoja',
      tooltip: 'Valores más altos suavizan el modelo'
    },

    { name: 'min_weight_fraction_leaf', label: 'Fracción mín peso en hoja', type: 'slider', default: 0.0, min: 0.0, max: 0.5, step: 0.01, category: 'advanced',
      description: 'Fracción mínima del peso total en un nodo hoja',
      tooltip: 'Para datasets con pesos de muestra desiguales'
    },

    { name: 'min_impurity_decrease', label: 'Decremento mín impureza', type: 'slider', default: 0.0, min: 0.0, max: 0.1, step: 0.001, category: 'advanced',
      description: 'Umbral para la división de nodos',
      tooltip: 'Solo divide si reduce la impureza al menos en este valor'
    },

    { name: 'max_leaf_nodes', label: 'Máx nodos hoja', type: 'number', default: null, min: 2, max: 10000, category: 'advanced',
      description: 'Número máximo de nodos hoja (None = sin límite)',
      tooltip: 'Controla la complejidad del modelo'
    },

    { name: 'ccp_alpha', label: 'Alpha poda complejidad', type: 'slider', default: 0.0, min: 0.0, max: 0.1, step: 0.001, category: 'advanced',
      description: 'Parámetro de poda por coste-complejidad',
      tooltip: 'Valores > 0 activan la poda post-entrenamiento'
    },

    // Criterios completos (avanzado)
    { name: 'criterion_advanced', label: 'Criterio avanzado', type: 'select', default: 'default', category: 'advanced',
      options: [
        { value: 'default', label: 'Usar criterio básico' },
        { value: 'log_loss', label: 'Log Loss (clasificación)' },
        { value: 'absolute_error', label: 'Error absoluto (regresión)' },
        { value: 'friedman_mse', label: 'Friedman MSE (regresión)' },
        { value: 'poisson', label: 'Poisson (regresión)' }
      ],
      description: 'Criterios adicionales para usuarios avanzados',
      tooltip: 'Solo usar si entiendes las diferencias técnicas'
    },

    // Muestreo y OOB
    { name: 'bootstrap', label: 'Bootstrap', type: 'checkbox', default: true, category: 'advanced',
      description: 'Usar muestreo bootstrap para construir árboles',
      tooltip: 'Desactivar hace que cada árbol use todo el dataset'
    },

    { name: 'max_samples', label: 'Máx muestras por árbol', type: 'slider', default: 1.0, min: 0.1, max: 1.0, step: 0.1, category: 'advanced',
      dependsOn: 'bootstrap', dependsOnValue: true,
      description: 'Fracción de muestras para entrenar cada árbol',
      tooltip: 'Solo aplica si Bootstrap está activado'
    },

    { name: 'oob_score', label: 'Calcular puntuación OOB', type: 'checkbox', default: false, category: 'advanced',
      dependsOn: 'bootstrap', dependsOnValue: true,
      description: 'Evaluar con muestras out-of-bag',
      tooltip: 'Valida sin separar datos adicionales. Solo con Bootstrap.'
    },

    // Paralelismo y reproducibilidad
    { name: 'n_jobs', label: 'Trabajos paralelos', type: 'select', default: -1, category: 'advanced',
      options: [
        { value: -1, label: 'Usar todos los núcleos' },
        { value: 1, label: '1 núcleo' },
        { value: 2, label: '2 núcleos' },
        { value: 4, label: '4 núcleos' },
        { value: 8, label: '8 núcleos' }
      ],
      description: 'Número de trabajos paralelos para entrenamiento',
      tooltip: '-1 usa todos los núcleos disponibles'
    },

    { name: 'random_state', label: 'Semilla aleatoria', type: 'number', default: null, min: 0, max: 999999, category: 'advanced',
      description: 'Semilla para reproducibilidad (None = aleatorio)',
      tooltip: 'Usar la misma semilla garantiza resultados reproducibles'
    },

    { name: 'verbose', label: 'Verbosidad', type: 'select', default: 0, category: 'advanced',
      options: [
        { value: 0, label: 'Silencioso' },
        { value: 1, label: 'Progreso básico' },
        { value: 2, label: 'Información detallada' }
      ],
      description: 'Nivel de información durante entrenamiento',
      tooltip: 'Valores más altos muestran más información'
    },

    // Entrenamiento incremental
    { name: 'warm_start', label: 'Inicio cálido', type: 'checkbox', default: false, category: 'advanced',
      description: 'Permitir añadir árboles a un modelo existente',
      tooltip: 'Para entrenamiento incremental reutilizando árboles previos'
    },

    // Pesos de clase personalizados (solo clasificación)
    { name: 'class_weight_custom', label: 'Pesos de clase personalizados', type: 'checkbox', default: false, category: 'advanced',
      problemType: 'classification',
      description: 'Definir pesos personalizados para cada clase',
      tooltip: 'Se desactiva si "Balancear clases automáticamente" está activo'
    },

    // Ajustes de inferencia (solo clasificación)
    { name: 'decision_threshold', label: 'Umbral de decisión', type: 'slider', default: 0.5, min: 0.1, max: 0.9, step: 0.05, category: 'advanced',
      problemType: 'classification',
      description: 'Umbral para clasificación binaria',
      tooltip: 'Solo para clasificación binaria. Valores > 0.5 favorecen especificidad'
    },

    { name: 'output_type', label: 'Tipo de salida', type: 'select', default: 'class', category: 'advanced',
      problemType: 'classification',
      options: [
        { value: 'class', label: 'Clase predicha' },
        { value: 'proba', label: 'Probabilidades' }
      ],
      description: 'Tipo de predicción a retornar',
      tooltip: 'Clase para decisiones finales, Probabilidades para análisis'
    }
  ],
  xgboost: [
    { name: 'n_estimators', label: 'Número de árboles', type: 'number', default: 100, min: 10, max: 1000, description: 'Número de árboles' },
    { name: 'max_depth', label: 'Profundidad máxima', type: 'number', default: 6, min: 1, max: 20, description: 'Profundidad máxima de los árboles' },
    { name: 'learning_rate', label: 'Tasa de aprendizaje', type: 'slider', default: 0.3, min: 0.01, max: 1, step: 0.01, description: 'Tasa de aprendizaje' },
    { name: 'subsample', label: 'Subsample', type: 'slider', default: 0.8, min: 0.5, max: 1, step: 0.1, description: 'Fracción de muestras para entrenar cada árbol' },
    { name: 'colsample_bytree', label: 'Columnas por árbol', type: 'slider', default: 0.8, min: 0.5, max: 1, step: 0.1, description: 'Fracción de columnas por árbol' }
  ]
};

const HyperparameterConfig: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelType, datasetId, predictorColumns, targetColumns, normalizationMethod } = location.state || {};
  
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [config, setConfig] = useState<Hyperparameter[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Preset configurations for Random Forest
  const presetConfigs = {
    rapido: {
      n_estimators: 150,
      max_depth_enabled: true,
      max_depth: 10,
      max_features: 'sqrt',
      bootstrap: true,
      oob_score: false,
      n_jobs: -1
    },
    balanceado: {
      n_estimators: 300,
      max_depth_enabled: false,
      max_features: 'sqrt',
      bootstrap: true,
      oob_score: false,
      n_jobs: -1
    },
    preciso: {
      n_estimators: 600,
      max_depth_enabled: false,
      max_features: 'sqrt',
      bootstrap: true,
      oob_score: true,
      min_samples_leaf: 2,
      n_jobs: -1
    }
  };

  useEffect(() => {
    if (modelType) {
      const modelConfig = hyperparameterConfigs[modelType] || [];
      setConfig(modelConfig);
      
      // Initialize with default values
      const defaults: Record<string, any> = {};
      modelConfig.forEach(param => {
        defaults[param.name] = param.default;
      });
      setHyperparameters(defaults);
    }
  }, [modelType]);

  const handleParameterChange = (name: string, value: any) => {
    setHyperparameters(prev => {
      const newParams = { ...prev, [name]: value };
      
      // Handle preset changes for Random Forest
      if (name === 'preset' && modelType === 'random_forest') {
        const presetValues = presetConfigs[value as keyof typeof presetConfigs];
        if (presetValues) {
          Object.assign(newParams, presetValues);
        }
      }
      
      // Handle problem type changes for max_features default
      if (name === 'problem_type' && modelType === 'random_forest') {
        if (value === 'classification') {
          newParams.max_features = 'sqrt';
        } else if (value === 'regression') {
          newParams.max_features = '1.0';
        }
      }
      
      // Handle criterion auto selection
      if (name === 'criterion' && value === 'auto' && modelType === 'random_forest') {
        const problemType = newParams.problem_type || 'regression';
        if (problemType === 'classification') {
          newParams.criterion = 'gini';
        } else {
          newParams.criterion = 'squared_error';
        }
      }
      
      return newParams;
    });
  };

  const handleContinue = () => {
    navigate('/train-test-split', {
      state: {
        modelType,
        datasetId,
        predictorColumns,
        targetColumns,
        normalizationMethod,
        hyperparameters
      }
    });
  };

  const renderParameterInput = (param: Hyperparameter) => {
    // Check if parameter should be shown based on dependencies
    if (param.dependsOn) {
      const dependentValue = hyperparameters[param.dependsOn];
      if (dependentValue !== param.dependsOnValue) {
        return null;
      }
    }
    
    // Check if parameter should be shown based on problem type
    if (param.problemType && param.problemType !== 'both') {
      const currentProblemType = hyperparameters.problem_type || 'regression';
      if (param.problemType !== currentProblemType) {
        return null;
      }
    }

    switch (param.type) {
      case 'number':
        return (
          <TextField
            fullWidth
            type="number"
            value={hyperparameters[param.name] ?? param.default ?? ''}
            onChange={(e) => {
              const value = e.target.value === '' ? null : Number(e.target.value);
              handleParameterChange(param.name, value);
            }}
            InputProps={{
              inputProps: { min: param.min, max: param.max }
            }}
          />
        );
      
      case 'slider':
        return (
          <Box sx={{ px: 2 }}>
            <Slider
              value={hyperparameters[param.name] ?? param.default}
              onChange={(e, value) => handleParameterChange(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step}
              valueLabelDisplay="auto"
              marks
            />
            <Typography variant="caption" align="center" display="block">
              {hyperparameters[param.name] ?? param.default}
            </Typography>
          </Box>
        );
      
      case 'select':
      case 'conditional':
        return (
          <FormControl fullWidth>
            <Select
              value={hyperparameters[param.name] ?? param.default}
              onChange={(e) => handleParameterChange(param.name, e.target.value)}
            >
              {param.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );
        
      case 'checkbox':
        return (
          <FormControlLabel
            control={
              <Checkbox
                checked={hyperparameters[param.name] ?? param.default ?? false}
                onChange={(e) => handleParameterChange(param.name, e.target.checked)}
              />
            }
            label=""
          />
        );
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1000, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Configuración de Hiperparámetros
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Modelo: <strong>{modelType?.toUpperCase()}</strong> | 
        Normalización: <strong>{normalizationMethod}</strong>
      </Alert>

      {/* Basic Options */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ mr: 2 }}>
            🟢 Opciones sencillas
          </Typography>
          <Chip label="Recomendado para empezar" size="small" color="success" />
        </Box>
        
        <Grid container spacing={3}>
          {config
            .filter(param => param.category === 'basic' || !param.category)
            .map((param) => {
              const input = renderParameterInput(param);
              if (!input) return null;
              
              return (
                <Grid item xs={12} md={6} key={param.name}>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle1" sx={{ mr: 1 }}>
                        {param.label}
                      </Typography>
                      {param.tooltip && (
                        <Tooltip title={param.tooltip} arrow>
                          <Typography variant="caption" sx={{ cursor: 'help', color: 'primary.main' }}>
                            ℹ️
                          </Typography>
                        </Tooltip>
                      )}
                    </Box>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                      {param.description}
                    </Typography>
                    {input}
                  </Box>
                </Grid>
              );
            })}
        </Grid>
      </Paper>

      {/* Advanced Options */}
      {modelType === 'random_forest' && (
        <Accordion expanded={showAdvanced} onChange={() => setShowAdvanced(!showAdvanced)}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="h6" sx={{ mr: 2 }}>
                ⚙️ Opciones avanzadas
              </Typography>
              <Chip label="Para usuarios expertos" size="small" color="warning" />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Configuración detallada para afinar el rendimiento del modelo Random Forest.
            </Typography>
            
            <Grid container spacing={3}>
              {config
                .filter(param => param.category === 'advanced')
                .map((param) => {
                  const input = renderParameterInput(param);
                  if (!input) return null;
                  
                  return (
                    <Grid item xs={12} md={6} key={param.name}>
                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Typography variant="subtitle1" sx={{ mr: 1 }}>
                            {param.label}
                          </Typography>
                          {param.tooltip && (
                            <Tooltip title={param.tooltip} arrow>
                              <Typography variant="caption" sx={{ cursor: 'help', color: 'primary.main' }}>
                                ℹ️
                              </Typography>
                            </Tooltip>
                          )}
                        </Box>
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                          {param.description}
                        </Typography>
                        {input}
                      </Box>
                    </Grid>
                  );
                })}
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Non-Random Forest models - show all parameters in simple grid */}
      {modelType !== 'random_forest' && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Ajusta los hiperparámetros del modelo
          </Typography>
          
          <Grid container spacing={3}>
            {config.map((param) => (
              <Grid item xs={12} md={6} key={param.name}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    {param.label}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                    {param.description}
                  </Typography>
                  {renderParameterInput(param)}
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      <Box sx={{ textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleContinue}
        >
          Continuar
        </Button>
      </Box>
    </Box>
  );
};

export default HyperparameterConfig;