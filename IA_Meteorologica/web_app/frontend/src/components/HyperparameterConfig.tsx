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
  MenuItem
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

interface Hyperparameter {
  name: string;
  label: string;
  type: 'number' | 'select' | 'slider';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: any; label: string }[];
  description: string;
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
    { name: 'n_estimators', label: 'Número de árboles', type: 'number', default: 100, min: 10, max: 1000, description: 'Número de árboles en el bosque' },
    { name: 'max_depth', label: 'Profundidad máxima', type: 'number', default: 10, min: 1, max: 50, description: 'Profundidad máxima de los árboles' },
    { name: 'min_samples_split', label: 'Min muestras para dividir', type: 'number', default: 2, min: 2, max: 20, description: 'Mínimo de muestras para dividir' },
    { name: 'min_samples_leaf', label: 'Min muestras en hoja', type: 'number', default: 1, min: 1, max: 20, description: 'Mínimo de muestras en nodo hoja' },
    { name: 'max_features', label: 'Max características', type: 'select', default: 'auto', options: [
      { value: 'auto', label: 'Auto' },
      { value: 'sqrt', label: 'Raíz cuadrada' },
      { value: 'log2', label: 'Log2' }
    ], description: 'Número máximo de características' }
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
    setHyperparameters(prev => ({
      ...prev,
      [name]: value
    }));
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
    switch (param.type) {
      case 'number':
        return (
          <TextField
            fullWidth
            type="number"
            value={hyperparameters[param.name] || ''}
            onChange={(e) => handleParameterChange(param.name, Number(e.target.value))}
            InputProps={{
              inputProps: { min: param.min, max: param.max }
            }}
          />
        );
      
      case 'slider':
        return (
          <Box sx={{ px: 2 }}>
            <Slider
              value={hyperparameters[param.name] || param.default}
              onChange={(e, value) => handleParameterChange(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step}
              valueLabelDisplay="auto"
              marks
            />
            <Typography variant="caption" align="center" display="block">
              {hyperparameters[param.name]}
            </Typography>
          </Box>
        );
      
      case 'select':
        return (
          <FormControl fullWidth>
            <Select
              value={hyperparameters[param.name] || param.default}
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