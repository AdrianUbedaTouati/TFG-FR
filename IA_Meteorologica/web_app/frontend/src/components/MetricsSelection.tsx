import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Alert,
  Grid,
  Chip
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

interface Metric {
  id: string;
  name: string;
  description: string;
  formula?: string;
}

const metricsConfig: Record<string, Metric[]> = {
  // Regression metrics for all models
  lstm: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ],
  cnn: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ],
  decision_tree: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ],
  transformer: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ],
  random_forest: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ],
  xgboost: [
    { id: 'mae', name: 'MAE', description: 'Mean Absolute Error - Error absoluto promedio', formula: '∑|y - ŷ|/n' },
    { id: 'mse', name: 'MSE', description: 'Mean Squared Error - Error cuadrático medio', formula: '∑(y - ŷ)²/n' },
    { id: 'rmse', name: 'RMSE', description: 'Root Mean Squared Error - Raíz del error cuadrático medio', formula: '√(∑(y - ŷ)²/n)' },
    { id: 'r2', name: 'R² Score', description: 'Coeficiente de determinación', formula: '1 - SS_res/SS_tot' }
  ]
};

const MetricsSelection: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { 
    modelType, 
    datasetId, 
    predictorColumns, 
    targetColumns, 
    normalizationMethod, 
    hyperparameters,
    trainSplit,
    valSplit,
    testSplit
  } = location.state || {};
  
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['mae', 'rmse']);
  const [availableMetrics, setAvailableMetrics] = useState<Metric[]>([]);

  useEffect(() => {
    if (modelType) {
      const metrics = metricsConfig[modelType] || [];
      setAvailableMetrics(metrics);
    }
  }, [modelType]);

  const handleMetricToggle = (metricId: string) => {
    setSelectedMetrics(prev => {
      if (prev.includes(metricId)) {
        return prev.filter(id => id !== metricId);
      } else {
        return [...prev, metricId];
      }
    });
  };

  const handleStartTraining = async () => {
    try {
      // Create training session
      const sessionData = {
        dataset: datasetId,
        model_type: modelType,
        predictor_columns: predictorColumns,
        target_columns: targetColumns,
        normalization_method: normalizationMethod,
        hyperparameters: hyperparameters,
        train_split: trainSplit,
        val_split: valSplit,
        test_split: testSplit,
        selected_metrics: selectedMetrics
      };

      const response = await axios.post(
        'http://localhost:8000/api/training-sessions/',
        sessionData
      );

      const sessionId = response.data.id;

      // Start training
      await axios.post(
        `http://localhost:8000/api/training-sessions/${sessionId}/train/`
      );

      // Navigate to training dashboard
      navigate('/training-dashboard', { 
        state: { sessionId } 
      });

    } catch (error) {
      console.error('Error starting training:', error);
      alert('Error al iniciar el entrenamiento');
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Selección de Métricas
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Selecciona las métricas para evaluar el rendimiento de tu modelo de predicción meteorológica.
      </Alert>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Métricas disponibles para {modelType?.toUpperCase()}
        </Typography>
        
        <FormGroup>
          {availableMetrics.map((metric) => (
            <Paper key={metric.id} sx={{ p: 2, mb: 2, bgcolor: '#f9f9f9' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={selectedMetrics.includes(metric.id)}
                    onChange={() => handleMetricToggle(metric.id)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {metric.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {metric.description}
                    </Typography>
                    {metric.formula && (
                      <Typography variant="caption" sx={{ fontFamily: 'monospace', display: 'block', mt: 0.5 }}>
                        Fórmula: {metric.formula}
                      </Typography>
                    )}
                  </Box>
                }
              />
            </Paper>
          ))}
        </FormGroup>
      </Paper>

      <Paper sx={{ p: 3, mb: 3, bgcolor: '#f5f5f5' }}>
        <Typography variant="h6" gutterBottom>
          Resumen de configuración
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2">
              <strong>Modelo:</strong> {modelType?.toUpperCase()}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              <strong>Normalización:</strong> {normalizationMethod}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              <strong>Variables predictoras:</strong> {predictorColumns?.length}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              <strong>Variables objetivo:</strong> {targetColumns?.length}
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Typography variant="body2">
              <strong>División de datos:</strong> {Math.round(trainSplit * 100)}% train, {Math.round(valSplit * 100)}% val, {Math.round(testSplit * 100)}% test
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      <Box sx={{ textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          color="primary"
          onClick={handleStartTraining}
          disabled={selectedMetrics.length === 0}
          sx={{ 
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
          }}
        >
          Iniciar Entrenamiento
        </Button>
      </Box>
    </Box>
  );
};

export default MetricsSelection;