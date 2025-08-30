import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Slider,
  Grid,
  Alert,
  Chip
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';

const TrainTestSplit: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelType, datasetId, predictorColumns, targetColumns, normalizationMethod, hyperparameters } = location.state || {};
  
  const [splits, setSplits] = useState({
    train: 70,
    validation: 15,
    test: 15
  });

  const handleSplitChange = (type: string, value: number) => {
    const newSplits = { ...splits };
    const oldValue = newSplits[type as keyof typeof splits];
    const diff = value - oldValue;
    
    newSplits[type as keyof typeof splits] = value;
    
    // Adjust other splits proportionally
    if (type === 'train') {
      const remaining = 100 - value;
      const valTestRatio = splits.validation / (splits.validation + splits.test);
      newSplits.validation = Math.round(remaining * valTestRatio);
      newSplits.test = remaining - newSplits.validation;
    } else if (type === 'validation') {
      if (splits.train + value + splits.test > 100) {
        newSplits.test = 100 - splits.train - value;
      }
    } else if (type === 'test') {
      if (splits.train + splits.validation + value > 100) {
        newSplits.validation = 100 - splits.train - value;
      }
    }
    
    // Ensure all values are valid
    Object.keys(newSplits).forEach(key => {
      newSplits[key as keyof typeof splits] = Math.max(5, Math.min(90, newSplits[key as keyof typeof splits]));
    });
    
    // Ensure sum is 100
    const sum = newSplits.train + newSplits.validation + newSplits.test;
    if (sum !== 100) {
      newSplits.test = 100 - newSplits.train - newSplits.validation;
    }
    
    setSplits(newSplits);
  };

  const handleContinue = () => {
    navigate('/metrics-selection', {
      state: {
        modelType,
        datasetId,
        predictorColumns,
        targetColumns,
        normalizationMethod,
        hyperparameters,
        trainSplit: splits.train / 100,
        valSplit: splits.validation / 100,
        testSplit: splits.test / 100
      }
    });
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        División de Datos
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Define cómo dividir tu dataset en conjuntos de entrenamiento, validación y prueba.
      </Alert>

      <Paper sx={{ p: 4, mb: 3 }}>
        <Grid container spacing={4}>
          <Grid item xs={12}>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">
                  Entrenamiento
                </Typography>
                <Chip 
                  label={`${splits.train}%`} 
                  color="primary"
                  sx={{ minWidth: 60 }}
                />
              </Box>
              <Slider
                value={splits.train}
                onChange={(e, value) => handleSplitChange('train', value as number)}
                min={50}
                max={90}
                step={5}
                marks
                sx={{ color: '#1976d2' }}
              />
              <Typography variant="caption" color="text.secondary">
                Datos usados para entrenar el modelo
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">
                  Validación
                </Typography>
                <Chip 
                  label={`${splits.validation}%`} 
                  color="secondary"
                  sx={{ minWidth: 60 }}
                />
              </Box>
              <Slider
                value={splits.validation}
                onChange={(e, value) => handleSplitChange('validation', value as number)}
                min={5}
                max={30}
                step={5}
                marks
                sx={{ color: '#9c27b0' }}
              />
              <Typography variant="caption" color="text.secondary">
                Datos para ajustar hiperparámetros durante el entrenamiento
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle1">
                  Prueba
                </Typography>
                <Chip 
                  label={`${splits.test}%`} 
                  color="success"
                  sx={{ minWidth: 60 }}
                />
              </Box>
              <Slider
                value={splits.test}
                onChange={(e, value) => handleSplitChange('test', value as number)}
                min={5}
                max={30}
                step={5}
                marks
                sx={{ color: '#2e7d32' }}
              />
              <Typography variant="caption" color="text.secondary">
                Datos para evaluar el rendimiento final del modelo
              </Typography>
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
          <Typography variant="subtitle2" align="center">
            Total: {splits.train + splits.validation + splits.test}%
          </Typography>
        </Box>
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

export default TrainTestSplit;