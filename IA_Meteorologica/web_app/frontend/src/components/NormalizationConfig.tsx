import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Grid,
  Chip
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

interface NormalizationMethod {
  value: string;
  label: string;
  description: string;
}

const normalizationMethods: Record<string, NormalizationMethod[]> = {
  lstm: [
    { value: 'min_max', label: 'Min-Max', description: 'Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Media 0 y desviación estándar 1' },
    { value: 'robust', label: 'Robust Scaler', description: 'Resistente a outliers' }
  ],
  cnn: [
    { value: 'min_max', label: 'Min-Max', description: 'Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Media 0 y desviación estándar 1' },
    { value: 'robust', label: 'Robust Scaler', description: 'Resistente a outliers' }
  ],
  transformer: [
    { value: 'min_max', label: 'Min-Max', description: 'Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Media 0 y desviación estándar 1' },
    { value: 'robust', label: 'Robust Scaler', description: 'Resistente a outliers' }
  ],
  decision_tree: [
    { value: 'none', label: 'Sin normalización', description: 'Los árboles no requieren normalización' },
    { value: 'min_max', label: 'Min-Max', description: 'Opcional: Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Opcional: Media 0 y desviación estándar 1' }
  ],
  random_forest: [
    { value: 'none', label: 'Sin normalización', description: 'Los árboles no requieren normalización' },
    { value: 'min_max', label: 'Min-Max', description: 'Opcional: Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Opcional: Media 0 y desviación estándar 1' }
  ],
  xgboost: [
    { value: 'none', label: 'Sin normalización', description: 'XGBoost no requiere normalización' },
    { value: 'min_max', label: 'Min-Max', description: 'Opcional: Escala valores entre 0 y 1' },
    { value: 'standard', label: 'Estandarización', description: 'Opcional: Media 0 y desviación estándar 1' }
  ]
};

const NormalizationConfig: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelType, datasetId, predictorColumns, targetColumns } = location.state || {};
  
  const [selectedMethod, setSelectedMethod] = useState('');
  const [availableMethods, setAvailableMethods] = useState<NormalizationMethod[]>([]);

  useEffect(() => {
    if (modelType) {
      const methods = normalizationMethods[modelType] || [];
      setAvailableMethods(methods);
      
      // Set default method
      if (methods.length > 0) {
        setSelectedMethod(methods[0].value);
      }
    }
  }, [modelType]);

  const handleContinue = () => {
    navigate('/hyperparameter-config', {
      state: {
        modelType,
        datasetId,
        predictorColumns,
        targetColumns,
        normalizationMethod: selectedMethod
      }
    });
  };

  const selectedMethodInfo = availableMethods.find(m => m.value === selectedMethod);

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Configuración de Normalización
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          Modelo: <strong>{modelType?.toUpperCase()}</strong>
          <br />
          Variables predictoras: {predictorColumns?.length || 0}
          <br />
          Variables objetivo: {targetColumns?.length || 0}
        </Typography>
      </Alert>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Método de Normalización
        </Typography>
        
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Selecciona un método</InputLabel>
          <Select
            value={selectedMethod}
            label="Selecciona un método"
            onChange={(e) => setSelectedMethod(e.target.value)}
          >
            {availableMethods.map((method) => (
              <MenuItem key={method.value} value={method.value}>
                {method.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {selectedMethodInfo && (
          <Alert severity="info" icon={false}>
            <Typography variant="subtitle2" fontWeight="bold">
              {selectedMethodInfo.label}
            </Typography>
            <Typography variant="body2">
              {selectedMethodInfo.description}
            </Typography>
          </Alert>
        )}
      </Paper>

      <Paper sx={{ p: 3, mb: 3, bgcolor: '#f5f5f5' }}>
        <Typography variant="subtitle1" gutterBottom>
          Variables seleccionadas:
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Predictoras:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {predictorColumns?.map((col: string) => (
              <Chip
                key={col}
                label={col}
                size="small"
                sx={{ backgroundColor: '#1565c0', color: 'white' }}
              />
            ))}
          </Box>
        </Box>
        
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Objetivo:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {targetColumns?.map((col: string) => (
              <Chip
                key={col}
                label={col}
                size="small"
                sx={{ backgroundColor: '#42a5f5', color: 'white' }}
              />
            ))}
          </Box>
        </Box>
      </Paper>

      <Box sx={{ textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleContinue}
          disabled={!selectedMethod}
        >
          Continuar
        </Button>
      </Box>
    </Box>
  );
};

export default NormalizationConfig;