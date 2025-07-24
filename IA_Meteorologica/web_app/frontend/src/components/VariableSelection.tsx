import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Chip,
  Grid,
  Alert,
  Divider
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';

interface VariableType {
  name: string;
  dtype: string;
  role: 'predictor' | 'target' | 'none';
}

const VariableSelection: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelType, datasetId, columns, dtypes } = location.state || {};

  const [variables, setVariables] = useState<VariableType[]>(
    columns?.map((col: string) => ({
      name: col,
      dtype: dtypes[col],
      role: 'none'
    })) || []
  );

  const handleVariableClick = (index: number) => {
    const newVariables = [...variables];
    const currentRole = newVariables[index].role;
    
    // Cycle through roles: none -> predictor -> target -> none
    if (currentRole === 'none') {
      newVariables[index].role = 'predictor';
    } else if (currentRole === 'predictor') {
      newVariables[index].role = 'target';
    } else {
      newVariables[index].role = 'none';
    }
    
    setVariables(newVariables);
  };

  const getChipColor = (role: string) => {
    switch (role) {
      case 'predictor':
        return 'primary';
      case 'target':
        return 'info';
      default:
        return 'default';
    }
  };

  const getChipStyle = (role: string) => {
    switch (role) {
      case 'predictor':
        return { backgroundColor: '#1565c0', color: 'white' }; // Azul oscuro
      case 'target':
        return { backgroundColor: '#42a5f5', color: 'white' }; // Azul claro
      default:
        return {};
    }
  };

  const handleContinue = () => {
    const predictors = variables.filter(v => v.role === 'predictor').map(v => v.name);
    const targets = variables.filter(v => v.role === 'target').map(v => v.name);

    if (predictors.length === 0 || targets.length === 0) {
      alert('Por favor, selecciona al menos una variable predictora y una variable objetivo');
      return;
    }

    navigate('/normalization-config', {
      state: {
        modelType,
        datasetId,
        predictorColumns: predictors,
        targetColumns: targets
      }
    });
  };

  const predictorCount = variables.filter(v => v.role === 'predictor').length;
  const targetCount = variables.filter(v => v.role === 'target').length;

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Selección de Variables
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Haz clic en las variables para cambiar su rol:</strong>
          <br />
          • <span style={{ color: '#1565c0' }}>Azul oscuro</span>: Variables predictoras (ayudan a predecir)
          <br />
          • <span style={{ color: '#42a5f5' }}>Azul claro</span>: Variables objetivo (a predecir)
          <br />
          • Gris: No utilizada
        </Typography>
      </Alert>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Variables del Dataset
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <Grid container spacing={2}>
          {variables.map((variable, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={variable.name}>
              <Chip
                label={
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {variable.name}
                    </Typography>
                    <Typography variant="caption" display="block">
                      {variable.dtype}
                    </Typography>
                  </Box>
                }
                onClick={() => handleVariableClick(index)}
                sx={{
                  width: '100%',
                  height: 'auto',
                  py: 1,
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  ...getChipStyle(variable.role),
                  '&:hover': {
                    transform: 'scale(1.05)',
                    boxShadow: 2
                  }
                }}
              />
            </Grid>
          ))}
        </Grid>
      </Paper>

      <Paper sx={{ p: 2, mb: 3, bgcolor: '#f5f5f5' }}>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="subtitle1">
              Variables Predictoras: <strong>{predictorCount}</strong>
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="subtitle1">
              Variables Objetivo: <strong>{targetCount}</strong>
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      <Box sx={{ textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleContinue}
          disabled={predictorCount === 0 || targetCount === 0}
        >
          Continuar
        </Button>
      </Box>
    </Box>
  );
};

export default VariableSelection;