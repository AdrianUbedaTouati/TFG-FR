import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';

interface ModelType {
  id: string;
  name: string;
  description: string;
  icon: string;
}

const models: ModelType[] = [
  {
    id: 'lstm',
    name: 'LSTM',
    description: 'Long Short-Term Memory - Ideal para series temporales y predicciones meteorológicas',
    icon: '🧠'
  },
  {
    id: 'cnn',
    name: 'CNN',
    description: 'Convolutional Neural Network - Para patrones espaciales en datos meteorológicos',
    icon: '🔲'
  },
  {
    id: 'decision_tree',
    name: 'Árbol de Decisiones',
    description: 'Modelo interpretable para relaciones no lineales simples',
    icon: '🌳'
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: 'Arquitectura de atención para capturar dependencias complejas',
    icon: '🔄'
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    description: 'Conjunto de árboles para predicciones robustas',
    icon: '🌲'
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    description: 'Gradient Boosting optimizado para alto rendimiento',
    icon: '⚡'
  }
];

const ModelSelection: React.FC = () => {
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<string>('');

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const handleContinue = () => {
    if (selectedModel) {
      navigate('/dataset-upload', { state: { modelType: selectedModel } });
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom align="center">
        Selecciona el Tipo de Modelo
      </Typography>
      <Typography variant="body1" gutterBottom align="center" sx={{ mb: 4 }}>
        Elige el algoritmo de aprendizaje automático que mejor se adapte a tu problema
      </Typography>

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'all 0.3s',
                border: selectedModel === model.id ? '2px solid #1976d2' : '1px solid #e0e0e0',
                transform: selectedModel === model.id ? 'scale(1.02)' : 'scale(1)',
                '&:hover': {
                  transform: 'scale(1.02)',
                  boxShadow: 3,
                },
              }}
              onClick={() => handleModelSelect(model.id)}
            >
              <CardContent>
                <Box sx={{ textAlign: 'center', mb: 2 }}>
                  <Typography variant="h2">{model.icon}</Typography>
                </Box>
                <Typography variant="h6" gutterBottom align="center">
                  {model.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" align="center">
                  {model.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleContinue}
          disabled={!selectedModel}
        >
          Continuar
        </Button>
      </Box>
    </Box>
  );
};

export default ModelSelection;