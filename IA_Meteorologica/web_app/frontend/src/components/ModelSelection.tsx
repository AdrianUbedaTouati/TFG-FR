import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';

interface ModelType {
  id: string;
  nameKey: keyof typeof import('../contexts/LanguageContext')['translations']['fr'];
  descriptionKey: keyof typeof import('../contexts/LanguageContext')['translations']['fr'];
  icon: string;
}

const ModelSelection: React.FC = () => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const [selectedModel, setSelectedModel] = useState<string>('');

  const models: ModelType[] = [
    {
      id: 'lstm',
      nameKey: 'lstm',
      descriptionKey: 'lstmDesc',
      icon: '🧠'
    },
    {
      id: 'cnn',
      nameKey: 'cnn',
      descriptionKey: 'cnnDesc',
      icon: '🔲'
    },
    {
      id: 'decision_tree',
      nameKey: 'decisionTree',
      descriptionKey: 'decisionTreeDesc',
      icon: '🌳'
    },
    {
      id: 'transformer',
      nameKey: 'transformer',
      descriptionKey: 'transformerDesc',
      icon: '🔄'
    },
    {
      id: 'random_forest',
      nameKey: 'randomForest',
      descriptionKey: 'randomForestDesc',
      icon: '🌲'
    },
    {
      id: 'nbeats',
      nameKey: 'nbeats',
      descriptionKey: 'nbeatsDesc',
      icon: '📊'
    },
    {
      id: 'nhits',
      nameKey: 'nhits',
      descriptionKey: 'nhitsDesc',
      icon: '📈'
    }
  ];

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
        {t.selectModel}
      </Typography>
      <Typography variant="body1" gutterBottom align="center" sx={{ mb: 4 }}>
        {t.modelSelection}
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
                  {t[model.nameKey]}
                </Typography>
                <Typography variant="body2" color="text.secondary" align="center">
                  {t[model.descriptionKey]}
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
          {t.next}
        </Button>
      </Box>
    </Box>
  );
};

export default ModelSelection;