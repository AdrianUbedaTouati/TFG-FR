import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';

// Import components
import ModelSelection from './components/ModelSelection';
import DatasetUpload from './components/DatasetUpload';
import VariableSelection from './components/VariableSelection';
import NormalizationConfig from './components/NormalizationConfig';
import HyperparameterConfig from './components/HyperparameterConfig';
import TrainTestSplit from './components/TrainTestSplit';
import MetricsSelection from './components/MetricsSelection';
import TrainingDashboard from './components/TrainingDashboard';
import WeatherMap from './components/WeatherMap';
import PredictionInterface from './components/PredictionInterface';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Sistema de Predicción Meteorológica con IA
              </Typography>
            </Toolbar>
          </AppBar>
          
          <Container maxWidth={false} sx={{ mt: 3 }}>
            <Routes>
              <Route path="/" element={<ModelSelection />} />
              <Route path="/dataset-upload" element={<DatasetUpload />} />
              <Route path="/variable-selection" element={<VariableSelection />} />
              <Route path="/normalization-config" element={<NormalizationConfig />} />
              <Route path="/hyperparameter-config" element={<HyperparameterConfig />} />
              <Route path="/train-test-split" element={<TrainTestSplit />} />
              <Route path="/metrics-selection" element={<MetricsSelection />} />
              <Route path="/training-dashboard" element={<TrainingDashboard />} />
              <Route path="/weather-map" element={<WeatherMap />} />
              <Route path="/prediction-interface" element={<PredictionInterface />} />
            </Routes>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
