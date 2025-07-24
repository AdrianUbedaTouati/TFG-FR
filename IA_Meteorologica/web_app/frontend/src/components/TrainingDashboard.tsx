import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  LinearProgress,
  Alert,
  Button,
  Chip
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TrainingHistory {
  train_loss: number[];
  val_loss: number[];
}

interface TestResults {
  [metric: string]: number;
}

const TrainingDashboard: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { sessionId } = location.state || {};
  
  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!sessionId) {
      navigate('/');
      return;
    }

    const fetchSessionStatus = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/training-sessions/${sessionId}/`
        );
        setSession(response.data);
        
        // Simulate progress
        if (response.data.status === 'training') {
          setProgress(prev => Math.min(prev + 10, 90));
        } else if (response.data.status === 'completed') {
          setProgress(100);
          setLoading(false);
        } else if (response.data.status === 'failed') {
          setLoading(false);
        }
      } catch (error) {
        console.error('Error fetching session:', error);
      }
    };

    const interval = setInterval(fetchSessionStatus, 2000);
    fetchSessionStatus();

    return () => clearInterval(interval);
  }, [sessionId, navigate]);

  const getChartData = () => {
    if (!session?.training_history) return null;

    const epochs = Array.from(
      { length: session.training_history.train_loss.length }, 
      (_, i) => i + 1
    );

    return {
      labels: epochs,
      datasets: [
        {
          label: 'Pérdida de Entrenamiento',
          data: session.training_history.train_loss,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1
        },
        {
          label: 'Pérdida de Validación',
          data: session.training_history.val_loss,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Historial de Entrenamiento'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  const handleViewMap = () => {
    navigate('/weather-map', { state: { sessionId } });
  };

  const handleNewPrediction = () => {
    navigate('/prediction-interface', { state: { sessionId } });
  };

  if (!session) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Panel de Entrenamiento
      </Typography>

      {/* Status Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Estado del Entrenamiento
            </Typography>
            <Chip
              label={session.status === 'training' ? 'Entrenando' : 
                     session.status === 'completed' ? 'Completado' : 
                     session.status === 'failed' ? 'Error' : 'Pendiente'}
              color={session.status === 'completed' ? 'success' : 
                     session.status === 'failed' ? 'error' : 'primary'}
              sx={{ mb: 2 }}
            />
            {session.status === 'training' && (
              <LinearProgress variant="determinate" value={progress} />
            )}
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2">
              <strong>Modelo:</strong> {session.model_type?.toUpperCase()}
            </Typography>
            <Typography variant="body2">
              <strong>Dataset:</strong> {session.dataset_name}
            </Typography>
            <Typography variant="body2">
              <strong>Creado:</strong> {new Date(session.created_at).toLocaleString()}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Error Message */}
      {session.status === 'failed' && session.error_message && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {session.error_message}
        </Alert>
      )}

      {/* Training Chart */}
      {session.training_history && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Gráfica de Entrenamiento
          </Typography>
          <Box sx={{ height: 400 }}>
            <Line data={getChartData()!} options={chartOptions} />
          </Box>
        </Paper>
      )}

      {/* Test Results */}
      {session.test_results && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resultados en Conjunto de Prueba
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(session.test_results).map(([metric, value]) => (
              <Grid item xs={12} sm={6} md={3} key={metric}>
                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                  <Typography variant="h4" color="primary">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </Typography>
                  <Typography variant="subtitle1">
                    {metric.toUpperCase()}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* Actions */}
      {session.status === 'completed' && (
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Button
            variant="contained"
            size="large"
            sx={{ mr: 2 }}
            onClick={handleViewMap}
          >
            Ver Mapa de Predicciones
          </Button>
          <Button
            variant="outlined"
            size="large"
            onClick={handleNewPrediction}
          >
            Realizar Nueva Predicción
          </Button>
        </Box>
      )}

      {loading && session.status === 'training' && (
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ mt: 2 }}>
            Entrenando modelo... Esto puede tomar varios minutos.
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default TrainingDashboard;