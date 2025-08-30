import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useDropzone } from 'react-dropzone';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useLocation } from 'react-router-dom';
import axios from 'axios';

// Fix for default markers in react-leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface PredictionResult {
  input: Record<string, any>;
  predictions: Record<string, number>;
  latitude?: number;
  longitude?: number;
}

const PredictionInterface: React.FC = () => {
  const location = useLocation();
  const { sessionId } = location.state || {};
  
  const [file, setFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [error, setError] = useState('');

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const uploadedFile = acceptedFiles[0];
      if (uploadedFile.name.endsWith('.csv')) {
        setFile(uploadedFile);
        setError('');
        setPredictions([]);
      } else {
        setError('Por favor, sube un archivo CSV');
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false
  });

  const handlePredict = async () => {
    if (!file || !sessionId) return;

    setPredicting(true);
    const formData = new FormData();
    formData.append('input_data', file);
    formData.append('session_id', sessionId.toString());

    try {
      const response = await axios.post(
        'http://localhost:8000/api/predict/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      // Process predictions and add coordinates if available
      const processedPredictions = response.data.map((pred: PredictionResult) => {
        // Try to extract coordinates from input data
        const lat = pred.input.latitude || pred.input.lat;
        const lon = pred.input.longitude || pred.input.lon || pred.input.lng;
        
        return {
          ...pred,
          latitude: lat,
          longitude: lon
        };
      });

      setPredictions(processedPredictions);
    } catch (err) {
      setError('Error al realizar las predicciones. Verifica que el archivo tenga el formato correcto.');
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  const hasCoordinates = predictions.some(p => p.latitude && p.longitude);

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Interfaz de Predicción
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Cargar Dataset para Predicción
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                p: 3,
                border: '2px dashed #ccc',
                borderRadius: 2,
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: isDragActive ? '#f0f0f0' : 'transparent',
                mb: 2
              }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: '#666', mb: 1 }} />
              {isDragActive ? (
                <Typography>Suelta el archivo aquí...</Typography>
              ) : (
                <Typography>
                  Arrastra un archivo CSV o haz clic para seleccionar
                </Typography>
              )}
            </Box>

            {file && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Archivo cargado: {file.name}
              </Alert>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={handlePredict}
              disabled={!file || predicting}
              startIcon={predicting ? <CircularProgress size={20} /> : null}
            >
              {predicting ? 'Prediciendo...' : 'Realizar Predicciones'}
            </Button>
            
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="caption">
                El archivo debe contener las mismas columnas predictoras que se usaron en el entrenamiento.
                Si incluye columnas 'latitude' y 'longitude', las predicciones se mostrarán en el mapa.
              </Typography>
            </Alert>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Información del Modelo
            </Typography>
            <Typography variant="body2" paragraph>
              Sesión ID: <strong>{sessionId}</strong>
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Este modelo fue entrenado para predecir variables meteorológicas.
              Asegúrate de que tu dataset de entrada tenga el mismo formato
              que el usado durante el entrenamiento.
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {predictions.length > 0 && (
        <>
          <Divider sx={{ my: 4 }} />
          
          <Typography variant="h5" gutterBottom>
            Resultados de Predicción
          </Typography>

          <TableContainer component={Paper} sx={{ mb: 4 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Entrada</strong></TableCell>
                  <TableCell><strong>Predicciones</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {predictions.slice(0, 10).map((pred, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {Object.entries(pred.input).slice(0, 3).map(([key, value]) => (
                        <Typography key={key} variant="caption" display="block">
                          {key}: {value}
                        </Typography>
                      ))}
                      {Object.keys(pred.input).length > 3 && (
                        <Typography variant="caption" color="text.secondary">
                          ...y {Object.keys(pred.input).length - 3} más
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      {Object.entries(pred.predictions).map(([key, value]) => (
                        <Typography key={key} variant="body2">
                          <strong>{key}:</strong> {value.toFixed(2)}
                        </Typography>
                      ))}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {predictions.length > 10 && (
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Mostrando 10 de {predictions.length} predicciones
                </Typography>
              </Box>
            )}
          </TableContainer>

          {hasCoordinates && (
            <>
              <Typography variant="h5" gutterBottom>
                Mapa de Predicciones
              </Typography>
              
              <Paper sx={{ height: 500, position: 'relative' }}>
                <MapContainer
                  center={[42.5, 0.5]}
                  zoom={5}
                  style={{ height: '100%', width: '100%' }}
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  />
                  
                  {predictions
                    .filter(p => p.latitude && p.longitude)
                    .map((pred, index) => (
                      <Marker
                        key={index}
                        position={[pred.latitude!, pred.longitude!]}
                      >
                        <Popup>
                          <Box>
                            <Typography variant="subtitle2" fontWeight="bold">
                              Predicción #{index + 1}
                            </Typography>
                            <Divider sx={{ my: 1 }} />
                            {Object.entries(pred.predictions).map(([key, value]) => (
                              <Typography key={key} variant="body2">
                                {key}: {value.toFixed(2)}
                              </Typography>
                            ))}
                          </Box>
                        </Popup>
                      </Marker>
                    ))}
                </MapContainer>
              </Paper>
            </>
          )}
        </>
      )}
    </Box>
  );
};

export default PredictionInterface;