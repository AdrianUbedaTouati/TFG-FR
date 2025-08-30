import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Chip
} from '@mui/material';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useLocation } from 'react-router-dom';
import axios from 'axios';

interface WeatherPoint {
  region: string;
  latitude: number;
  longitude: number;
  predictions: {
    temperature?: number;
    humidity?: number;
    pressure?: number;
    wind_speed?: number;
    [key: string]: number | undefined;
  };
}

const WeatherMap: React.FC = () => {
  const location = useLocation();
  const { sessionId } = location.state || {};
  
  const [weatherData, setWeatherData] = useState<WeatherPoint[]>([]);
  const [selectedVariable, setSelectedVariable] = useState('temperature');
  const [predictionDate, setPredictionDate] = useState(
    new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString().split('T')[0]
  );

  useEffect(() => {
    fetchWeatherData();
  }, [sessionId, predictionDate]);

  const fetchWeatherData = async () => {
    try {
      const response = await axios.get(
        `http://localhost:8000/api/predictions/map/`,
        {
          params: {
            session_id: sessionId,
            date: predictionDate
          }
        }
      );
      setWeatherData(response.data);
    } catch (error) {
      console.error('Error fetching weather data:', error);
      // Generate mock data for demonstration
      setWeatherData(generateMockData());
    }
  };

  const generateMockData = (): WeatherPoint[] => {
    const cities = [
      // Spain
      { region: 'Madrid', latitude: 40.4168, longitude: -3.7038 },
      { region: 'Barcelona', latitude: 41.3851, longitude: 2.1734 },
      { region: 'Valencia', latitude: 39.4699, longitude: -0.3763 },
      { region: 'Sevilla', latitude: 37.3891, longitude: -5.9845 },
      { region: 'Bilbao', latitude: 43.2630, longitude: -2.9350 },
      { region: 'Zaragoza', latitude: 41.6488, longitude: -0.8891 },
      // France
      { region: 'Paris', latitude: 48.8566, longitude: 2.3522 },
      { region: 'Lyon', latitude: 45.7640, longitude: 4.8357 },
      { region: 'Marseille', latitude: 43.2965, longitude: 5.3698 },
      { region: 'Toulouse', latitude: 43.6047, longitude: 1.4442 },
      { region: 'Bordeaux', latitude: 44.8378, longitude: -0.5792 },
      { region: 'Nice', latitude: 43.7102, longitude: 7.2620 }
    ];

    return cities.map(city => ({
      ...city,
      predictions: {
        temperature: 15 + Math.random() * 15,
        humidity: 40 + Math.random() * 40,
        pressure: 1000 + Math.random() * 30,
        wind_speed: Math.random() * 20
      }
    }));
  };

  const getColorForValue = (value: number, variable: string): string => {
    // Color scales for different variables
    const scales = {
      temperature: {
        min: 0, max: 35,
        colors: ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
      },
      humidity: {
        min: 0, max: 100,
        colors: ['#ffffe0', '#ffff99', '#99ccff', '#3366ff', '#000080']
      },
      pressure: {
        min: 980, max: 1040,
        colors: ['#ff0000', '#ff9900', '#ffff00', '#00ff00', '#0000ff']
      },
      wind_speed: {
        min: 0, max: 30,
        colors: ['#ffffff', '#ccffcc', '#99ff99', '#66ff66', '#00ff00']
      }
    };

    const scale = scales[variable as keyof typeof scales] || scales.temperature;
    const normalized = (value - scale.min) / (scale.max - scale.min);
    const index = Math.floor(normalized * (scale.colors.length - 1));
    return scale.colors[Math.max(0, Math.min(index, scale.colors.length - 1))];
  };

  const variables = [
    { id: 'temperature', name: 'Temperatura (°C)', unit: '°C' },
    { id: 'humidity', name: 'Humedad (%)', unit: '%' },
    { id: 'pressure', name: 'Presión (hPa)', unit: 'hPa' },
    { id: 'wind_speed', name: 'Velocidad del viento (km/h)', unit: 'km/h' }
  ];

  const selectedVarInfo = variables.find(v => v.id === selectedVariable);

  return (
    <Box sx={{ p: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h4" gutterBottom align="center">
        Mapa de Predicciones Meteorológicas
      </Typography>

      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Variable</InputLabel>
              <Select
                value={selectedVariable}
                label="Variable"
                onChange={(e) => setSelectedVariable(e.target.value)}
              >
                {variables.map(variable => (
                  <MenuItem key={variable.id} value={variable.id}>
                    {variable.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" align="center">
              Predicción para: <strong>{new Date(predictionDate).toLocaleDateString('es-ES')}</strong>
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
              <Chip label="España" color="primary" />
              <Chip label="Francia" color="secondary" />
            </Box>
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ flex: 1, position: 'relative' }}>
        <MapContainer
          center={[42.5, 0.5]}
          zoom={5}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {weatherData.map((point, index) => {
            const value = point.predictions[selectedVariable] || 0;
            const color = getColorForValue(value, selectedVariable);
            
            return (
              <CircleMarker
                key={index}
                center={[point.latitude, point.longitude]}
                radius={20}
                fillColor={color}
                color="#000"
                weight={1}
                opacity={1}
                fillOpacity={0.8}
              >
                <Popup>
                  <Box>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {point.region}
                    </Typography>
                    <Typography variant="body2">
                      {selectedVarInfo?.name}: {value.toFixed(1)} {selectedVarInfo?.unit}
                    </Typography>
                    <hr />
                    {Object.entries(point.predictions).map(([key, val]) => {
                      const varInfo = variables.find(v => v.id === key);
                      if (!varInfo || key === selectedVariable) return null;
                      return (
                        <Typography key={key} variant="caption" display="block">
                          {varInfo.name}: {val?.toFixed(1)} {varInfo.unit}
                        </Typography>
                      );
                    })}
                  </Box>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
        
        {/* Legend */}
        <Paper sx={{ 
          position: 'absolute', 
          bottom: 20, 
          right: 20, 
          p: 2,
          zIndex: 1000 
        }}>
          <Typography variant="subtitle2" gutterBottom>
            Escala de {selectedVarInfo?.name}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ 
              width: 150, 
              height: 20, 
              background: `linear-gradient(to right, ${
                selectedVariable === 'temperature' ? '#0000ff, #00ffff, #00ff00, #ffff00, #ff0000' :
                selectedVariable === 'humidity' ? '#ffffe0, #ffff99, #99ccff, #3366ff, #000080' :
                selectedVariable === 'pressure' ? '#ff0000, #ff9900, #ffff00, #00ff00, #0000ff' :
                '#ffffff, #ccffcc, #99ff99, #66ff66, #00ff00'
              })`,
              borderRadius: 1
            }} />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption">
              {selectedVariable === 'temperature' ? '0°C' :
               selectedVariable === 'humidity' ? '0%' :
               selectedVariable === 'pressure' ? '980' :
               '0'}
            </Typography>
            <Typography variant="caption">
              {selectedVariable === 'temperature' ? '35°C' :
               selectedVariable === 'humidity' ? '100%' :
               selectedVariable === 'pressure' ? '1040' :
               '30'}
            </Typography>
          </Box>
        </Paper>
      </Paper>
    </Box>
  );
};

export default WeatherMap;