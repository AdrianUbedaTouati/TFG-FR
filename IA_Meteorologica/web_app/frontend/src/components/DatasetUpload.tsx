import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Alert,
  CircularProgress,
  Chip,
  Grid
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

const DatasetUpload: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelType } = location.state || {};
  
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [datasetInfo, setDatasetInfo] = useState<any>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const uploadedFile = acceptedFiles[0];
      if (uploadedFile.name.endsWith('.csv')) {
        setFile(uploadedFile);
        setError('');
      } else {
        setError('Por favor, sube un archivo CSV');
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name);

    try {
      const response = await axios.post('http://localhost:8000/api/datasets/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const datasetId = response.data.id;
      
      // Get dataset columns
      const columnsResponse = await axios.get(
        `http://localhost:8000/api/datasets/${datasetId}/columns/`
      );
      
      setDatasetInfo({
        id: datasetId,
        ...columnsResponse.data
      });

      // Navigate to variable selection
      setTimeout(() => {
        navigate('/variable-selection', { 
          state: { 
            modelType, 
            datasetId,
            columns: columnsResponse.data.columns,
            dtypes: columnsResponse.data.dtypes
          } 
        });
      }, 1000);

    } catch (err) {
      setError('Error al subir el archivo. Por favor, intenta de nuevo.');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom align="center">
        Sube tu Dataset
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Modelo seleccionado: <strong>{modelType?.toUpperCase()}</strong>
      </Alert>

      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          border: '2px dashed #ccc',
          borderRadius: 2,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? '#f0f0f0' : 'transparent',
          transition: 'background-color 0.3s',
          mb: 3
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 64, color: '#666', mb: 2 }} />
        {isDragActive ? (
          <Typography>Suelta el archivo aquí...</Typography>
        ) : (
          <div>
            <Typography variant="h6" gutterBottom>
              Arrastra y suelta un archivo CSV aquí
            </Typography>
            <Typography variant="body2" color="text.secondary">
              o haz clic para seleccionar
            </Typography>
          </div>
        )}
      </Paper>

      {file && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Archivo seleccionado:
          </Typography>
          <Chip
            label={file.name}
            onDelete={() => setFile(null)}
            color="primary"
            sx={{ mb: 2 }}
          />
          <Typography variant="body2" color="text.secondary">
            Tamaño: {(file.size / 1024 / 1024).toFixed(2)} MB
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {datasetInfo && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Dataset cargado exitosamente. {datasetInfo.shape[0]} filas y {datasetInfo.shape[1]} columnas detectadas.
        </Alert>
      )}

      <Box sx={{ textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleUpload}
          disabled={!file || uploading}
          startIcon={uploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
        >
          {uploading ? 'Subiendo...' : 'Subir y Continuar'}
        </Button>
      </Box>
    </Box>
  );
};

export default DatasetUpload;