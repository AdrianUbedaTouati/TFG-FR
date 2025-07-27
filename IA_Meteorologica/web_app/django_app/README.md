# Sistema de Predicción Meteorológica con IA

## Descripción
Aplicación web Django para entrenar modelos de Machine Learning y Deep Learning para predicción meteorológica.

## Características

### 1. Selección de Modelos
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- Árbol de Decisiones
- Transformer
- Random Forest
- XGBoost

### 2. Proceso de Entrenamiento
1. **Carga de Dataset**: Sube archivos CSV con datos meteorológicos
2. **Selección de Variables**: 
   - Variables predictoras (azul oscuro)
   - Variables objetivo (azul claro)
3. **Normalización**: Métodos específicos por tipo de modelo
4. **Hiperparámetros**: Configuración personalizada para cada modelo
5. **División de Datos**: Train/Validation/Test
6. **Métricas**: MAE, MSE, RMSE, R²

### 3. Visualización
- Gráficas de entrenamiento en tiempo real
- Resultados de métricas en conjunto de prueba
- Mapa interactivo de predicciones para España y Francia

### 4. Predicción
- Interfaz para cargar nuevos datos
- Visualización de predicciones en mapa
- Exportación de resultados

## ⚠️ Important: Modal Guidelines

**This application does NOT use modal backdrops to prevent gray overlay issues.**

When creating modals, ALWAYS use:
```javascript
// Use the helper function
showModal('myModalId');

// Or create with backdrop: false
const modal = createModal(modalElement);
```

See `MODAL_GUIDELINES.md` for detailed instructions.

## Instalación

### Backend (Django)

```bash
cd web_app/django_app
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend (React)

```bash
cd web_app/frontend
npm install
npm start
```

## Uso

1. Accede a http://localhost:3000
2. Selecciona el tipo de modelo
3. Sube tu dataset CSV
4. Configura las variables y parámetros
5. Inicia el entrenamiento
6. Visualiza los resultados y realiza predicciones

## Estructura del Dataset

El dataset CSV debe contener:
- Columnas con variables meteorológicas (temperatura, humedad, presión, etc.)
- Opcionalmente: columnas 'latitude' y 'longitude' para visualización en mapa

## API Endpoints

- `/api/datasets/` - Gestión de datasets
- `/api/training-sessions/` - Sesiones de entrenamiento
- `/api/predictions/` - Predicciones
- `/api/predictions/map/` - Datos para mapa

## Tecnologías

- **Backend**: Django, Django REST Framework
- **ML/DL**: scikit-learn, XGBoost, TensorFlow, PyTorch
- **Frontend**: React, Material-UI, Leaflet
- **Visualización**: Chart.js, react-leaflet