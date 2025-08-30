# Arquitectura del Proyecto Web - IA Meteorológica

## Visión General

Este proyecto es una plataforma web para entrenamiento y predicción meteorológica usando modelos de Machine Learning. Combina una arquitectura híbrida con:

- **Backend**: Django (Python) con API RESTful
- **Frontend**: Aplicación React (TypeScript) + Templates Django con JavaScript vanilla
- **Base de Datos**: SQLite (desarrollo)
- **Almacenamiento**: Sistema de archivos local para datasets y modelos

## Estructura del Proyecto

```
web_app/
├── backend/                 # Backend Flask simple (legacy)
│   ├── app.py              
│   └── requirements.txt
│
├── django_app/             # Aplicación principal Django
│   ├── manage.py           # Script de gestión Django
│   ├── db.sqlite3          # Base de datos SQLite
│   ├── requirements.txt    # Dependencias Python
│   │
│   ├── weather_prediction/ # Configuración principal Django
│   │   ├── settings.py     # Configuración del proyecto
│   │   ├── urls.py         # URLs principales
│   │   ├── views.py        # Vistas principales
│   │   ├── auth_views.py   # Autenticación
│   │   └── translations.py # Sistema de traducciones
│   │
│   ├── ml_trainer/         # App principal de ML
│   │   ├── models.py       # Modelos de datos
│   │   ├── views/          # Vistas organizadas por función
│   │   ├── urls.py         # URLs de la API
│   │   ├── serializers.py  # Serialización DRF
│   │   └── ml_utils.py     # Utilidades de ML
│   │
│   ├── templates/          # Templates Django
│   │   ├── base.html       # Template base
│   │   └── *.html          # Templates específicos
│   │
│   ├── static/             # Archivos estáticos
│   │   ├── css/            # Estilos
│   │   └── js/             # JavaScript
│   │
│   └── media/              # Archivos subidos
│       ├── datasets/       # Datasets CSV
│       └── models/         # Modelos entrenados
│
└── frontend/               # Aplicación React
    ├── src/
    │   ├── App.tsx         # Componente principal
    │   ├── components/     # Componentes React
    │   └── contexts/       # Context API
    └── package.json        # Dependencias Node.js
```

## Arquitectura del Backend (Django)

### 1. Capa de Modelos (models.py)

Los modelos principales siguen una arquitectura jerárquica:

```
CustomNormalizationFunction
    └── Funciones de normalización personalizadas

Dataset
    └── Parent Dataset (auto-referencial para normalización)
        └── Historial de transformaciones

ModelDefinition
    └── Plantilla reutilizable de modelo
        └── TrainingSession (múltiples entrenamientos)
            └── WeatherPrediction (predicciones)
```

**Características clave:**
- Uso extensivo de JSONField para configuraciones flexibles
- Relaciones auto-referenciales para tracking de normalizaciones
- Soft references para mantener integridad tras eliminaciones
- Campos de progreso para monitoreo en tiempo real

### 2. Capa de Vistas

Organización modular en `ml_trainer/views/`:

- **dataset_views.py**: CRUD de datasets, análisis, transformaciones
- **normalization_views.py**: Aplicación de normalizaciones
- **model_views.py**: Gestión de definiciones de modelos
- **training_views.py**: Control de sesiones de entrenamiento
- **prediction_views.py**: Generación de predicciones
- **export_views.py**: Exportación de código y modelos

**Patrones utilizados:**
- Class-Based Views (CBV) para operaciones CRUD
- ViewSets de Django REST Framework
- Vistas asíncronas para operaciones largas
- Decoradores para autenticación y permisos

### 3. API RESTful

Estructura de endpoints siguiendo principios REST:

```
/api/
├── datasets/                 # Gestión de datasets
│   ├── {id}/columns/        # Operaciones de columnas
│   ├── {id}/normalization/  # Normalización
│   └── {id}/analysis/       # Análisis
├── models/                   # Definiciones de modelos
│   └── {id}/trainings/      # Historial de entrenamientos
├── training-sessions/        # Sesiones de entrenamiento
│   ├── {id}/train/          # Iniciar entrenamiento
│   └── {id}/results/        # Resultados
└── predictions/              # Predicciones
    └── map/                  # Visualización geográfica
```

### 4. Sistema de Procesamiento ML

**ml_utils.py & ml_utils_pytorch.py:**
- Abstracción de frameworks (Keras, PyTorch, Scikit-learn)
- Pipeline unificado de entrenamiento
- Gestión de callbacks para progreso
- Serialización de modelos

**Flujo de entrenamiento:**
1. Carga y preparación de datos
2. Aplicación de normalizaciones
3. División train/val/test
4. Configuración del modelo según framework
5. Entrenamiento con callbacks de progreso
6. Evaluación y guardado de resultados

## Arquitectura del Frontend

### 1. Aplicación React (SPA)

**Componentes principales:**
- **ModelSelection**: Selección inicial del tipo de modelo
- **DatasetUpload**: Carga de archivos CSV con validación
- **VariableSelection**: Selección de predictores/targets
- **HyperparameterConfig**: Configuración específica por modelo
- **TrainingDashboard**: Monitoreo en tiempo real
- **WeatherMap**: Visualización geográfica con Leaflet

**Gestión de estado:**
- React Context para idioma (ES/FR/EN)
- Estado local en componentes
- Paso de estado entre rutas via location.state

**Integración con backend:**
- Axios para peticiones HTTP
- Polling para actualizaciones de progreso
- Manejo de CSRF tokens Django
- Base URL configurable

### 2. Templates Django + JavaScript

Para funcionalidades no cubiertas por React:
- Gestión de datasets (datasets.html)
- Normalización avanzada (normalize.html)
- Listado de modelos (models.html)
- Dashboard principal (dashboard.html)

**JavaScript integrado:**
- dataset-analysis.js: Análisis avanzado de variables
- training-progress-enhanced.js: Progreso específico por modelo
- random-forest-config.js: Configuración especializada RF

## Flujo de Datos

### 1. Pipeline de ML

```
1. Usuario sube CSV → Django FileField → media/datasets/
2. Análisis de columnas → Detección de tipos → Metadata en BD
3. Normalización (opcional) → Nuevo dataset derivado
4. Configuración de modelo → ModelDefinition en BD
5. Entrenamiento → TrainingSession → Callbacks de progreso
6. Modelo entrenado → media/models/ → Resultados en BD
7. Predicciones → API → Visualización en mapa
```

### 2. Autenticación y Autorización

- Django Authentication System
- Sesiones basadas en cookies
- CSRF Protection
- Middleware personalizado para idioma
- Permisos a nivel de vista

### 3. Internacionalización

Sistema trilingüe (FR/ES/EN):
- Frontend: React Context + traducciones en componentes
- Backend: Sistema de traducciones Django
- Persistencia en localStorage
- Francés como idioma por defecto

## Características Técnicas Destacadas

### 1. Gestión de Archivos
- Validación de CSV en upload
- Generación de nombres únicos con hash
- Limpieza automática de archivos huérfanos
- Soporte para datasets grandes

### 2. Procesamiento Asíncrono
- Callbacks para progreso de entrenamiento
- Polling desde frontend (2 segundos)
- Estado persistente en BD
- Capacidad de detener entrenamientos

### 3. Visualización de Datos
- Chart.js para gráficos de entrenamiento
- Leaflet para mapas interactivos
- Escalas de color dinámicas
- Responsive design

### 4. Optimizaciones
- Caché de mejores scores en ModelDefinition
- Lazy loading de datasets grandes
- Paginación en listados
- Índices en campos frecuentemente consultados

## Consideraciones de Despliegue

### Desarrollo
- Django Development Server (puerto 8000)
- React Development Server (puerto 3000)
- SQLite como BD
- Debug=True para desarrollo

### Producción (Recomendaciones)
- Gunicorn/uWSGI como servidor WSGI
- Nginx como proxy reverso
- PostgreSQL como BD
- Separación de static/media files
- HTTPS obligatorio
- Variables de entorno para secretos

## Puntos de Extensión

1. **Nuevos modelos ML**: Agregar en ModelType enum y ml_utils
2. **Nuevas normalizaciones**: Extender normalization_methods.py
3. **Nuevos tipos de split**: Modificar data_splitter.py
4. **Nuevas visualizaciones**: Agregar componentes React
5. **Nuevos idiomas**: Extender translations.py y LanguageContext

## Seguridad

- CSRF tokens en todas las peticiones POST
- Validación de archivos subidos
- Sanitización de entrada de usuario
- Permisos basados en usuario
- No exposición de rutas de archivos reales
- Secretos en variables de entorno (producción)