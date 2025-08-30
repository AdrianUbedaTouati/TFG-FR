# Funcionalidades del Sistema Web - IA Meteorológica

## 1. Gestión de Usuarios y Autenticación

### 1.1 Registro y Login
- **Registro de nuevos usuarios** con validación de email
- **Login seguro** con autenticación Django
- **Gestión de sesiones** con cookies seguras
- **Recuperación de contraseña** (si está configurado el email)
- **Perfil de usuario** con estadísticas de uso

### 1.2 Roles y Permisos
- **Usuario estándar**: Acceso completo a funcionalidades ML
- **Administrador**: Acceso adicional al panel de Django Admin
- **Gestión de permisos** por usuario para datasets y modelos

## 2. Gestión de Datasets

### 2.1 Carga de Datasets
- **Carga de archivos CSV** con drag & drop
- **Validación automática** del formato y estructura
- **Detección de tipos de columnas** (numérico, texto, fecha)
- **Vista previa** de las primeras filas
- **Metadatos automáticos**: tamaño, columnas, tipos

### 2.2 Análisis de Datasets
- **Estadísticas descriptivas** por columna:
  - Media, mediana, desviación estándar
  - Valores únicos, mínimo, máximo
  - Detección de valores nulos
- **Visualizaciones**:
  - Histogramas para variables numéricas
  - Gráficos de barras para categóricas
  - Matrices de correlación
  - Box plots para detectar outliers
- **Análisis de calidad**:
  - Porcentaje de valores faltantes
  - Detección de duplicados
  - Identificación de anomalías

### 2.3 Transformación de Datos
- **Operaciones por columna**:
  - Renombrar columnas
  - Eliminar columnas
  - Cambiar tipos de datos
- **Limpieza de datos**:
  - Eliminar filas con valores nulos
  - Rellenar valores faltantes (media, mediana, modo, valor personalizado)
  - Eliminar duplicados
- **Transformaciones de texto**:
  - Convertir a mayúsculas/minúsculas
  - Eliminar espacios en blanco
  - Reemplazar valores
- **Transformaciones numéricas**:
  - Operaciones matemáticas (+, -, *, /, ^)
  - Redondeo y truncamiento
  - Conversión de unidades

### 2.4 Normalización de Datos
- **Métodos estándar**:
  - Min-Max Scaling (0-1)
  - Standard Scaling (Z-score)
  - Robust Scaling (resistente a outliers)
- **Normalización personalizada**:
  - Editor de código para funciones custom
  - Funciones para transformaciones numéricas
  - Funciones para procesamiento de texto
  - Código de inicialización para imports
- **Características avanzadas**:
  - Vista previa antes de aplicar
  - Historial de normalizaciones (genealogía)
  - Reversibilidad (mantiene dataset original)

## 3. Configuración y Entrenamiento de Modelos

### 3.1 Tipos de Modelos Soportados

#### Modelos de Deep Learning:
- **LSTM (Long Short-Term Memory)**:
  - Predicción de series temporales
  - Configuración de capas y neuronas
  - Dropout y regularización
  
- **GRU (Gated Recurrent Unit)**:
  - Alternativa más eficiente a LSTM
  - Menos parámetros, entrenamiento más rápido
  
- **CNN (Convolutional Neural Networks)**:
  - Extracción de patrones espaciales
  - Configuración de filtros y kernels
  
- **Transformer**:
  - Arquitectura de atención
  - Para secuencias largas

#### Modelos Clásicos de ML:
- **Random Forest**:
  - Configuración avanzada (30+ parámetros)
  - Presets: Rápido, Balanceado, Preciso
  - Importancia de variables
  
- **XGBoost**:
  - Gradient boosting optimizado
  - Regularización L1/L2
  - Early stopping
  
- **Decision Tree**:
  - Árbol simple interpretable
  - Visualización del árbol
  - Reglas de decisión

#### Modelos Especializados:
- **N-BEATS**:
  - Arquitectura especializada para series temporales
  - Descomposición en tendencia y estacionalidad
  
- **N-HiTS**:
  - Versión mejorada de N-BEATS
  - Mejor eficiencia computacional

### 3.2 Configuración de Modelos

#### Selección de Variables:
- **Variables predictoras** (features)
- **Variables objetivo** (targets)
- **Validación automática** según tipo de modelo
- **Sugerencias** basadas en correlaciones

#### Hiperparámetros:
- **Configuración básica** para principiantes
- **Configuración avanzada** para expertos
- **Presets optimizados** por tipo de problema
- **Tooltips explicativos** para cada parámetro

#### División de Datos:
- **División aleatoria**: Train/Val/Test personalizable
- **División estratificada**: Mantiene proporciones de clases
- **División temporal**: Para series temporales
- **División por grupos**: Para datos agrupados
- **División secuencial**: Mantiene orden de datos

#### Métodos de Ejecución:
- **Entrenamiento estándar**: Una sola ejecución
- **K-Fold Cross Validation**: Validación robusta
- **Stratified K-Fold**: Para datos desbalanceados
- **Time Series Split**: Para datos temporales
- **Leave One Group Out**: Validación por grupos

### 3.3 Proceso de Entrenamiento

- **Inicio de entrenamiento** con un clic
- **Monitoreo en tiempo real**:
  - Progreso por épocas/iteraciones
  - Métricas de entrenamiento y validación
  - Gráficos de pérdida en vivo
  - Logs detallados con timestamps
- **Control del entrenamiento**:
  - Pausar/reanudar (según modelo)
  - Detener anticipadamente
  - Guardar checkpoints
- **Notificaciones**:
  - Finalización exitosa
  - Errores o advertencias
  - Mejoras en métricas

## 4. Evaluación y Análisis de Resultados

### 4.1 Métricas de Evaluación

#### Para Regresión:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coeficiente de determinación)
- **MAPE** (Mean Absolute Percentage Error)

#### Para Clasificación:
- **Accuracy** (Exactitud)
- **Precision** (Precisión)
- **Recall** (Sensibilidad)
- **F1-Score**
- **ROC AUC**
- **Matriz de confusión**

### 4.2 Visualizaciones de Resultados
- **Curvas de aprendizaje**: Loss vs épocas
- **Comparación predicción vs real**
- **Análisis de residuos**
- **Importancia de variables** (Random Forest, XGBoost)
- **Matrices de confusión** interactivas
- **Curvas ROC** para clasificación

### 4.3 Análisis Comparativo
- **Comparación entre modelos**
- **Tabla de rankings** por métrica
- **Análisis de trade-offs** (precisión vs velocidad)
- **Exportación de resultados** (CSV, JSON)

## 5. Sistema de Predicciones

### 5.1 Predicciones Individuales
- **Formulario de entrada** para nuevos datos
- **Validación de inputs** según modelo
- **Predicción instantánea**
- **Intervalos de confianza** (si aplica)

### 5.2 Predicciones por Lotes
- **Carga de CSV** con múltiples registros
- **Procesamiento batch** eficiente
- **Descarga de resultados** en CSV

### 5.3 Visualización Geográfica
- **Mapa interactivo** (Leaflet)
- **Predicciones por región**:
  - España y Francia
  - Códigos de color por variable
  - Tooltips con valores detallados
- **Variables visualizables**:
  - Temperatura
  - Humedad
  - Presión
  - Velocidad del viento
  - Precipitación

## 6. Gestión de Modelos

### 6.1 Biblioteca de Modelos
- **Listado de modelos** entrenados
- **Filtros** por tipo, fecha, rendimiento
- **Búsqueda** por nombre o descripción
- **Ordenamiento** por métricas

### 6.2 Operaciones con Modelos
- **Clonar modelo**: Duplicar configuración
- **Reentrenar**: Con mismos o nuevos datos
- **Exportar modelo**:
  - Archivo del modelo (.h5, .pkl, .pt)
  - Código Python reproducible
  - Configuración en JSON
- **Importar modelo**: Desde archivos externos
- **Eliminar modelo**: Con confirmación

### 6.3 Versionado
- **Historial de entrenamientos** por modelo
- **Comparación entre versiones**
- **Rollback** a versiones anteriores
- **Notas y comentarios** por versión

## 7. Exportación y Generación de Código

### 7.1 Exportación de Modelos
- **Formatos soportados**:
  - Keras/TensorFlow (.h5, SavedModel)
  - PyTorch (.pt, .pth)
  - Scikit-learn (.pkl, .joblib)
- **Metadatos incluidos**:
  - Configuración de preprocesamiento
  - Hiperparámetros
  - Métricas de evaluación

### 7.2 Generación de Código Python
- **Código completo y ejecutable**:
  - Imports necesarios
  - Carga y preprocesamiento de datos
  - Definición del modelo
  - Entrenamiento y evaluación
  - Guardado del modelo
- **Personalización**:
  - Comentarios explicativos
  - Docstrings
  - Logging configurable
- **Compatibilidad**:
  - Python 3.8+
  - Notebooks Jupyter
  - Scripts standalone

## 8. Características de Interfaz

### 8.1 Diseño y UX
- **Tema Cyberpunk/Neon**:
  - Colores vibrantes con efectos glow
  - Animaciones suaves
  - Glassmorphism effects
- **Responsive Design**:
  - Adaptable a móviles y tablets
  - Layouts flexibles
  - Touch-friendly

### 8.2 Internacionalización
- **Idiomas soportados**:
  - Francés (por defecto)
  - Español
  - Inglés
- **Cambio dinámico** sin recargar
- **Persistencia** de preferencia

### 8.3 Notificaciones y Feedback
- **SweetAlert2** para mensajes elegantes
- **Toasts** para notificaciones rápidas
- **Progress bars** para operaciones largas
- **Loading spinners** contextuales

## 9. Características Técnicas Avanzadas

### 9.1 Optimización de Rendimiento
- **Lazy loading** de datasets grandes
- **Paginación** en listados
- **Caché** de resultados frecuentes
- **Procesamiento asíncrono**

### 9.2 Seguridad
- **CSRF protection** en todas las formas
- **Validación de entrada** exhaustiva
- **Sanitización** de datos de usuario
- **Autenticación requerida** para operaciones

### 9.3 Extensibilidad
- **API RESTful** documentada
- **Webhooks** para eventos (opcional)
- **Plugins** para nuevos modelos
- **Temas** personalizables

## 10. Herramientas de Administración

### 10.1 Panel de Administración Django
- **Gestión de usuarios**
- **Monitoreo de recursos**
- **Logs de actividad**
- **Estadísticas de uso**

### 10.2 Mantenimiento
- **Limpieza de archivos** huérfanos
- **Respaldos automáticos** (configurables)
- **Monitoreo de espacio** en disco
- **Alertas de sistema**

## 11. Integraciones y APIs

### 11.1 API REST
- **Endpoints documentados** para todas las operaciones
- **Autenticación** por token o sesión
- **Rate limiting** configurable
- **Swagger/OpenAPI** documentation

### 11.2 Webhooks y Eventos
- **Eventos de entrenamiento** (inicio, fin, error)
- **Eventos de predicción**
- **Eventos de sistema**
- **Configuración flexible** de endpoints

### 11.3 Exportación de Datos
- **Formatos soportados**:
  - CSV
  - JSON
  - Excel (con pandas)
  - Parquet (big data)
- **Filtros y selección** de columnas
- **Compresión** opcional