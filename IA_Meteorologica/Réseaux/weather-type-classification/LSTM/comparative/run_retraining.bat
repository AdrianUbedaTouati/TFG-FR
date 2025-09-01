@echo off
echo ================================================================================
echo PROCESO DE CORRECCION DE MODELOS ESPECIALISTAS
echo ================================================================================
echo.
echo Este proceso:
echo 1. Preparara datasets corregidos con todas las clases
echo 2. Reentrenara los modelos especialistas
echo 3. Evaluara el rendimiento mejorado
echo.
echo Esto puede tomar varios minutos...
echo.
pause

echo.
echo ============================================================
echo [EJECUTANDO] Preparacion de datasets corregidos
echo ============================================================
python prepare_corrected_datasets.py
if errorlevel 1 (
    echo.
    echo ERROR: Fallo la preparacion de datasets
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [EJECUTANDO] Reentrenamiento de especialistas
echo ============================================================
echo.
echo ADVERTENCIA: Este proceso puede tomar 10-20 minutos...
echo.
python retrain_specialists.py
if errorlevel 1 (
    echo.
    echo ERROR: Fallo el reentrenamiento
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [EJECUTANDO] Evaluacion de arquitecturas corregidas
echo ============================================================
python evaluate_architectures_corrected.py
if errorlevel 1 (
    echo.
    echo ERROR: Fallo la evaluacion
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo PROCESO COMPLETADO
echo ================================================================================
echo.
echo Todos los pasos completados exitosamente
echo.
echo Resultados guardados en:
echo   - Datasets corregidos: data_corrected\
echo   - Modelos reentrenados: outputs\retrained_*\
echo   - Evaluacion comparativa: outputs\comparison_corrected\
echo.
echo Revise el archivo outputs\comparison_corrected\comparison_report.txt
echo para ver el analisis detallado de mejoras.
echo.
pause