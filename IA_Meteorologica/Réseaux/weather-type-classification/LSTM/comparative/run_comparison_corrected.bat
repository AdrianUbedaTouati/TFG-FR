@echo off
echo ================================================================================
echo COMPARACION DE ARQUITECTURAS CON MODELOS REENTRENADOS
echo ================================================================================
echo.
echo Este script compara:
echo 1. Modelo unico (4 clases directas)
echo 2. Arquitectura jerarquica con especialistas REENTRENADOS
echo.

python run_comparison_corrected.py

if errorlevel 1 (
    echo.
    echo ERROR durante la comparacion
    echo.
    echo Asegurese de que:
    echo 1. Los modelos reentrenados existen (ejecute primero retrain_specialists.py)
    echo 2. El entorno virtual esta activado
    echo 3. Todas las dependencias estan instaladas
    pause
    exit /b 1
)

echo.
echo Comparacion completada exitosamente.
echo Los resultados estan en: outputs\comparison_results_corrected\
pause