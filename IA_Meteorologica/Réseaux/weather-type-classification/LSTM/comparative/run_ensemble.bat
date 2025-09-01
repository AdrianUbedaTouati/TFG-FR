@echo off
echo ================================================================================
echo EVALUACION DE VOTING ENSEMBLE
echo ================================================================================
echo.
echo Este script evalua una arquitectura de Voting Ensemble que combina:
echo 1. Modelo unico (4 clases directas)
echo 2. Arquitectura jerarquica (3 modelos)
echo.
echo Estrategias disponibles:
echo - hard: Voting duro basado en predicciones
echo - soft: Promedio de probabilidades
echo - weighted: Promedio ponderado con pesos fijos
echo - weighted_confidence: Ponderado por confianza dinamica
echo - cascade: Estrategia adaptativa en cascada
echo - all: Evaluar todas y seleccionar la mejor
echo.

set /p use_corrected="Usar modelos especialistas corregidos? (s/n): "
set corrected_flag=

if /i "%use_corrected%"=="s" (
    set corrected_flag=--use-corrected
    echo Usando modelos CORREGIDOS
) else (
    echo Usando modelos ORIGINALES
)

echo.
set /p strategy="Ingrese estrategia (cascade/all/hard/soft/weighted/weighted_confidence): "

if "%strategy%"=="" set strategy=cascade

echo.
echo Ejecutando ensemble con estrategia: %strategy%
echo.

if "%strategy%"=="all" (
    python evaluate_ensemble.py --evaluate-all %corrected_flag%
) else (
    python evaluate_ensemble.py --strategy %strategy% %corrected_flag%
)

if errorlevel 1 (
    echo.
    echo ERROR durante la evaluacion del ensemble
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo EVALUACION COMPLETADA
echo ================================================================================
echo.
echo Resultados guardados en: outputs\ensemble_results\
echo.
echo Archivos generados:
echo - ensemble_analysis.png: Analisis visual completo
echo - ensemble_improvements.png: Graficos de mejora
echo - ensemble_report.txt: Reporte detallado
echo - confusion_matrices_comparison.png: Matrices de confusion
echo.
pause