@echo off
echo ================================================================================
echo LIMPIEZA DE ARCHIVOS DE SALIDA
echo ================================================================================
echo.
echo Este script eliminara los archivos de salida antiguos para liberar espacio.
echo Se mantendran los modelos reentrenados.
echo.
set /p confirm="Esta seguro que desea continuar? (s/n): "

if /i not "%confirm%"=="s" (
    echo.
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Limpiando directorios de salida...

REM Eliminar salidas de comparacion (mantener solo la mas reciente)
if exist outputs\comparison_results rd /s /q outputs\comparison_results
if exist outputs\comparison_corrected rd /s /q outputs\comparison_corrected

REM Limpiar logs antiguos
if exist logs\*.log del /q logs\*.log

echo.
echo Archivos limpiados. Se mantuvieron:
echo - Modelos reentrenados (outputs\retrained_*)
echo - Resultados mas recientes
echo.
pause