"""
Script para ejecutar el proceso completo de corrección de los modelos especialistas
"""
import os
import subprocess
import sys

def run_command(cmd, description):
    """Ejecuta un comando y muestra el progreso"""
    print(f"\n{'='*60}")
    print(f"[EJECUTANDO] {description}")
    print(f"{'='*60}")
    print(f"Comando: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR al ejecutar: {description}")
        print(f"Código de salida: {result.returncode}")
        return False
    
    print(f"\n✅ {description} completado exitosamente")
    return True

def main():
    print("="*80)
    print("PROCESO DE CORRECCIÓN DE MODELOS ESPECIALISTAS")
    print("="*80)
    print("\nEste proceso:")
    print("1. Preparará datasets corregidos con todas las clases")
    print("2. Reentrenará los modelos especialistas")
    print("3. Evaluará el rendimiento mejorado")
    print("\nEsto puede tomar varios minutos...")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("prepare_corrected_datasets.py"):
        print("\n❌ ERROR: Debe ejecutar este script desde el directorio comparative/")
        return
    
    # Paso 1: Preparar datasets corregidos
    if not run_command(
        f"{sys.executable} prepare_corrected_datasets.py",
        "Preparación de datasets corregidos"
    ):
        return
    
    # Verificar que se crearon los datasets
    if not os.path.exists("data_corrected/weather_classification_CloudySunny_corrected.csv"):
        print("\n❌ ERROR: No se crearon los datasets corregidos")
        return
    
    print("\n✅ Datasets corregidos creados exitosamente")
    
    # Paso 2: Reentrenar modelos
    print("\n" + "="*80)
    print("INICIANDO REENTRENAMIENTO DE MODELOS")
    print("="*80)
    print("\n⚠️  Este proceso puede tomar 10-20 minutos...")
    
    if not run_command(
        f"{sys.executable} retrain_specialists.py",
        "Reentrenamiento de especialistas"
    ):
        return
    
    # Verificar que se crearon los modelos
    if not os.path.exists("outputs/retrained_a/a_corrected.pt"):
        print("\n❌ ERROR: No se crearon los modelos reentrenados")
        return
    
    print("\n✅ Modelos reentrenados exitosamente")
    
    # Paso 3: Evaluar arquitecturas
    print("\n" + "="*80)
    print("EVALUANDO ARQUITECTURAS")
    print("="*80)
    
    if not run_command(
        f"{sys.executable} evaluate_architectures_corrected.py",
        "Evaluación de arquitecturas corregidas"
    ):
        return
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("PROCESO COMPLETADO")
    print("="*80)
    print("\n✅ Todos los pasos completados exitosamente")
    print("\nResultados guardados en:")
    print("  - Datasets corregidos: data_corrected/")
    print("  - Modelos reentrenados: outputs/retrained_*/")
    print("  - Evaluación comparativa: outputs/comparison_corrected/")
    print("\nRevise el archivo outputs/comparison_corrected/comparison_report.txt")
    print("para ver el análisis detallado de mejoras.")

if __name__ == "__main__":
    main()