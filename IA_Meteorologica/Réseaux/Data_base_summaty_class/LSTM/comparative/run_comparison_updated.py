#!/usr/bin/env python3
"""
Script actualizado para ejecutar la comparación con opción de usar modelos reentrenados
"""

import sys
import os
import argparse

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Comparación de arquitecturas de clasificación meteorológica')
    parser.add_argument('--use-corrected', action='store_true', 
                       help='Usar modelos especialistas reentrenados (corregidos)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directorio de salida personalizado')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SISTEMA DE COMPARACIÓN DE ARQUITECTURAS DE CLASIFICACIÓN METEOROLÓGICA")
    print("="*80)
    
    if args.use_corrected:
        print("\n⚡ Usando modelos especialistas REENTRENADOS")
        # Verificar que existen los modelos reentrenados
        if not (os.path.exists("outputs/retrained_a/a_corrected.pt") and 
                os.path.exists("outputs/retrained_b/b_corrected.pt")):
            print("\n❌ ERROR: No se encontraron modelos reentrenados.")
            print("   Ejecute primero: python retrain_specialists.py")
            sys.exit(1)
        
        # Importar y ejecutar la versión con modelos corregidos
        from run_comparison_corrected import main as main_corrected
        
        try:
            results_single, results_hier = main_corrected()
            print("\n✓ Evaluación con modelos REENTRENADOS completada")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n📊 Usando modelos especialistas ORIGINALES")
        # Importar y ejecutar la versión original
        from evaluate_architectures import main as main_original
        
        try:
            results_single, results_hier = main_original()
            print("\n✓ Evaluación con modelos ORIGINALES completada")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Mostrar ruta de resultados
    if args.use_corrected:
        output_path = "outputs/comparison_results_corrected/"
    else:
        output_path = "outputs/comparison_results/"
    
    if args.output_dir:
        output_path = args.output_dir
    
    print(f"\n✓ Los resultados se han guardado en: {output_path}")
    print("\nArchivos generados:")
    print("  - confusion_matrices_comparison.png")
    print("  - metrics_comparison.png")
    print("  - per_class_metrics_comparison.png")
    print("  - detailed_report.json")
    print("  - summary_report.txt")
    
    if args.use_corrected:
        print("  - corrected_models_report.txt")

if __name__ == "__main__":
    main()