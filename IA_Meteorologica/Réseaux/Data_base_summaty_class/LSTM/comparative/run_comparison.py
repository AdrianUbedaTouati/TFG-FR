#!/usr/bin/env python3
"""
Script para ejecutar la comparación de arquitecturas de forma sencilla
"""

import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_architectures import main

if __name__ == "__main__":
    print("="*80)
    print("SISTEMA DE COMPARACIÓN DE ARQUITECTURAS DE CLASIFICACIÓN METEOROLÓGICA")
    print("="*80)
    print("\nComparando:")
    print("1. Modelo único (4 clases directas)")
    print("2. Arquitectura jerárquica (3 modelos especializados)")
    print("\n" + "-"*80)
    
    try:
        results_single, results_hier = main()
        print("\n✓ Evaluación completada exitosamente")
        print(f"✓ Los resultados se han guardado en: outputs/comparison_results/")
        print("\nArchivos generados:")
        print("  - confusion_matrices_comparison.png")
        print("  - confusion_matrices_normalized.png")
        print("  - metrics_comparison.png")
        print("  - per_class_metrics_comparison.png")
        print("  - error_distribution.png")
        print("  - detailed_report.json")
        print("  - summary_report.txt")
        
    except Exception as e:
        print(f"\n✗ Error durante la evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)