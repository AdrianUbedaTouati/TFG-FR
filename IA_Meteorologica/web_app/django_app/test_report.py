#!/usr/bin/env python
"""
Test script to verify report generation
Run from Django app directory: python test_report.py
"""

import os
import sys
import django

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_prediction.settings')
django.setup()

from ml_trainer.models import Dataset
from ml_trainer.views.report_views import DatasetReportView
from django.test import RequestFactory

def test_report_generation():
    """Test the enhanced report generation"""
    try:
        # Get the first dataset
        dataset = Dataset.objects.first()
        if not dataset:
            print("No datasets found in the database")
            return
        
        print(f"Testing report generation for dataset: {dataset.name}")
        
        # Create a fake request
        factory = RequestFactory()
        request = factory.get(f'/api/datasets/{dataset.pk}/report/')
        
        # Create view instance and generate report
        view = DatasetReportView()
        response = view.get(request, pk=dataset.pk)
        
        if response.status_code == 200:
            # Save the report to a file for inspection
            output_file = f"test_report_{dataset.name}.html"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Report generated successfully! Saved to: {output_file}")
            print(f"  File size: {len(response.content):,} bytes")
        else:
            print(f"✗ Error generating report: Status code {response.status_code}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_report_generation()