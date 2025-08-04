#!/usr/bin/env python3
"""
Modal Backdrop Checker
Scans templates for potential modal backdrop issues
"""

import os
import re
import sys
from pathlib import Path

# Patterns to check for
BACKDROP_PATTERNS = [
    r'backdrop\s*:\s*["\']?static["\']?',
    r'backdrop\s*:\s*true',
    r'data-bs-backdrop\s*=\s*["\']static["\']',
    r'data-backdrop\s*=\s*["\']static["\']',
]

# Files to exclude
EXCLUDE_PATTERNS = ['.swp', '.pyc', '__pycache__', '.git']

def check_file(filepath):
    """Check a single file for backdrop issues"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for i, line in enumerate(lines, 1):
            for pattern in BACKDROP_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'file': filepath,
                        'line': i,
                        'content': line.strip(),
                        'pattern': pattern
                    })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return issues

def scan_templates(base_path='templates'):
    """Scan all template files"""
    all_issues = []
    files_checked = 0
    
    for root, dirs, files in os.walk(base_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(exc in d for exc in EXCLUDE_PATTERNS)]
        
        for file in files:
            if file.endswith(('.html', '.js')) and not any(exc in file for exc in EXCLUDE_PATTERNS):
                filepath = os.path.join(root, file)
                files_checked += 1
                issues = check_file(filepath)
                all_issues.extend(issues)
    
    return all_issues, files_checked

def main():
    """Main function"""
    print("🔍 Scanning for modal backdrop issues...")
    print("-" * 50)
    
    # Scan templates
    issues, files_checked = scan_templates()
    
    if issues:
        print(f"❌ Found {len(issues)} potential backdrop issues in {files_checked} files:\n")
        
        for issue in issues:
            print(f"📄 {issue['file']}")
            print(f"   Line {issue['line']}: {issue['content']}")
            print(f"   Pattern: {issue['pattern']}")
            print()
        
        print("\n⚠️  Please fix these issues to prevent gray overlay problems!")
        print("📖 See MODAL_GUIDELINES.md for correct implementation")
        return 1
    else:
        print(f"✅ No backdrop issues found in {files_checked} files!")
        print("🎉 All modals are correctly configured")
        return 0

if __name__ == "__main__":
    sys.exit(main())