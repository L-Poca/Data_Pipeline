#!/usr/bin/env python3
"""Script to verify that the package is installable with pip install -e .

This script tests that:
1. The package can be installed
2. The src module can be imported
3. Basic package metadata is accessible
"""

import subprocess
import sys

def test_package_install():
    """Test that the package appears in pip list."""
    result = subprocess.run(
        ['pip', 'list'],
        capture_output=True,
        text=True
    )
    
    if 'data-pipeline' in result.stdout:
        print("✅ Package 'data-pipeline' is installed")
        return True
    else:
        print("❌ Package 'data-pipeline' is NOT installed")
        return False

def test_basic_import():
    """Test basic import of the src module."""
    try:
        import src
        print(f"✅ src module imported successfully, version: {src.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import src module: {e}")
        return False

def test_package_metadata():
    """Test that package metadata is accessible."""
    try:
        import src
        assert hasattr(src, '__version__')
        assert hasattr(src, '__author__')
        print(f"✅ Package metadata accessible")
        print(f"   Version: {src.__version__}")
        print(f"   Author: {src.__author__}")
        return True
    except Exception as e:
        print(f"❌ Failed to access package metadata: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing pip installable package setup...\n")
    
    tests = [
        ("Package Installation", test_package_install),
        ("Basic Import", test_basic_import),
        ("Package Metadata", test_package_metadata),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print('='*60)
        results.append(test_func())
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The package is correctly installable.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        sys.exit(1)
