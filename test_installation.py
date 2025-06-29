#!/usr/bin/env python3
"""
Test script to verify the regression analyzer installation and basic functionality.
Run this to make sure everything is set up correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from google.adk.agents import Agent
        print("✅ Google ADK imported successfully")
    except ImportError as e:
        print(f"❌ Google ADK import failed: {e}")
        return False
    
    return True

def test_agent_creation():
    """Test that the regression analyzer agent can be created."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from regression_analyzer.agent import root_agent
        print(f"✅ Agent created: {root_agent.name}")
        print(f"   Model: {root_agent.model}")
        print(f"   Tools: {len(root_agent.tools)} tools available")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False

def test_analysis_tools():
    """Test that the analysis tools can be imported and basic functionality works."""
    print("\n🛠️ Testing analysis tools...")
    
    try:
        from regression_analyzer.analysis_tools import (
            load_and_preprocess_data,
            identify_top_factors,
            perform_regression_analysis,
            generate_formula_and_insights,
            create_analysis_summary
        )
        print("✅ All analysis tools imported successfully")
        
        # Test basic tool functionality with sample data
        sample_csv = """feature1,feature2,target
1,2,10
2,4,20
3,6,30
4,8,40
5,10,50"""
        
        result = load_and_preprocess_data(sample_csv, "target")
        if result["status"] == "success":
            print("✅ Basic data processing test passed")
            return True
        else:
            print(f"❌ Basic data processing test failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis tools test failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test that the environment is configured for ADK."""
    print("\n🌍 Testing environment configuration...")
    
    import os
    
    # Check for API key configuration
    if os.getenv('GOOGLE_API_KEY'):
        print("✅ GOOGLE_API_KEY found in environment")
        vertex_ai = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'TRUE').upper() == 'FALSE'
        print(f"✅ Using Google AI Studio: {vertex_ai}")
        return True
    elif os.getenv('GOOGLE_CLOUD_PROJECT'):
        print("✅ GOOGLE_CLOUD_PROJECT found in environment")
        vertex_ai = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper() == 'TRUE'
        print(f"✅ Using Vertex AI: {vertex_ai}")
        return True
    else:
        print("⚠️  No Google AI credentials found in environment")
        print("   Please set either:")
        print("   - GOOGLE_API_KEY (for Google AI Studio)")
        print("   - GOOGLE_CLOUD_PROJECT (for Vertex AI)")
        return False

def main():
    """Run all tests."""
    print("🧮 REGRESSION ANALYZER INSTALLATION TEST")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Package Imports", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Analysis Tools", test_analysis_tools),
        ("Environment Config", test_environment)
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your regression analyzer is ready to use!")
        print("\nNext steps:")
        print("1. Run the demo: python demo.py")
        print("2. Launch web UI: adk web")
        print("3. Use CLI: adk run .")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Set up your Google AI credentials (see README.md)")
        print("3. Install the package: pip install -e .")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 