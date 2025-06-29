#!/usr/bin/env python3
"""
Launch script for MultiRegal - Automatic Multivariable Regression Analysis Calculator

This script launches the Streamlit web interface for the regression analyzer.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    print("🔮 Starting MultiRegal - Automatic Regression Analysis Calculator")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Please run this script from the project root directory.")
        return 1
    
    # Set Streamlit configuration
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    try:
        # Launch Streamlit
        print("🚀 Launching web interface...")
        print("📱 Open your browser and go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down MultiRegal...")
        return 0
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 