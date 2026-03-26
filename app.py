"""
Entry point for Streamlit Cloud — delegates to ui/main.py.

Run with: streamlit run app.py
"""

import sys
import os

# Ensure project root is on the path so 'engines' and 'ui' packages resolve.
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the actual app.
from ui.main import *  # noqa: F401,F403
