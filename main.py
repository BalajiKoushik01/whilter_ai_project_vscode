"""
Vehicle Detection - Main Script
==============================

This is a wrapper script that maintains backward compatibility with the original main.py.
It imports and calls the main function from the vehicle_detection package.
"""

import sys
from vehicle_detection.main import main

if __name__ == "__main__":
    sys.exit(main())
