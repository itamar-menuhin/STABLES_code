# main.py

"""
Main entry point of the application.

This script manages the execution order of the other scripts in the project:
1. create_hy.py - Responsible for creating or handling hybrid data structures.
2. utils.py - Provides utility functions for various tasks.
3. calc_features.py - Calculates features based on the processed data.

The scripts are executed in the following order:
- create_hy.py
- utils.py
- calc_features.py
"""

from create_hy import *
from utils import *
from calc_features import *

def main():
    # Step 1: Execute functions from create_hy.py
    print("Running create_hy...")
    create_hybrid_data()  # Assuming this is a function in create_hy.py

    # Step 2: Execute functions from utils.py
    print("Running utils...")
    perform_util_tasks()  # Assuming this is a function in utils.py

    # Step 3: Execute functions from calc_features.py
    print("Calculating features...")
    results = calculate_features()  # Assuming this is a function in calc_features.py
    print("Feature calculation results:", results)

if __name__ == "__main__":
    main()