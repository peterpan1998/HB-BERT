# -*- coding: utf-8 -*-
"""
Sample script using configuration file
Demonstrate how to import and use configuration parameters from config.py
"""

from config import TRAINING_CONFIG, SPIKING_CONFIG, FILE_PATHS, DATA_CONFIG, EVAL_CONFIG, DEVICE_CONFIG

def print_all_configs():
    print("=" * 50)
    print("allocation:")
    print("=" * 50)
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("allocation:")
    print("=" * 50) 
    for key, value in SPIKING_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("allocation:")
    print("=" * 50)
    for key, value in FILE_PATHS.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("allocation:")
    print("=" * 50)
    for key, value in DATA_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("allocation:")
    print("=" * 50)
    for key, value in EVAL_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("allocation:")
    print("=" * 50)
    for key, value in DEVICE_CONFIG.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print_all_configs()
