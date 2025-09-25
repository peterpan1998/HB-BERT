# -*- coding: utf-8 -*-
"""
使用配置文件的示例脚本
演示如何从config.py中导入和使用配置参数
"""

from config import TRAINING_CONFIG, SPIKING_CONFIG, FILE_PATHS, DATA_CONFIG, EVAL_CONFIG, DEVICE_CONFIG

def print_all_configs():
    """打印所有配置参数"""
    print("=" * 50)
    print("训练配置参数:")
    print("=" * 50)
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("脉冲神经网络配置:")
    print("=" * 50) 
    for key, value in SPIKING_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("文件路径配置:")
    print("=" * 50)
    for key, value in FILE_PATHS.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("数据配置:")
    print("=" * 50)
    for key, value in DATA_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("评估配置:")
    print("=" * 50)
    for key, value in EVAL_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("设备配置:")
    print("=" * 50)
    for key, value in DEVICE_CONFIG.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print_all_configs()
