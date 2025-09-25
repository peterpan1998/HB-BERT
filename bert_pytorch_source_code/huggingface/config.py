# -*- coding: utf-8 -*-
"""
配置文件 - 存储所有训练参数和文件路径
"""

# 训练参数配置
TRAINING_CONFIG = {
    # 输出和日志配置
    'output_dir': '',
    'logging_dir': '',
    'logging_steps': 10,
    'metric_for_best_model': "eval_accuracy",
    
    # 训练超参数
    'num_train_epochs': 1,
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1 ,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'learning_rate': 2e-5,
    
    # 模型参数
    'num_labels': 3,  # MNLI任务的标签数
    'max_length': 128,
}

# 脉冲神经网络参数
SPIKING_CONFIG = {
    'alpha_spiking_q': "number",#0.52,
    'alpha_origin_q': "number", #1.07, 
}

# 文件路径配置
FILE_PATHS = {
    # 数据文件路径
    'train_file': " ",
    'validation_file': " ",
    'test_file': " ",
    # 模型路径
    'local_model_path': ' ',
}

# 数据预处理配置
DATA_CONFIG = {
    'text_columns': ['premise', 'hypothesis'],  # MNLI数据集的文本列
    'label_column': 'label',
    'padding': 'max_length',
    'truncation': True,
    'max_length': TRAINING_CONFIG['max_length'],
}

# 评估配置
EVAL_CONFIG = {
    'eval_strategy': "epoch",
    'save_strategy': "epoch",
    'load_best_model_at_end': True,
    'early_stopping_patience': 8,  # 早停耐心值
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,  # 是否使用CUDA
}
