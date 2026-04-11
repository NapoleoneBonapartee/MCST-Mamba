"""
Test script for MCSTWeather model
验证模型结构和基本功能
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 测试 WeatherProcessor
print("=" * 50)
print("Testing WeatherProcessor...")
print("=" * 50)

# 创建模拟天气数据
np.random.seed(42)
dates = pd.date_range(start='2012-03-01', periods=100, freq='5min')
weather_data = pd.DataFrame({
    'TMAX': 15 + np.random.randn(100) * 5,
    'TMIN': 10 + np.random.randn(100) * 5,
    'AWBT': 12 + np.random.randn(100) * 4,
    'RHAV': 60 + np.random.randn(100) * 20,
    'RHMX': 80 + np.random.randn(100) * 10,
    'RHMN': 40 + np.random.randn(100) * 10,
    'WSF2': 5 + np.random.randn(100) * 3,
    'WSF5': 8 + np.random.randn(100) * 4,
    'AWND': 6 + np.random.randn(100) * 3,
    'ADPT': 10200 + np.random.randn(100) * 50,
    'ASTP': 10080 + np.random.randn(100) * 50,
    'ASLP': 10150 + np.random.randn(100) * 30,
    'PRCP': np.random.choice([0, 0, 0, 0, 0.5, 2.0, 5.0], 100),
    'WDF2': np.random.uniform(0, 360, 100),
    'WT01': np.nan,
    'WT02': np.nan,
}, index=dates)

# 保存模拟数据
weather_data.to_csv('./raw_data/METR_LA/weather_test.csv')
print("Mock weather data saved to ./raw_data/METR_LA/weather_test.csv")

# 测试 WeatherProcessor
from libcity.model.traffic_speed_prediction.MCSTWeather import WeatherProcessor

processor = WeatherProcessor('./raw_data/METR_LA/weather_test.csv')
processor.load_data()
print(f"\nWeather features shape: {processor.weather_features.shape}")
print(f"Weather feature names: {processor.feature_names}")
print(f"Number of weather features: {processor.num_weather_features}")

# 测试获取特征
test_timestamps = [dates[10], dates[20], dates[30]]
features = processor.get_features_at_timestamps(test_timestamps)
print(f"\nFeatures at test timestamps shape: {features.shape}")
print("WeatherProcessor test passed!\n")

# 测试 MambaWeather 模块
print("=" * 50)
print("Testing MambaWeather module...")
print("=" * 50)

from libcity.model.traffic_speed_prediction.MambaWeather import MambaWeather, SimpleMambaWeatherBlock

# 测试 MambaWeather
batch_size = 2
seq_len = 12
d_model = 32
weather_embed_dim = 16

mamba_weather = MambaWeather(
    d_model=d_model,
    d_state=16,
    d_conv=4,
    expand=2,
    weather_embed_dim=weather_embed_dim
)

# 创建测试输入
hidden_states = torch.randn(batch_size, seq_len, d_model)
weather_embed = torch.randn(batch_size, seq_len, weather_embed_dim)

# 前向传播
output = mamba_weather(hidden_states, weather_embed)
print(f"Input shape: {hidden_states.shape}")
print(f"Weather embed shape: {weather_embed.shape}")
print(f"Output shape: {output.shape}")
assert output.shape == hidden_states.shape, "Output shape mismatch!"
print("MambaWeather test passed!\n")

# 测试 SimpleMambaWeatherBlock
print("Testing SimpleMambaWeatherBlock...")
block = SimpleMambaWeatherBlock(
    d_model=d_model,
    d_state=16,
    d_conv=4,
    expand=2,
    weather_embed_dim=weather_embed_dim,
    dropout=0.1
)
output = block(hidden_states, weather_embed)
print(f"Block output shape: {output.shape}")
assert output.shape == hidden_states.shape, "Block output shape mismatch!"
print("SimpleMambaWeatherBlock test passed!\n")

# 测试 MCSTWeather 模型（简化版）
print("=" * 50)
print("Testing MCSTWeather model initialization...")
print("=" * 50)

from libcity.model.traffic_speed_prediction.MCSTWeather import MCSTWeather

# 创建模拟配置
class MockConfig:
    def __init__(self):
        self.config = {
            'input_window': 12,
            'output_window': 12,
            'device': 'cpu',
            'add_time_in_day': True,
            'add_day_in_week': False,
            'steps_per_day': 288,
            'input_embedding_dim': 16,
            'tod_embedding_dim': 16,
            'dow_embedding_dim': 16,
            'spatial_embedding_dim': 8,
            'adaptive_embedding_dim': 16,
            'weather_embed_dim': 16,
            'd_model': 32,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.1,
            'use_weather': True,
            'weather_file': './raw_data/METR_LA/weather_test.csv',
            'dataset': 'METR_LA',
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class MockDataFeature:
    def __init__(self):
        self.data = {
            'num_nodes': 10,
            'feature_dim': 1,
            'output_dim': 1,
        }
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def get_scaler(self):
        return None

config = MockConfig()
data_feature = MockDataFeature()

try:
    model = MCSTWeather(config, data_feature)
    print(f"Model initialized successfully!")
    print(f"Model device: {model.device}")
    print(f"Number of nodes: {model.num_nodes}")
    print(f"Weather processor initialized: {model.weather_processor is not None}")
    if model.weather_processor:
        print(f"Number of weather features: {model.weather_processor.num_weather_features}")
    
    # 测试前向传播
    batch = {
        'X': torch.randn(2, 12, 10, 1),  # (batch, window, nodes, features)
    }
    output = model(batch)
    print(f"\nForward pass test:")
    print(f"Input shape: {batch['X'].shape}")
    print(f"Output shape: {output.shape}")
    expected_output_shape = (2, 12, 10, 1)  # (batch, output_window, nodes, output_dim)
    assert output.shape == expected_output_shape, f"Output shape {output.shape} != expected {expected_output_shape}"
    print("MCSTWeather model test passed!\n")
    
except Exception as e:
    print(f"Error during MCSTWeather test: {e}")
    import traceback
    traceback.print_exc()

# 清理测试文件
import os
if os.path.exists('./raw_data/METR_LA/weather_test.csv'):
    os.remove('./raw_data/METR_LA/weather_test.csv')
    print("Cleaned up test weather file.")

print("=" * 50)
print("All tests completed!")
print("=" * 50)
