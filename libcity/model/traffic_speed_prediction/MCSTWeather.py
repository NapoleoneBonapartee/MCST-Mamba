"""
MCSTWeather: Multi-Channel Spatio-Temporal Mamba with Weather Integration
基于 MCSTMamba_optimized，融合天气数据进行交通预测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from libcity.model.traffic_speed_prediction.MambaWeather import SimpleMambaWeatherBlock


class WeatherProcessor:
    """天气数据处理器：处理缺失值、构建衍生特征，支持按时间特征查询"""
    
    def __init__(self, weather_file_path, steps_per_day=288):
        self.weather_file_path = weather_file_path
        self.steps_per_day = steps_per_day  # 每天的时间步数（5分钟间隔=288）
        self.weather_df = None
        self.weather_features = None
        self.feature_names = None
        # 时间到天气特征的映射
        self.time_to_features = {}  # {(day_of_year, time_idx): feature_vector}
        self.feature_array = None  # 按时间顺序排列的特征数组
        self.start_time = None
        self.end_time = None
        self._logger = getLogger()
        
    def load_data(self):
        """加载天气数据并构建时间映射"""
        if not os.path.exists(self.weather_file_path):
            raise FileNotFoundError(f"Weather file not found: {self.weather_file_path}")
        
        self.weather_df = pd.read_csv(self.weather_file_path, index_col=0, parse_dates=True)
        self.start_time = self.weather_df.index.min()
        self.end_time = self.weather_df.index.max()
        
        self._process_missing_values()
        self._build_derived_features()
        self._build_time_mapping()
        return self
    
    def _process_missing_values(self):
        """处理缺失值：前向填充 + 后向填充"""
        # 数值列：先线性插值，再前向填充，最后后向填充
        numeric_cols = self.weather_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 线性插值
            self.weather_df[col] = self.weather_df[col].interpolate(method='linear', limit_direction='both')
            # 前向填充（处理开头的缺失值）
            self.weather_df[col] = self.weather_df[col].ffill()
            # 后向填充（处理末尾的缺失值）
            self.weather_df[col] = self.weather_df[col].bfill()
            # 剩余缺失值用均值填充
            self.weather_df[col] = self.weather_df[col].fillna(self.weather_df[col].mean())
        
        # 类别列：用众数填充
        categorical_cols = self.weather_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.weather_df[col] = self.weather_df[col].fillna(self.weather_df[col].mode()[0] if not self.weather_df[col].mode().empty else 0)
    
    def _build_derived_features(self):
        """构建天气衍生特征"""
        df = self.weather_df.copy()
        derived_features = pd.DataFrame(index=df.index)
        
        # 1. 温度相关特征（如果存在）
        if 'TMAX' in df.columns and 'TMIN' in df.columns:
            # 日温差
            derived_features['temp_range'] = df['TMAX'] - df['TMIN']
            # 平均温度
            derived_features['temp_avg'] = (df['TMAX'] + df['TMIN']) / 2
        
        if 'AWBT' in df.columns and 'TMAX' in df.columns:
            # 体感温度差
            derived_features['apparent_temp_diff'] = df['AWBT'] - df['TMAX']
        
        # 2. 湿度相关特征
        if 'RHAV' in df.columns:
            # 湿度等级（低、中、高）
            derived_features['humidity_level'] = pd.cut(df['RHAV'], bins=[0, 30, 60, 100], labels=[0, 1, 2]).astype(float)
            # 湿度变化率（一阶差分）
            derived_features['humidity_change'] = df['RHAV'].diff().fillna(0)
        
        if 'RHMX' in df.columns and 'RHMN' in df.columns:
            # 湿度范围
            derived_features['humidity_range'] = df['RHMX'] - df['RHMN']
        
        # 3. 风速相关特征
        if 'WSF2' in df.columns and 'WSF5' in df.columns:
            # 风速比（短时 gust 与平均风速比）
            derived_features['wind_gust_ratio'] = df['WSF5'] / (df['WSF2'] + 1e-6)
            # 平均风速
            derived_features['wind_avg'] = (df['WSF2'] + df['WSF5']) / 2
        
        if 'AWND' in df.columns:
            derived_features['wind_speed'] = df['AWND']
        
        # 4. 气压相关特征
        if 'ADPT' in df.columns and 'ASTP' in df.columns:
            # 站压与海平面气压差
            derived_features['pressure_diff'] = df['ASTP'] - df['ADPT']
        
        if 'ASLP' in df.columns:
            # 气压变化率
            derived_features['pressure_change'] = df['ASLP'].diff().fillna(0)
            # 气压等级（用于判断天气稳定性）
            derived_features['pressure_level'] = pd.cut(df['ASLP'], bins=[0, 1010, 1020, 1100], labels=[0, 1, 2]).astype(float)
        
        # 5. 降水相关特征
        if 'PRCP' in df.columns:
            # 是否有降水
            derived_features['has_precipitation'] = (df['PRCP'] > 0).astype(float)
            # 降水强度等级
            derived_features['precipitation_level'] = pd.cut(df['PRCP'], bins=[-0.1, 0, 2.5, 10, 1000], labels=[0, 1, 2, 3]).astype(float)
        
        # 6. 风向特征（转换为正弦/余弦编码）
        if 'WDF2' in df.columns:
            derived_features['wind_dir_sin'] = np.sin(np.radians(df['WDF2']))
            derived_features['wind_dir_cos'] = np.cos(np.radians(df['WDF2']))
        
        # 7. 天气现象特征（WT系列）
        wt_cols = [col for col in df.columns if col.startswith('WT')]
        if wt_cols:
            # 统计同时发生的天气现象数量
            derived_features['weather_phenomena_count'] = df[wt_cols].notna().sum(axis=1).astype(float)
            # 是否有恶劣天气（雾、雨、雪等）
            severe_weather_cols = [c for c in wt_cols if any(x in c for x in ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06'])]
            if severe_weather_cols:
                derived_features['has_severe_weather'] = df[severe_weather_cols].notna().any(axis=1).astype(float)
        
        # 8. 时间特征
        derived_features['hour'] = df.index.hour
        derived_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        derived_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 填充可能的缺失值
        derived_features = derived_features.fillna(0)
        
        self.weather_features = derived_features
        self.feature_names = list(derived_features.columns)
        self.num_weather_features = len(self.feature_names)
        
        return self
    
    def _build_time_mapping(self):
        """构建时间到特征的映射，便于快速查询"""
        # 将时间戳转换为 (day_of_year, time_idx) 的键
        for i, timestamp in enumerate(self.weather_features.index):
            day_of_year = timestamp.dayofyear
            # 计算当天的时间索引（0 到 steps_per_day-1）
            minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
            time_idx = int(minutes_since_midnight / (24 * 60 / self.steps_per_day))
            time_idx = min(time_idx, self.steps_per_day - 1)  # 确保不越界
            
            key = (day_of_year, time_idx)
            self.time_to_features[key] = self.weather_features.iloc[i].values
        
        # 也保存为数组形式，支持索引访问
        self.feature_array = self.weather_features.values
        self._logger.info(f"Built time mapping for {len(self.time_to_features)} timestamps")
        
    def get_features_by_time(self, day_of_year, time_idx):
        """
        根据天数和时间索引获取天气特征
        Args:
            day_of_year: 一年中的第几天 (1-366)
            time_idx: 当天的时间索引 (0 到 steps_per_day-1)
        Returns:
            numpy array: 天气特征向量
        """
        key = (day_of_year, time_idx)
        if key in self.time_to_features:
            return self.time_to_features[key]
        
        # 如果找不到，找到最近的时间
        # 先尝试同一天的其他时间
        for offset in [0] + list(range(1, self.steps_per_day)):
            for sign in [1, -1]:
                new_time_idx = time_idx + sign * offset
                if 0 <= new_time_idx < self.steps_per_day:
                    new_key = (day_of_year, new_time_idx)
                    if new_key in self.time_to_features:
                        return self.time_to_features[new_key]
        
        # 如果还是找不到，返回默认特征（均值）
        if self.feature_array is not None:
            return self.feature_array.mean(axis=0)
        return None
    
    def normalize_features(self, scaler=None):
        """标准化天气特征"""
        if scaler is None:
            # 使用 Z-score 标准化
            self.feature_mean = self.weather_features.mean()
            self.feature_std = self.weather_features.std().replace(0, 1)  # 避免除零
            self.weather_features = (self.weather_features - self.feature_mean) / self.feature_std
        else:
            self.weather_features = scaler.transform(self.weather_features)
        return self


class MCSTWeather(AbstractTrafficStateModel):
    """
    MCST-Mamba with Weather Integration
    融合天气数据的时空交通预测模型
    """
    
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # 保存 config 供后续使用
        self.config = config
        
        # 基础数据特征
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        
        # 模型配置
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self._logger = getLogger()
        
        # 时间特征配置
        self.add_time_in_day = config.get("add_time_in_day", False)
        self.add_day_in_week = config.get("add_day_in_week", False)
        self.steps_per_day = config.get("steps_per_day", 288)
        
        # 嵌入维度配置
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24) if self.add_time_in_day else 0
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24) if self.add_day_in_week else 0
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 16)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)
        
        # 天气相关配置
        self.weather_embed_dim = config.get('weather_embed_dim', 64)
        self.weather_file = config.get('weather_file', None)  # 天气文件路径
        self.use_weather = config.get('use_weather', True)
        
        # 计算模型总维度（不含天气，天气单独处理）
        self.model_dim = (
            self.input_embedding_dim +
            self.tod_embedding_dim +
            self.dow_embedding_dim +
            self.spatial_embedding_dim +
            self.adaptive_embedding_dim
        )
        
        # 创建嵌入层
        self.input_proj = nn.Linear(self.feature_dim, self.input_embedding_dim)
        
        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        
        # 空间嵌入
        self.spatial_embedding = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.spatial_embedding)
        
        # 自适应嵌入
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.input_window, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        
        # Mamba 参数
        self.d_model = config.get('d_model', 96)
        self.d_state = config.get('d_state', 32)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # 输入投影到 Mamba 维度
        self.mamba_input_proj = nn.Linear(self.model_dim, self.d_model)
        
        # 天气数据处理器
        self.weather_processor = None
        if self.use_weather:
            self._setup_weather_processor()
        
        # 天气特征嵌入层
        if self.use_weather and self.weather_processor is not None:
            self.weather_feature_proj = nn.Sequential(
                nn.Linear(self.weather_processor.num_weather_features, self.weather_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.weather_embed_dim * 2, self.weather_embed_dim)
            )
            # 为每个节点复制天气嵌入
            self.weather_spatial_expand = nn.Linear(self.weather_embed_dim, self.weather_embed_dim)
        
        # 时空 MambaWeather 块
        self._logger.info("Building MCSTWeather model with weather integration")
        
        # 时序处理块
        self.temporal_block = SimpleMambaWeatherBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            weather_embed_dim=self.weather_embed_dim,
            dropout=self.dropout
        )
        
        # 空间处理块
        self.spatial_block = SimpleMambaWeatherBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            weather_embed_dim=self.weather_embed_dim,
            dropout=self.dropout
        )
        
        # 组合权重
        self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))
        
        # 时序投影：将 input_window 映射到 output_window
        self.temporal_proj = nn.Linear(self.input_window, self.output_window)
        
        # 输出投影
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
        # 预计算时间嵌入
        self._precompute_time_embeddings()
        
        # 移动到设备
        self.to(self.device)
        
        self._logger.info(f"MCSTWeather model initialized with weather_embed_dim={self.weather_embed_dim}")
    
    def _setup_weather_processor(self):
        """设置天气数据处理器"""
        # 尝试自动查找天气文件
        if self.weather_file is None:
            # 尝试从数据集路径推断
            dataset_name = self.config.get('dataset', 'METR_LA')
            possible_paths = [
                f'./raw_data/{dataset_name}/weather.csv',
                f'./raw_data/weather/{dataset_name.lower()}_weather.csv',
                f'./raw_data/weather/noaa_weather_5min.csv',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.weather_file = path
                    break
        
        if self.weather_file and os.path.exists(self.weather_file):
            try:
                self.weather_processor = WeatherProcessor(self.weather_file).load_data()
                self.weather_processor.normalize_features()
                self._logger.info(f"Weather data loaded from {self.weather_file}")
                self._logger.info(f"Weather features: {self.weather_processor.feature_names}")
            except Exception as e:
                self._logger.warning(f"Failed to load weather data: {e}")
                self.weather_processor = None
        else:
            self._logger.warning(f"Weather file not found: {self.weather_file}")
            self.weather_processor = None
    
    def _precompute_time_embeddings(self):
        """预计算时间嵌入索引"""
        if self.add_time_in_day:
            tod = torch.linspace(0, 0.99, self.input_window, device=self.device)
            self.register_buffer('tod_indices', (tod * self.steps_per_day).long().clamp_(0, self.steps_per_day - 1))
        else:
            self.tod_indices = None
        
        if self.add_day_in_week:
            dow = torch.arange(0, self.input_window, device=self.device) % 7
            self.register_buffer('dow_indices', dow.long().clamp_(0, 6))
        else:
            self.dow_indices = None
    
    def _get_weather_embedding(self, batch_size, time_idx_list=None):
        """
        获取天气嵌入 - 简化的实现，直接使用 tod_indices
        Args:
            batch_size: batch大小
            time_idx_list: 时间索引列表（可选，如果不提供则使用 tod_indices）
        Returns:
            weather_embed: (batch_size, input_window, num_nodes, weather_embed_dim)
        """
        if not self.use_weather or self.weather_processor is None:
            # 返回零向量
            return torch.zeros(batch_size, self.input_window, self.num_nodes, self.weather_embed_dim, device=self.device)
        
        # 获取时间索引列表
        if time_idx_list is None or len(time_idx_list) == 0:
            # 使用预计算的 tod_indices
            if self.tod_indices is not None:
                time_idx_list = self.tod_indices.cpu().tolist()
            else:
                time_idx_list = list(range(self.input_window))
        
        # 确保 time_idx_list 是整数列表
        time_idx_list = [int(t) % self.steps_per_day for t in time_idx_list[:self.input_window]]
        
        # 如果长度不够，循环使用
        while len(time_idx_list) < self.input_window:
            time_idx_list.append(time_idx_list[-1] + 1 if time_idx_list else 0)
        time_idx_list = time_idx_list[:self.input_window]
        
        # 根据时间索引获取天气特征
        weather_features_list = []
        for time_idx in time_idx_list:
            # 使用 day_of_year=1 作为默认
            features = self.weather_processor.get_features_by_time(
                day_of_year=1,
                time_idx=time_idx % self.steps_per_day
            )
            weather_features_list.append(features)
        
        weather_features = np.stack(weather_features_list, axis=0)  # (input_window, num_features)
        weather_tensor = torch.tensor(weather_features, dtype=torch.float32, device=self.device)
        
        # 投影到嵌入维度
        weather_embed = self.weather_feature_proj(weather_tensor)  # (input_window, weather_embed_dim)
        
        # 扩展到batch和nodes维度
        weather_embed = weather_embed.unsqueeze(0).unsqueeze(2)  # (1, input_window, 1, weather_embed_dim)
        weather_embed = weather_embed.expand(batch_size, -1, self.num_nodes, -1)  # (B, L, N, D_w)
        
        # 空间扩展（为每个节点学习不同的天气影响）
        B, L, N, D = weather_embed.shape
        weather_embed = weather_embed.reshape(B * L * N, D)
        weather_embed = self.weather_spatial_expand(weather_embed)
        weather_embed = weather_embed.reshape(B, L, N, -1)
        
        return weather_embed
    
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: Batch对象，包含 'X' 和其他特征
        Returns:
            预测结果: (batch_size, output_window, num_nodes, output_dim)
        """
        # 获取输入
        x = batch['X'].to(self.device)  # (B, input_window, num_nodes, feature_dim)
        batch_size = x.shape[0]
        
        # 尝试从 batch 获取时间索引（如果 Dataset 提供了）
        time_idx_list = None
        if hasattr(batch, 'data'):
            time_idx_raw = batch.data.get('time_idx', None)
            if time_idx_raw is not None:
                # 转换为列表
                if isinstance(time_idx_raw, torch.Tensor):
                    time_idx_list = time_idx_raw.cpu().flatten().tolist()
                elif isinstance(time_idx_raw, np.ndarray):
                    time_idx_list = time_idx_raw.flatten().tolist()
                elif isinstance(time_idx_raw, list):
                    time_idx_list = time_idx_raw
        
        # 特征提取
        features = []
        
        # 主特征投影
        x_main = self.input_proj(x)  # (B, L, N, input_embedding_dim)
        features.append(x_main)
        
        # 时间嵌入
        if self.add_time_in_day and self.tod_indices is not None:
            tod_indices = self.tod_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            tod_emb = self.tod_embedding(tod_indices)
            features.append(tod_emb)
        
        if self.add_day_in_week and self.dow_indices is not None:
            dow_indices = self.dow_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.num_nodes)
            dow_emb = self.dow_embedding(dow_indices)
            features.append(dow_emb)
        
        # 空间嵌入
        spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.input_window, -1, -1)
        features.append(spatial_emb)
        
        # 自适应嵌入
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            features.append(adp_emb)
        
        # 拼接所有特征
        x = torch.cat(features, dim=-1)  # (B, L, N, model_dim)
        
        # 投影到 Mamba 维度
        x = self.mamba_input_proj(x)  # (B, L, N, d_model)
        
        # 获取天气嵌入 (B, L, N, weather_embed_dim)
        weather_embed = self._get_weather_embedding(batch_size, time_idx_list)
        
        # 时序处理：每个节点独立处理
        # 重塑为 (N, B*L, d_model) 以便高效处理
        x_temporal = x.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, self.d_model)
        
        # 天气嵌入也需要相应重塑
        w_temporal = weather_embed.permute(2, 0, 1, 3).reshape(self.num_nodes, batch_size * self.input_window, self.weather_embed_dim)
        
        # 通过时序块
        x_temporal = self.temporal_block(x_temporal, w_temporal)
        
        # 重塑回 (B, L, N, d_model)
        x_temporal = x_temporal.reshape(self.num_nodes, batch_size, self.input_window, self.d_model).permute(1, 2, 0, 3)
        
        # 空间处理
        x_spatial_input = x_temporal.reshape(batch_size * self.input_window, self.num_nodes, self.d_model)
        w_spatial = weather_embed.reshape(batch_size * self.input_window, self.num_nodes, self.weather_embed_dim)
        
        x_spatial = self.spatial_block(x_spatial_input, w_spatial)
        
        # 重塑回 (B, L, N, d_model)
        x_spatial = x_spatial.reshape(batch_size, self.input_window, self.num_nodes, self.d_model)
        
        # 组合时序和空间输出
        x_combined = x_temporal * self.combine_weights[0] + x_spatial * self.combine_weights[1]
        
        # 最终处理和输出投影
        x_out = self.final_layer_norm(x_combined)  # (B, input_window, N, d_model)
        x_out = self.output_proj(x_out)  # (B, input_window, N, output_dim)
        
        # 时序投影：将 input_window 映射到 output_window
        # 转置为 (B, N, output_dim, input_window) -> 投影 -> 转置回 (B, output_window, N, output_dim)
        x_out = x_out.permute(0, 2, 3, 1)  # (B, N, output_dim, input_window)
        x_out = self.temporal_proj(x_out)  # (B, N, output_dim, output_window)
        x_out = x_out.permute(0, 3, 1, 2)  # (B, output_window, N, output_dim)
        
        return x_out  # 直接预测未来 output_window 步
    
    def calculate_loss(self, batch):
        """计算训练损失"""
        y_true = batch['y']
        if hasattr(y_true, 'to'):
            y_true = y_true.to(self.device)
        y_predicted = self.predict(batch)
        
        # 反归一化
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # 计算 masked MAE 损失
        return loss.masked_mae_torch(y_predicted, y_true, 0)
    
    def predict(self, batch):
        """
        直接预测未来 output_window 步
        Args:
            batch: 包含 'X' 的 Batch 对象
        Returns:
            预测结果: (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)  # forward 直接输出未来预测
