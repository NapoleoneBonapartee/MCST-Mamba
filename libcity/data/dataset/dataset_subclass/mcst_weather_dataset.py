"""
MCSTWeatherDataset: 为 MCSTWeather 模型提供时间索引的 Dataset
"""

import numpy as np
import pandas as pd
import sys
import os

# 直接导入 TrafficStatePointDataset，避免触发 dataset_subclass 的 __init__.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_state_point_dataset import TrafficStatePointDataset


class MCSTWeatherDataset(TrafficStatePointDataset):
    """
    为 MCSTWeather 模型提供时间索引信息的 Dataset
    在 X 和 y 之外，还返回时间索引信息（day_of_year 和 time_idx）
    """

    def __init__(self, config):
        super().__init__(config)
        # 修改缓存文件名，避免与普通 PointDataset 冲突
        self.cache_file_name = self.cache_file_name.replace('point_based_', 'mcst_weather_')
        
        # 获取 steps_per_day（从 config 或默认值）
        self.steps_per_day = config.get('steps_per_day', 288)
        
        # 添加时间索引到 feature_name
        self.feature_name = {
            'X': 'float', 
            'y': 'float',
            'time_idx': 'int'  # 一天中的时间索引 (0 到 steps_per_day-1)
        }
        
        # 加载原始数据的时间信息
        self._load_time_info()
    
    def _load_time_info(self):
        """加载原始数据的时间信息"""
        # 从第一个数据文件加载时间戳
        if isinstance(self.data_files, list):
            data_file = self.data_files[0]
        else:
            data_file = self.data_files
        
        dyna_file = self.data_path + data_file + '.dyna'
        df = pd.read_csv(dyna_file)
        
        # 解析时间戳
        df['time'] = pd.to_datetime(df['time'])
        
        # 获取唯一的时间戳（因为 dyna 文件每个时间有多个节点）
        self.timestamps = df['time'].unique()
        self.timestamps = pd.to_datetime(self.timestamps)
        self.timestamps = sorted(self.timestamps)
        
        self._logger.info(f"Loaded {len(self.timestamps)} timestamps from {data_file}")
    
    def _generate_input_data(self, df):
        """
        生成输入数据，同时生成时间索引
        
        Returns:
            tuple: (x, y, time_indices)
                x: (num_samples, input_window, ..., feature_dim)
                y: (num_samples, output_window, ..., feature_dim)  
                time_indices: (num_samples,) - 每个样本对应的时间索引
        """
        num_samples_total = len(self.timestamps)
        
        # 调用父类方法生成 x 和 y
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y, time_indices = [], [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples_total - abs(max(y_offsets)))
        
        for t in range(min_t, max_t):
            # 获取输入时间窗口的最后一个时间戳
            t_end = t  # x_offsets 的最后一个元素是 0，所以 t_end = t + 0 = t
            timestamp = self.timestamps[t_end]
            
            # 计算 time_idx（一天中的时间索引）
            minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
            time_idx = int(minutes_since_midnight / (24 * 60 / self.steps_per_day))
            time_idx = min(time_idx, self.steps_per_day - 1)
            
            # 获取数据
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            
            x.append(x_t)
            y.append(y_t)
            time_indices.append(time_idx)
        
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        time_indices = np.array(time_indices)
        
        return x, y, time_indices
    
    def _generate_data(self):
        """
        加载数据文件并生成时间索引
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        
        # 加载外部数据
        if self.load_external and hasattr(self, 'ext_file') and \
           self.data_path and self.ext_file and self._file_exists(self.data_path + self.ext_file + '.ext'):
            ext_data = self._load_ext()
        else:
            ext_data = None
        
        x_list, y_list, time_indices_list = [], [], []
        
        for filename in data_files:
            df = self._load_dyna(filename)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            
            # 调用修改后的 _generate_input_data
            x, y, time_indices = self._generate_input_data(df)
            
            x_list.append(x)
            y_list.append(y)
            time_indices_list.append(time_indices)
        
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        time_indices = np.concatenate(time_indices_list)
        
        self._logger.info("Dataset created with time indices")
        self._logger.info(f"x shape: {x.shape}, y shape: {y.shape}, time_indices shape: {time_indices.shape}")
        
        # 保存时间索引供后续使用
        self.time_indices = time_indices
        
        return self._split_train_val_test_with_time(x, y, time_indices)
    
    def _file_exists(self, filepath):
        """检查文件是否存在"""
        import os
        return os.path.exists(filepath)
    
    def _split_train_val_test_with_time(self, x, y, time_indices):
        """
        划分训练集、测试集、验证集，同时划分时间索引
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # 划分数据
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        
        # 划分时间索引
        time_train = time_indices[:num_train]
        time_val = time_indices[num_train: num_train + num_val]
        time_test = time_indices[-num_test:]
        
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape) + ", time: " + str(time_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape) + ", time: " + str(time_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape) + ", time: " + str(time_test.shape))

        if self.cache_dataset:
            from libcity.utils import ensure_dir
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
                time_train=time_train,
                time_val=time_val,
                time_test=time_test,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        
        return x_train, y_train, x_val, y_val, x_test, y_test, time_train, time_val, time_test
    
    def _load_cache_train_val_test(self):
        """加载缓存数据，包括时间索引"""
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        
        # 加载时间索引（如果存在）
        if 'time_train' in cat_data:
            time_train = cat_data['time_train']
            time_val = cat_data['time_val']
            time_test = cat_data['time_test']
        else:
            # 兼容旧缓存，生成默认时间索引
            time_train = np.zeros(len(x_train), dtype=np.int64)
            time_val = np.zeros(len(x_val), dtype=np.int64)
            time_test = np.zeros(len(x_test), dtype=np.int64)
        
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        
        # 保存时间索引
        self.time_train = time_train
        self.time_val = time_val
        self.time_test = time_test
        
        return x_train, y_train, x_val, y_val, x_test, y_test, time_train, time_val, time_test
    
    def get_data(self):
        """
        返回数据的 DataLoader，包含时间索引
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        time_train, time_val, time_test = [], [], []
        
        if self.data is None:
            self.data = {}
            if self.cache_dataset and self._file_exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test, time_train, time_val, time_test = \
                    self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test, time_train, time_val, time_test = \
                    self._generate_data()
        
        # 在测试集上添加随机扰动
        if self.robustness_test:
            x_test = self._add_noise(x_test)
        
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        
        # 把训练集的 X, y, time_idx 聚合在一起成为 list
        # 注意：每个元素是 (x, y, time_idx) 的元组
        train_data = list(zip(x_train, y_train, time_train))
        eval_data = list(zip(x_val, y_val, time_val))
        test_data = list(zip(x_test, y_test, time_test))
        
        # 转 Dataloader
        # 注意：禁用 pad_with_last_sample，因为数据包含 time_idx 三元组，无法用 numpy 处理 padding
        from libcity.data.utils_optimized import generate_dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=False)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
