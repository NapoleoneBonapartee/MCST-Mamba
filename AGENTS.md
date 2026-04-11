# MCST-Mamba / LibCity 项目指南

> 本文档为 AI 编程助手提供项目背景、架构说明和开发指南。
> 
> This document provides project background, architecture description, and development guidelines for AI coding agents.

---

## 1. 项目概述 (Project Overview)

MCST-Mamba 是基于 [LibCity](https://github.com/LibCity/Bigscity-LibCity) 框架开发的交通预测深度学习项目。该项目专注于使用 Mamba 状态空间模型（State Space Models）进行多通道时空交通预测。

### 1.1 核心功能 (Core Features)

- **交通状态预测 (Traffic State Prediction)**: 预测交通流量、速度、需求等时空序列数据
- **轨迹位置预测 (Trajectory Location Prediction)**: 预测下一个位置/POI
- **到达时间估计 (ETA)**: 估计行程时间
- **地图匹配 (Map Matching)**: GPS 轨迹与道路网络匹配
- **道路表示学习 (Road Representation)**: 学习道路网络嵌入

### 1.2 技术栈 (Technology Stack)

- **Python**: 3.9
- **深度学习框架**: PyTorch 2.0.0 + CUDA 11.8
- **核心依赖**:
  - `mamba-ssm==1.2.2` - Mamba 状态空间模型实现
  - `dgl==2.1.0` - 深度图神经网络库
  - `torch-geometric==2.6.1` - PyTorch 几何深度学习扩展
  - `ray==2.44.1` - 分布式训练与超参数调优
  - `hyperopt==0.2.7` - 贝叶斯优化
  - `tensorboard==2.19.0` - 训练可视化
  - `transformers==4.51.0` - HuggingFace Transformers

---

## 2. 项目结构 (Project Structure)

```
MCST-Mamba/
├── libcity/                    # 核心库代码
│   ├── config/                 # 配置文件
│   │   ├── config_parser.py    # 配置解析器
│   │   ├── task_config.json    # 任务-模型映射注册表
│   │   ├── model/              # 模型配置文件 (JSON)
│   │   ├── data/               # 数据集配置文件
│   │   ├── executor/           # 执行器配置文件
│   │   └── evaluator/          # 评估器配置文件
│   ├── data/                   # 数据加载与处理
│   │   ├── dataset/            # 数据集类定义
│   │   │   ├── dataset_subclass/  # 特定模型数据集
│   │   │   ├── traffic_state_datatset.py
│   │   │   └── ...
│   │   ├── utils_optimized.py  # 优化的数据加载工具
│   │   ├── batch.py            # Batch 数据结构
│   │   └── list_dataset.py     # ListDataset 实现
│   ├── model/                  # 模型实现
│   │   ├── traffic_speed_prediction/   # 交通速度预测模型
│   │   │   ├── MCSTMamba.py            # 主模型
│   │   │   ├── MCSTMamba_optimized.py  # 优化版本
│   │   │   └── ...
│   │   ├── traffic_flow_prediction/    # 交通流量预测模型
│   │   ├── traffic_demand_prediction/  # 交通需求预测模型
│   │   ├── trajectory_loc_prediction/  # 轨迹位置预测模型
│   │   ├── eta/                        # ETA 模型
│   │   ├── map_matching/               # 地图匹配模型
│   │   ├── road_representation/        # 道路表示学习模型
│   │   ├── abstract_model.py           # 模型抽象基类
│   │   └── loss.py                     # 损失函数定义
│   ├── executor/               # 训练执行器
│   │   ├── traffic_state_executor.py
│   │   ├── traffic_state_executor_optimized.py
│   │   ├── abstract_executor.py
│   │   └── ...
│   ├── evaluator/              # 评估器
│   │   ├── traffic_state_evaluator.py
│   │   └── ...
│   ├── pipeline/               # 主流程管道
│   │   └── pipeline.py         # run_model, hyper_parameter 等
│   └── utils/                  # 工具函数
│       ├── utils.py            # 核心工具 (get_model, get_executor)
│       ├── argument_list.py    # 参数解析工具
│       └── ...
├── raw_data/                   # 原始数据集目录
│   ├── METR_LA/               # METR-LA 数据集示例
│   └── PEMSD4/                # PEMSD4 数据集示例
├── test/                       # 传统模型测试
├── run_model.py                # 单模型训练脚本 [主要入口]
├── run_hyper.py                # Hyperopt 超参数调优
├── hyper_tune.py               # Ray Tune 超参数调优
├── evaluate_trained_model.py   # 模型评估脚本
├── unit_test.py                # 单元测试
├── requirements.txt            # Conda 环境依赖
├── requirements_pip2.txt       # Pip 依赖列表
├── MODEL_FRAMEWORK_ANALYSIS.md # 框架分析文档
└── OPTIMIZATION_GUIDE.md       # 优化指南文档
```

---

## 3. 架构设计 (Architecture Design)

### 3.1 配置驱动架构 (Configuration-Driven)

项目采用"配置驱动 + 动态加载"设计模式：

```
命令行参数 > config_file.json > 默认配置
```

配置加载优先级（从高到低）：
1. 命令行参数 (`--learning_rate 0.001`)
2. 用户配置文件 (`--config_file my_config`)
3. 模型默认配置 (`libcity/config/model/{task}/{model}.json`)
4. 数据集配置 (`libcity/config/data/{dataset_class}.json`)
5. 执行器配置 (`libcity/config/executor/{executor}.json`)
6. 评估器配置 (`libcity/config/evaluator/{evaluator}.json`)
7. 数据集元信息 (`raw_data/{dataset}/config.json`)

### 3.2 核心组件 (Core Components)

```python
# 典型执行流程
ConfigParser -> get_dataset -> get_model -> get_executor -> train/evaluate
```

| 组件 | 职责 | 关键文件 |
|------|------|----------|
| ConfigParser | 配置解析与合并 | `libcity/config/config_parser.py` |
| Dataset | 数据加载与预处理 | `libcity/data/dataset/*.py` |
| Model | 模型前向/损失计算 | `libcity/model/*/*.py` |
| Executor | 训练/评估循环 | `libcity/executor/*.py` |
| Evaluator | 指标计算 | `libcity/evaluator/*.py` |

### 3.3 动态加载机制 (Dynamic Loading)

模型和执行器通过 `importlib` 动态加载：

```python
# libcity/utils/utils.py
def get_model(config, data_feature):
    if config['task'] == 'traffic_state_pred':
        return getattr(importlib.import_module('libcity.model.traffic_speed_prediction'),
                       config['model'])(config, data_feature)
    # ... 其他任务

def get_executor(config, model, data_feature):
    return getattr(importlib.import_module('libcity.executor'),
                   config['executor'])(config, model, data_feature)
```

---

## 4. 常用命令 (Common Commands)

### 4.1 训练模型 (Training)

```bash
# 基本训练命令
python run_model.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --exp_id exp001

# 使用自定义配置
python run_model.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --config_file MCSTMamba_optimized

# 接续训练 (自动寻找最新 checkpoint)
python run_model.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --exp_id exp001 \
    --resume auto

# 接续训练 (指定 epoch)
python run_model.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --exp_id exp001 \
    --resume 50
```

### 4.2 超参数调优 (Hyperparameter Tuning)

```bash
# 使用 Hyperopt
python run_hyper.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --params_file hyper_example.txt \
    --hyper_algo grid_search \
    --max_evals 100

# 使用 Ray Tune
python hyper_tune.py \
    --task traffic_state_pred \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --space_file hyper_example \
    --search_alg HyperOpt \
    --num_samples 20
```

### 4.3 模型评估 (Evaluation)

```bash
python evaluate_trained_model.py \
    --model MCSTMamba \
    --dataset PEMSD4 \
    --model_dir MCSTMamba_PEMSD4_20250408_141451 \
    --epoch 95 \
    --evaluate_channels_separately
```

### 4.4 单元测试 (Unit Testing)

```bash
# 测试交通状态预测模型
python -c "from unit_test import test_new_tsp_model; test_new_tsp_model()"

# 测试轨迹位置预测模型
python -c "from unit_test import test_new_tlp_model; test_new_tlp_model()"
```

---

## 5. 代码规范 (Code Style Guidelines)

### 5.1 语言与注释 (Language & Comments)

- **代码**: 使用英文（类名、函数名、变量名）
- **注释**: 混合使用中英文，关键逻辑使用中文解释
- **文档字符串**: 中英文双语，优先中文

```python
def train(self, train_dataloader, eval_dataloader):
    """
    use data to train model with config
    使用数据训练模型
    """
    # 训练代码
```

### 5.2 命名规范 (Naming Conventions)

- **类名**: UpperCamelCase (e.g., `TrafficStateExecutor`, `MCSTMamba`)
- **函数/方法**: lower_case_with_underscores (e.g., `get_model`, `train_epoch`)
- **私有方法**: 单下划线前缀 (e.g., `_train_epoch`, `_build_optimizer`)
- **常量**: 全大写 (e.g., `DEFAULT_EPOCHS`)

### 5.3 代码格式 (Code Formatting)

- **行长度**: 最大 120 字符（见 `.flake8` 配置）
- **缩进**: 4 空格
- **空行**: 类之间 2 空行，方法之间 1 空行

### 5.4 类型提示 (Type Hints)

推荐使用类型提示增强代码可读性：

```python
def run_model(task: str, model_name: str, dataset_name: str, 
              config_file: str = None, saved_model: bool = True) -> None:
```

---

## 6. 开发指南 (Development Guide)

### 6.1 添加新模型 (Adding a New Model)

1. **创建模型文件**: `libcity/model/traffic_speed_prediction/MyModel.py`

2. **继承基类**:
```python
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

class MyModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # 初始化层
    
    def forward(self, batch):
        # 前向传播
        pass
    
    def predict(self, batch):
        # 预测方法（通常调用 forward）
        pass
    
    def calculate_loss(self, batch):
        # 计算损失
        pass
```

3. **添加配置文件**: `libcity/config/model/traffic_state_pred/MyModel.json`

4. **注册模型**: 在 `libcity/config/task_config.json` 中添加：
```json
{
    "traffic_state_pred": {
        "allowed_model": ["MyModel", ...],
        "MyModel": {
            "dataset_class": "TrafficStatePointDataset",
            "executor": "TrafficStateExecutor",
            "evaluator": "TrafficStateEvaluator"
        }
    }
}
```

5. **导出模型**: 在 `libcity/model/traffic_speed_prediction/__init__.py` 中添加导入。

### 6.2 添加新数据集 (Adding a New Dataset)

1. 准备数据文件放入 `raw_data/MyDataset/`
   - `MyDataset.dyna` - 动态数据（时间序列）
   - `MyDataset.geo` - 地理信息（节点/边）
   - `MyDataset.rel` - 关系数据（邻接矩阵）
   - `config.json` - 数据集元信息

2. （可选）创建自定义 Dataset 类继承 `TrafficStateDataset`

### 6.3 配置文件格式 (Config File Format)

```json
{
    "max_epoch": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "lr_scheduler": "multisteplr",
    "input_window": 12,
    "output_window": 12,
    "add_time_in_day": true,
    "add_day_in_week": false,
    "d_model": 128,
    "num_layers": 3,
    "dropout": 0.1
}
```

---

## 7. 测试策略 (Testing Strategy)

### 7.1 单元测试 (Unit Tests)

- **文件**: `unit_test.py`
- **功能**: 
  - `test_new_tsp_model()` - 测试交通状态预测模型 API
  - `test_new_tlp_model()` - 测试轨迹位置预测模型 API
  - `test_new_traj_encoder()` - 测试轨迹编码器

### 7.2 传统模型测试 (Traditional Model Tests)

- **目录**: `test/`
- **文件**: `test_HA.py`, `test_ARIMA.py`, `test_VAR.py`, `test_SVR.py`
- **用途**: 与深度学习模型做基线对比

### 7.3 回归测试 (Regression Testing)

```bash
# 快速测试模型是否能正常前向传播
python test_model.py
```

---

## 8. 关键文件说明 (Key File Reference)

| 文件 | 说明 |
|------|------|
| `run_model.py` | 主训练脚本，支持单模型训练和评估 |
| `libcity/pipeline/pipeline.py` | 核心流程控制，包含 `run_model()` 函数 |
| `libcity/config/task_config.json` | 任务-模型-数据集映射注册表 |
| `libcity/config/config_parser.py` | 配置解析器，处理配置优先级 |
| `libcity/utils/utils.py` | 核心工具函数（动态加载、日志等） |
| `libcity/executor/traffic_state_executor.py` | 默认训练执行器 |
| `libcity/model/traffic_speed_prediction/MCSTMamba.py` | 主模型实现 |
| `evaluate_trained_model.py` | 训练后模型评估与可视化 |

---

## 9. 注意事项 (Important Notes)

### 9.1 GPU 与显存 (GPU & Memory)

- 默认使用 GPU (`cuda:0`)，可通过 `--gpu_id` 指定
- 混合精度训练支持: `--fp16 true`
- 显存优化: 使用 `gradient_accumulation_steps` 模拟大 batch

### 9.2 Checkpoint 与恢复 (Checkpoint & Resume)

- Checkpoint 保存路径: `./libcity/cache/{exp_id}/model_cache/`
- 保存内容包括: 模型状态、优化器状态、学习率调度器、随机种子状态
- 支持从任意 epoch 接续训练

### 9.3 日志与可视化 (Logging & Visualization)

- 日志目录: `./libcity/log/`
- TensorBoard: `./libcity/cache/{exp_id}/`
- 评估可视化: `./libcity/cache/{exp_id}/visualization/`
- 训练总结: `./libcity/cache/{exp_id}/logs/training_summary.txt`

### 9.4 随机种子 (Random Seeds)

项目自动设置以下随机种子确保可复现性：
- Python `random`
- NumPy
- PyTorch (CPU & CUDA)
- `torch.backends.cudnn.deterministic = True`

---

## 10. 参考文档 (Reference Documents)

- `MODEL_FRAMEWORK_ANALYSIS.md` - 框架架构详细分析
- `OPTIMIZATION_GUIDE.md` - MCSTMamba 优化指南
- 配置示例: `hyper_example.json`, `hyper_example.txt`

---

*文档版本: 1.0 | 最后更新: 2026-04-11*
