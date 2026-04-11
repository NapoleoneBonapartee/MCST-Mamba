# MCSTWeather Model

MCSTWeather (Multi-Channel Spatio-Temporal Mamba with Weather) 是一个融合天气数据的交通预测模型，基于 MCSTMamba_optimized 改进而来。

## 主要特点

1. **天气数据融合**: 通过改造 Mamba 模块，将天气嵌入拼接到 x_proj 输入，使动态参数 B, C, Δ 同时依赖输入序列和天气条件
2. **天气衍生特征**: 自动构建温度、湿度、风速、气压、降水等多维度天气衍生特征
3. **缺失值处理**: 对天气数据进行前向填充、后向填充、线性插值等缺失值处理
4. **时空联合建模**: 分别使用时序和空间 MambaWeather 块处理交通数据

## 模型架构

```
Input Traffic Data (B, L, N, F)
    ↓
[Feature Embeddings] → [Mamba Input Projection]
    ↓
Weather Data (CSV) → [Weather Processor] → [Weather Features] → [Weather Embedding]
    ↓
Temporal MambaWeather Block (with weather_embed)
    ↓
Spatial MambaWeather Block (with weather_embed)
    ↓
[Output Projection] → Predictions (B, output_window, N, output_dim)
```

## 文件结构

```
libcity/model/traffic_speed_prediction/
├── MambaWeather.py      # 改造后的 Mamba 模块（支持天气嵌入）
├── MCSTWeather.py       # 主模型（包含天气数据处理器）
└── __init__.py          # 模型导出

libcity/config/model/traffic_state_pred/
└── MCSTWeather.json     # 默认配置文件
```

## 使用方法

### 基本使用

```bash
python run_model.py \
    --task traffic_state_pred \
    --model MCSTWeather \
    --dataset METR_LA \
    --exp_id weather_test_001
```

### 自定义天气文件路径

```bash
python run_model.py \
    --task traffic_state_pred \
    --model MCSTWeather \
    --dataset METR_LA \
    --exp_id weather_test_002 \
    --weather_file ./raw_data/METR_LA/weather.csv
```

### 禁用天气功能（退化为 MCSTMamba_optimized）

```bash
python run_model.py \
    --task traffic_state_pred \
    --model MCSTWeather \
    --dataset METR_LA \
    --exp_id no_weather_test \
    --use_weather false
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weather_embed_dim` | 64 | 天气嵌入维度 |
| `use_weather` | true | 是否使用天气数据 |
| `weather_file` | null | 天气文件路径（自动推断） |

## 天气数据格式

天气数据应为 CSV 文件，包含以下列（METR_LA 数据集示例）：

```csv
,TMAX,TMIN,AWBT,RHAV,RHMX,RHMN,WSF2,WSF5,AWND,ADPT,ASTP,ASLP,PRCP,WDF2,WT01,WT02,...
2012-03-01 00:00:00,16.7,11.1,100.0,65.0,80.0,44.0,8.0,10.7,3.3,10200.0,10078.0,10200.0,0.0,280.0,,,...
...
```

## 天气衍生特征

模型自动构建以下天气衍生特征：

### 温度特征
- `temp_range`: 日温差 (TMAX - TMIN)
- `temp_avg`: 平均温度
- `apparent_temp_diff`: 体感温度差 (AWBT - TMAX)

### 湿度特征
- `humidity_level`: 湿度等级（低/中/高）
- `humidity_change`: 湿度变化率
- `humidity_range`: 湿度范围

### 风速特征
- `wind_gust_ratio`: 阵风比率
- `wind_avg`: 平均风速
- `wind_speed`: 风速

### 气压特征
- `pressure_diff`: 站压与海平面气压差
- `pressure_change`: 气压变化率
- `pressure_level`: 气压等级

### 降水特征
- `has_precipitation`: 是否有降水
- `precipitation_level`: 降水强度等级

### 风向特征
- `wind_dir_sin/cos`: 风向正弦/余弦编码

### 天气现象特征
- `weather_phenomena_count`: 天气现象数量
- `has_severe_weather`: 是否有恶劣天气

### 时间特征
- `hour_sin/cos`: 小时正弦/余弦编码

## 注意事项

1. 天气文件默认自动查找路径：
   - `./raw_data/{dataset}/weather.csv`
   - `./raw_data/weather/{dataset}_weather.csv`
   - `./raw_data/weather/noaa_weather_5min.csv`

2. 如果天气文件不存在或加载失败，模型会自动禁用天气功能并输出警告。

3. 天气特征使用 Z-score 标准化。

4. MambaWeather 模块的关键改造：
   ```python
   # 原始 x_proj 输入：卷积输出 (B, L, d_inner)
   # 改造后 x_proj 输入：[卷积输出; 天气嵌入] (B, L, d_inner + weather_embed_dim)
   x_aug = torch.cat([x_transposed, weather_projected], dim=-1)
   x_dbl = self.x_proj(x_aug)  # 计算 dt, B, C
   ```
