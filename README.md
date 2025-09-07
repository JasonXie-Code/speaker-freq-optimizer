# 音箱频响优化工具 / Speaker Frequency Response Optimization Tool

[English](#english) | [中文](#chinese)

<div id="chinese">

## 项目简介

基于粉红噪声的专业音箱频响分析与EQ校正工具，提供实时频响分析、智能EQ调节建议、音频试听和5频段音箱问题诊断功能。

### 主要特性

- **智能频响分析**：支持WAV、MP3、FLAC等多种音频格式，最多可同时分析10个文件
- **双算法模式**：
  - 平滑优化算法：减少400Hz以上频段共鸣，听感更自然（推荐）
  - 精确校正算法：严格按测量数据校正，适合专业监听
- **精确EQ调节**：15段和31段专业EQ，基于Web Audio API标准双二阶峰值滤波器
- **实时预览**：拖拽调节EQ参数，图表同步更新预测效果
- **音频试听**：上传测试音频，实时体验EQ效果，支持无缝切换
- **5频段诊断**：超低频/低频/中频/高频/超高频专项问题诊断
- **智能音量匹配**：基于ISO 226等响度曲线的智能音量补偿
- **数据导出**：CSV格式导出EQ设置，可直接导入其他音频设备

### 技术特点

- **标准化算法**：使用Web Audio API标准实现，与专业音频设备兼容
- **频率下潜优化**：精确计算-6dB频率极限，评估音箱延伸性能
- **人耳感知模型**：基于ISO 226等响度曲线，确保不同EQ设置下音量一致
- **无损音频处理**：-12dB预衰减防止削波，保证音频质量

## 安装指南

### 环境要求

- Python 3.8+
- 现代浏览器（支持Web Audio API）
- 8GB RAM（推荐）

### 快速安装

```bash
# 克隆项目
git clone https://github.com/JasonXie-Code/speaker-freq-optimizer.git
cd speaker-freq-optimizer

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

访问 http://localhost:8088 开始使用

### Docker 部署（可选）

```bash
# 构建镜像
docker build -t speaker-freq-optimizer .

# 运行容器
docker run -p 8088:8088 speaker-freq-optimizer
```

## 使用指南

### 1. 音频分析

1. 准备粉红噪声录音文件（WAV/MP3/FLAC格式）
2. 上传1-10个音频文件，系统会自动平均处理
3. 选择算法模式（推荐使用平滑优化算法）
4. 点击"开始分析"

### 2. EQ调节

- **实时调节**：拖拽滑块调节各频段增益（±12dB）
- **精确输入**：双击数值标签直接输入精确值
- **快速操作**：重置、归零、导出功能
- **分页浏览**：15段EQ显示8个频段/页，31段EQ显示8个频段/页

### 3. 目标频率范围

设定EQ优化的频率范围，超出范围的频段将不做处理：
- 低频下潜：20-1000Hz（建议根据音箱规格设置）
- 高频延伸：1000-20000Hz（建议根据音箱规格设置）

### 4. 音频试听

1. 上传测试音频文件
2. 选择EQ模式：无EQ、15段EQ、31段EQ
3. 实时听音对比，无缝切换
4. 音量自动匹配，确保公平对比

### 5. 数据导出

- 单独导出15段或31段EQ设置
- 导出全部设置到CSV文件
- 可直接导入支持标准EQ格式的设备

## 技术架构

### 后端技术栈

- **Python Flask**：Web服务框架
- **librosa**：音频信号处理
- **scipy**：科学计算和滤波器设计
- **numpy**：数值计算

### 前端技术栈

- **Web Audio API**：实时音频处理
- **SVG**：高质量频响图表渲染
- **响应式设计**：支持桌面和移动设备

### 核心算法

- **频谱分析**：Welch方法功率谱密度估计
- **EQ滤波器**：标准双二阶峰值滤波器（IIR）
- **音量补偿**：基于ISO 226等响度曲线
- **平滑算法**：自适应频谱平滑，减少中高频共鸣

## API文档

### POST /api/analyze
音频分析接口

**请求参数：**
- `audio_files`: 音频文件列表
- `enable_smoothing`: 是否启用平滑算法（可选，默认true）

**响应格式：**
```json
{
  "success": true,
  "eq_suggestions": {
    "eq_15": [...],
    "eq_31": [...]
  },
  "chart_data": {...},
  "frequency_limits": {...},
  "speaker_issues": [...]
}
```

### POST /api/reoptimize
EQ重新优化接口

**请求参数：**
```json
{
  "frequencies": [...],
  "measured_spectrum_diff": [...],
  "target_low_freq": 20,
  "target_high_freq": 20000,
  "enable_smoothing": true
}
```

## 贡献指南

欢迎提交问题和改进建议！

### 开发环境设置

```bash
# 克隆开发分支
git clone -b develop https://github.com/JasonXie-Code/speaker-freq-optimizer.git

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black . && flake8 .
```

### 提交规范

- 功能：`feat: 添加新功能描述`
- 修复：`fix: 修复问题描述`
- 文档：`docs: 更新文档`
- 优化：`perf: 性能优化`

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 问题反馈

- **Issues**: [GitHub Issues](https://github.com/JasonXie-Code/speaker-freq-optimizer/issues)
- **讨论**: [GitHub Discussions](https://github.com/JasonXie-Code/speaker-freq-optimizer/discussions)

---

</div>

<div id="english">

## About

Professional speaker frequency response analysis and EQ correction tool based on pink noise, featuring real-time frequency analysis, intelligent EQ adjustment suggestions, audio testing, and 5-band speaker diagnostics.

### Key Features

- **Smart Frequency Analysis**: Supports WAV, MP3, FLAC formats, analyzes up to 10 files simultaneously
- **Dual Algorithm Modes**:
  - Smooth Optimization: Reduces resonance above 400Hz for natural sound (recommended)
  - Precise Correction: Strict data-based correction for professional monitoring
- **Precise EQ Control**: 15-band and 31-band professional EQ using Web Audio API standard biquad peaking filters
- **Real-time Preview**: Drag to adjust EQ parameters with synchronized chart updates
- **Audio Testing**: Upload test audio for real-time EQ effect experience with seamless switching
- **5-Band Diagnosis**: Specialized diagnostics for sub-bass/bass/midrange/treble/air frequencies
- **Smart Volume Matching**: Intelligent volume compensation based on ISO 226 equal loudness curves
- **Data Export**: CSV format EQ settings export, compatible with professional audio devices

### Technical Highlights

- **Standardized Algorithms**: Web Audio API standard implementation, compatible with professional audio equipment
- **Frequency Extension Optimization**: Precise -6dB frequency limit calculation for speaker extension performance evaluation
- **Human Auditory Model**: ISO 226 equal loudness curve based, ensuring consistent volume across different EQ settings
- **Lossless Audio Processing**: -12dB pre-attenuation prevents clipping while maintaining audio quality

## Installation

### Requirements

- Python 3.8+
- Modern browser with Web Audio API support
- 8GB RAM (recommended)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/JasonXie-Code/speaker-freq-optimizer.git
cd speaker-freq-optimizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start application
python app.py
```

Visit http://localhost:8088 to start using

### Docker Deployment (Optional)

```bash
# Build image
docker build -t speaker-freq-optimizer .

# Run container
docker run -p 8088:8088 speaker-freq-optimizer
```

## Usage Guide

### 1. Audio Analysis

1. Prepare pink noise recording files (WAV/MP3/FLAC format)
2. Upload 1-10 audio files, system automatically averages them
3. Select algorithm mode (smooth optimization recommended)
4. Click "Start Analysis"

### 2. EQ Adjustment

- **Real-time Control**: Drag sliders to adjust frequency band gains (±12dB)
- **Precise Input**: Double-click value labels for exact numerical input
- **Quick Actions**: Reset, zero, and export functions
- **Pagination**: 15-band EQ shows 8 bands/page, 31-band EQ shows 8 bands/page

### 3. Target Frequency Range

Set frequency range for EQ optimization, bands outside range remain unprocessed:
- Low frequency extension: 20-1000Hz (set according to speaker specs)
- High frequency extension: 1000-20000Hz (set according to speaker specs)

### 4. Audio Testing

1. Upload test audio file
2. Select EQ mode: No EQ, 15-band EQ, 31-band EQ
3. Real-time listening comparison with seamless switching
4. Automatic volume matching ensures fair comparison

### 5. Data Export

- Export 15-band or 31-band EQ settings individually
- Export all settings to CSV file
- Direct import to devices supporting standard EQ formats

## Technical Architecture

### Backend Stack

- **Python Flask**: Web service framework
- **librosa**: Audio signal processing
- **scipy**: Scientific computing and filter design
- **numpy**: Numerical computation

### Frontend Stack

- **Web Audio API**: Real-time audio processing
- **SVG**: High-quality frequency response chart rendering
- **Responsive Design**: Desktop and mobile device support

### Core Algorithms

- **Spectrum Analysis**: Welch method power spectral density estimation
- **EQ Filtering**: Standard biquad peaking filters (IIR)
- **Volume Compensation**: ISO 226 equal loudness curve based
- **Smoothing Algorithm**: Adaptive spectrum smoothing to reduce mid-high frequency resonance

## API Documentation

### POST /api/analyze
Audio analysis endpoint

**Request Parameters:**
- `audio_files`: List of audio files
- `enable_smoothing`: Enable smoothing algorithm (optional, default true)

**Response Format:**
```json
{
  "success": true,
  "eq_suggestions": {
    "eq_15": [...],
    "eq_31": [...]
  },
  "chart_data": {...},
  "frequency_limits": {...},
  "speaker_issues": [...]
}
```

### POST /api/reoptimize
EQ reoptimization endpoint

**Request Parameters:**
```json
{
  "frequencies": [...],
  "measured_spectrum_diff": [...],
  "target_low_freq": 20,
  "target_high_freq": 20000,
  "enable_smoothing": true
}
```

## Contributing

Issues and improvement suggestions are welcome!

### Development Environment Setup

```bash
# Clone development branch
git clone -b develop https://github.com/JasonXie-Code/speaker-freq-optimizer.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && flake8 .
```

### Commit Convention

- Feature: `feat: add new feature description`
- Fix: `fix: fix issue description`
- Docs: `docs: update documentation`
- Performance: `perf: performance optimization`

## License

MIT License - see [LICENSE](LICENSE) file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/JasonXie-Code/speaker-freq-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JasonXie-Code/speaker-freq-optimizer/discussions)

</div>

---

**⭐ If this project helps you, please consider giving it a star! ⭐**
