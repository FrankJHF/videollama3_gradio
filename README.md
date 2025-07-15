## 系统要求

- **Python**: >= 3.10
- **CUDA**: >= 12.1 (支持 GPU 加速)
- **内存**: >= 16GB RAM
- **显存**: >= 8GB VRAM (推荐 16GB+)

## 快速安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd videollama3_gui
```

### 2. 使用 uv 安装依赖 (推荐)

```bash
# 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync

# 激活环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 3. 传统安装方式

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements-infer.txt
```

## 📁 项目结构

```
videollama3_gui/
├── app.py                    # 主应用程序
├── infer.py                  # 推理脚本
├── fix_weights.py            # 权重修复工具
├── config.yaml               # 配置文件
├── pyproject.toml            # uv 项目配置
├── examples/                 # 示例视频
│   ├── 装运过程火灾.mp4
│   ├── 设备漏油着火.mp4
│   └── ...
├── VideoLLaMA3/              # 核心框架
├── model_ck/                 # 主模型目录
├── model_c2h/                # 备用模型目录
└── assets/                   # 静态资源
```

## 🎮 使用方法

### 启动 Web 界面

```bash
python app.py
```

访问 `http://localhost:7860` 即可使用 Web 界面。

### 命令行推理

```bash
python infer.py
```

### 配置修改

编辑 `config.yaml` 文件调整模型和推理参数：

```yaml
model:
  path: model_ck              # 模型路径
  device: cuda                # 计算设备
  torch_dtype: bfloat16       # 数据类型
  attn_implementation: flash_attention_2

inference:
  fps: 2                      # 视频采样帧率
  max_frames: 160             # 最大处理帧数
  max_new_tokens: 180         # 最大生成token数
  timeout: 300                # 推理超时时间(秒)
```
