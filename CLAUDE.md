# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fire emergency detection system built on VideoLLaMA3, providing a Gradio web interface for analyzing video content to detect fire-related emergencies. The system specializes in Chinese fire emergency response scenarios and outputs structured JSON analysis.

## Commands

### Installation
```bash
# Recommended: Install with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# Alternative: Traditional pip install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-infer.txt
```

### Running the Application
```bash
# Start web interface (main application)
python app.py

# Command-line inference
python infer.py

# Alternative versions
python app_hf_template.py      # HuggingFace template version
python app_wo_visualization.py # No visualization version
```

### Model Management
```bash
# Fix model weights if needed
python fix_weights.py
```

## Architecture

### Core Components

- **app.py**: Main Gradio application with visualization and timeline features
- **infer.py**: Command-line inference script for batch processing
- **config.yaml**: Central configuration for model and inference parameters
- **VideoLLaMA3/**: Core multimodal language model framework
- **model_ck/**, **model_c2h/**, **model_hf/**: Different model checkpoint directories

### Key Modules

- **VideoLLaMA3.videollama3**: Core model initialization and inference
  - `model_init()`: Initialize model with custom parameters
  - `mm_infer()`: Multimodal inference for video/image analysis
  - `disable_torch_init()`: Optimization for faster model loading

- **VideoLLaMA3.videollama3.mm_utils**: Multimedia utilities
  - `load_video()`: Video processing with frame sampling
  - `load_images()`: Image loading and preprocessing

### Model Architecture

The system uses a multimodal architecture combining:
- Vision encoder for video/image processing
- Language model (Qwen2-based) for text generation
- Vision projector bridging visual and language representations
- Flash attention for efficient processing

### Configuration System

Configuration is managed through `config.yaml`:
- Model settings: path, device, torch_dtype, attention implementation
- Inference parameters: fps, max_frames, max_new_tokens, timeout
- All parameters can be overridden programmatically

### Emergency Detection Pipeline

1. Video preprocessing: Frame extraction at specified fps
2. Visual encoding: Convert frames to visual tokens
3. Prompt processing: Structured Chinese prompts for emergency detection
4. Inference: Generate JSON-structured emergency analysis
5. Post-processing: Timeline visualization and event categorization

## Development Notes

- The system is optimized for CUDA ≥ 12.1 and requires ≥ 8GB VRAM
- Uses bfloat16 precision and flash attention for efficiency
- Hardcoded for Chinese fire emergency scenarios
- Model paths are configurable but default to `model_ck/`
- Gradio interface runs on port 7860 by default
- Video processing supports up to 160 frames by default
- Output format is strictly JSON with emergency detection fields

## File Structure

- Root-level Python files are application entry points
- `VideoLLaMA3/` contains the core ML framework
- `examples/` contains test video files (Chinese fire scenarios)
- `model_*/` directories contain different model checkpoints
- `assets/` contains static resources for the interface