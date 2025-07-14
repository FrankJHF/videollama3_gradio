# 第一步：准备环境
## 选择一：实用现有的环境(这个环境是训练用的，推理只需要子集)
conda activate videollama3 


## 选择二：从头开始构建环境（更小一点）

Basic Dependencies:

* Python >= 3.10
* Pytorch >= 2.4.0
* CUDA Version >= 11.8
* transformers >= 4.46.3

Install required packages:

**[Inference-only]**

```bash
conda create -n videollama3_infer python=3.10
conda activate videollama3_infer

pip install torch==2.4.0 torchvision==0.17.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation
pip install transformers==4.46.3 accelerate==1.0.1
pip install decord ffmpeg-python imageio opencv-python
```


# 第二步 开始推理
直接运行/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3/videollama3/infer.py
PS：最顶部有一个os.chdir('/home/zhouting/SurveillanceVideoFireAnalysis/VideoLLaMA3')，若切换设备改成对应目录即可

截止2025.7.9号最好的模型（模型大小4.2G）
model_path = '/data/zhouting/outputs/fire_analysis/models/videollama3_2b_local_new_2stage/stageII_batch32_180f_3e_directly_2b'