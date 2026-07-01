# MuQ + VideoMAE V2 最小验证环境

本仓库用于复现 MuQ 与 VideoMAE V2 的最小可行性验证环境。当前验证基线：

- Python 3.8.20
- PyTorch 2.0.1 + CUDA 11.8
- torchvision 0.15.2
- torchaudio 2.0.2
- timm 0.4.12
- DeepSpeed 0.9.5
- VideoMAEv2 commit `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`
- 已验证硬件：NVIDIA GeForce RTX 4090，驱动 580.105.08

模型权重、数据集、conda 环境、wheel 缓存和 cookies 不在 Git 中。

## 目录结构

```text
weights/
├── imagebind/imagebind_huge.pth
├── muq/
│   ├── MuQ-large-msd-iter/
│   └── MuQ-MuLan-large/
└── videomae/hf_videomae_base/

outputs/
├── imagebind/baseline/
└── muq_videomae/
```

`outputs/imagebind/` 只保存 ImageBind 实验，`outputs/muq_videomae/` 保存
MuQ 音频特征与 VideoMAE 视频特征相关的检查、特征和检索结果。权重统一放在
`weights/`，实验脚本不再从项目根目录或隐藏 checkpoint 目录加载模型。

## 新服务器部署

前置条件：

- Linux x86_64
- NVIDIA 驱动支持 CUDA 11.8
- Git、Conda
- 建议至少预留 10GB 磁盘空间

克隆仓库及 VideoMAEv2 submodule：

```bash
git clone --recurse-submodules https://github.com/haovh18/vieo-audio.git
cd vieo-audio
```

如果克隆时未拉取 submodule：

```bash
git submodule update --init --recursive
```

在项目目录下创建环境：

```bash
PROJECT_DIR=$(pwd)
mkdir -p "$PROJECT_DIR/envs"

conda env create \
  -p "$PROJECT_DIR/envs/vmusic" \
  -f environment.yml

conda activate "$PROJECT_DIR/envs/vmusic"
```

若 `conda env create` 所用网络较慢，可分步安装：

```bash
conda create -p "$PROJECT_DIR/envs/vmusic" python=3.8 -y
conda activate "$PROJECT_DIR/envs/vmusic"

pip install torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install timm==0.4.12 deepspeed==0.9.5 "pydantic<2" \
  av==12.3.0 einops decord opencv-python librosa soundfile \
  nnAudio transformers tqdm pandas scikit-learn matplotlib scipy \
  pyyaml yacs tensorboardX termcolor submitit muq
```

DeepSpeed 必须保持 `0.9.5` 且 Pydantic 必须低于 2。较新的 DeepSpeed
版本在 Python 3.8 下导入失败。

## 验证

PyTorch、CUDA 和 MuQ：

```bash
python - <<'PY'
import sys
import torch
import torchvision
import torchaudio
import timm
from muq import MuQ

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)
print("timm:", timm.__version__)
print("MuQ import OK")
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

VideoMAE V2：

```bash
python - <<'PY'
import os
import sys
import torch

repo = os.path.abspath("code/VideoMAEv2")
sys.path[:0] = [os.path.join(repo, "models"), repo]

import modeling_finetune
import run_class_finetuning

print("Repo:", repo)
print("modeling_finetune import OK:", modeling_finetune.__file__)
print("run_class_finetuning import OK:", run_class_finetuning.__file__)
print("CUDA:", torch.cuda.is_available())
PY
```

当前版本的 `modeling_finetune.py` 位于 `code/VideoMAEv2/models/`，
因此验证脚本会同时加入仓库根目录和 `models/` 到 Python 搜索路径。

## Git 内容说明

以下内容仅保留在本机，不会推送：

- `envs/`：conda 环境
- `.wheels/`：下载缓存
- `datasets/`：数据集
- `weights/`：ImageBind、MuQ、MuQ-MuLan 和 VideoMAE 权重
- `outputs/`：按模型方案区分的本地实验产物
- `youtube-cookies.txt`：敏感凭据

`code/VideoMAEv2` 作为 Git submodule 管理，以固定上游代码版本并避免复制
第三方仓库历史。
