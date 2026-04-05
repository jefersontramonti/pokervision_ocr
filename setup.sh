#!/bin/bash
# ============================================================
# Poker Vision OCR — Setup completo para RunPod
# Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel
# GPU: RTX A4500 (20GB VRAM) | Driver CUDA: 12.4
# ============================================================
set -e

echo "=========================================="
echo "  Poker Vision OCR — Setup"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Passo 1: Instalar chandra-ocr COM todas as dependências
# (vai puxar torch 2.11 incompatível — corrigimos no passo 2)
echo ""
echo "🔧 [1/3] Instalando chandra-ocr + dependências..."
pip install "chandra-ocr[hf]" flask flask-cors pillow numpy opencv-python-headless \
    --ignore-installed blinker --no-cache-dir 2>&1 | tail -5

# Passo 2: Forçar PyTorch compatível com o driver CUDA 12.4
# O chandra instala torch 2.11 (precisa CUDA 13.0) — sobrescrevemos com 2.5.1+cu124
echo ""
echo "🔧 [2/3] Corrigindo PyTorch → 2.5.1+cu124..."
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-deps --force-reinstall --no-cache-dir 2>&1 | tail -3

# Passo 3: Verificar tudo
echo ""
echo "🔧 [3/3] Verificando..."
python3 -c "
import torch
torch.backends.cudnn.enabled = False
print(f'  torch: {torch.__version__}')
print(f'  CUDA:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:   {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:  {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
from chandra.model import InferenceManager
print(f'  Chandra: OK')
"

echo ""
echo "=========================================="
echo "  ✅ Setup completo!"
echo "  Rode: cd /workspace/pokervision_ocr && python3 server.py"
echo "=========================================="
