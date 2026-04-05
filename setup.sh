#!/bin/bash
# ============================================================
# Setup para RunPod cu124 (RTX A4500 / 3090 / 4090)
# ============================================================

echo "🔧 Instalando dependências Python..."
pip install --ignore-installed "chandra-ocr[hf]" flask flask-cors pillow numpy opencv-python-headless --no-cache-dir

echo "🔧 Fixando PyTorch para cu124 (sweet spot: 2.5.1)..."
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-deps --force-reinstall --no-cache-dir

echo "✅ Verificando CUDA..."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch:', torch.__version__)"
