#!/bin/bash
echo "🔧 Instalando dependências..."
pip install "chandra-ocr[hf]" flask flask-cors pillow numpy opencv-python-headless --no-cache-dir
echo "🔧 Corrigindo PyTorch para cu124..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --no-deps --force-reinstall --no-cache-dir
echo "✅ Verificando..."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch:', torch.__version__)"
