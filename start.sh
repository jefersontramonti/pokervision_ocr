#!/bin/bash
# ============================================================
# Poker Vision OCR Server — Script de inicialização
# ============================================================

echo "=========================================="
echo "  Poker Vision OCR Server"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Pré-carregar modelo Chandra OCR no startup (evita timeout na primeira request)
echo "🔄 Pré-carregando Chandra OCR 2..."
python3 -c "
import torch
torch.backends.cudnn.enabled = False
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from PIL import Image

manager = InferenceManager(method='hf')
dummy = Image.new('RGB', (200, 50), (255, 255, 255))
results = manager.generate([BatchInputItem(image=dummy, prompt_type='ocr')])
print('✅ Chandra OCR 2 carregado com sucesso!')
" 2>&1

# Iniciar Flask server
echo "🚀 Iniciando servidor na porta 8080..."
cd /workspace/pokervision_ocr
exec python3 runpod-ocr-server.py
