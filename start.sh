#!/bin/bash
# ============================================================
# Poker Vision OCR Server — Script de inicialização
# ============================================================

echo "=========================================="
echo "  Poker Vision OCR Server"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Iniciar Flask server (modelo carrega lazy no primeiro request ou warmup)
echo "🚀 Iniciando servidor na porta 8080..."
exec python3 runpod-ocr-server.py
