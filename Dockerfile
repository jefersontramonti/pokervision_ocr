# ============================================================
# Poker Vision OCR Server — RunPod Docker Image
# Base: CUDA 12.4 + Python 3.11
# ============================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Dependências do sistema (opencv headless precisa de libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python (sem torch — já vem na imagem base)
COPY requirements.txt .
RUN pip install --ignore-installed --no-cache-dir -r requirements.txt

# Fixar torch para versão compatível cu124
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-deps --force-reinstall --no-cache-dir

# Copiar código do servidor
COPY runpod-ocr-server.py .
COPY start.sh .
RUN chmod +x start.sh

# Porta do Flask
EXPOSE 8080

# Health check para RunPod
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Iniciar servidor
CMD ["./start.sh"]
