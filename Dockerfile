# ============================================================
# Poker Vision OCR Server — RunPod Docker Image
# Base: CUDA 12.1 + Python 3.11 (compatível com RTX 3090)
# ============================================================

FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Dependências do sistema (opencv headless precisa de libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
