# ============================================================
# Poker Vision OCR Server — RunPod Docker Image
# Base: CUDA 12.4 + Python 3.11 (compatível com RTX A4500)
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

# Copiar e executar setup (instala deps + corrige torch)
COPY setup.sh .
COPY requirements.txt .
RUN chmod +x setup.sh && ./setup.sh

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
