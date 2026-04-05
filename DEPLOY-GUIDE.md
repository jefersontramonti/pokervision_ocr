# Poker Vision OCR Server — Deploy no RunPod

## Arquivos do Projeto (GitHub)

Apenas esses arquivos vão pro repositório:

```
poker-vision-ocr/
├── runpod-ocr-server.py    ← Servidor Flask + Chandra OCR
├── requirements.txt        ← Dependências Python
├── Dockerfile              ← Imagem Docker para RunPod
├── start.sh                ← Script de inicialização
├── calibrate.html          ← Ferramenta de calibração (usa no browser local)
├── poker-vision.html       ← App principal (usa no browser local)
└── .gitignore
```

## Passo a Passo

### 1. Criar Repo no GitHub (PyCharm)

1. Abra PyCharm → **New Project** → escolha a pasta dos arquivos
2. **VCS** → **Enable Version Control** → Git
3. Terminal do PyCharm:
   ```bash
   git init
   git add runpod-ocr-server.py requirements.txt Dockerfile start.sh calibrate.html poker-vision.html .gitignore
   git commit -m "Initial commit: Poker Vision OCR Server"
   ```
4. Crie o repo no GitHub (github.com → New Repository → `poker-vision-ocr`)
5. Terminal:
   ```bash
   git remote add origin https://github.com/SEU_USER/poker-vision-ocr.git
   git push -u origin main
   ```

### 2. Criar GPU Pod no RunPod

1. Acesse [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**
2. Escolha a GPU: **RTX 3090** ($0.27/hr) ou **RTX 4090** ($0.39/hr)
3. Em **Template**, escolha: **RunPod Pytorch 2.2.0** (já tem CUDA + Python)
4. **Container Disk**: 20 GB (Chandra OCR pesa ~8GB)
5. **Volume Disk**: 10 GB (para cache do modelo)
6. **Expose HTTP Ports**: adicione `8080`
7. Clique **Deploy**

### 3. Configurar o Pod (SSH)

Quando o pod estiver rodando:

1. Clique **Connect** → **Start Web Terminal** (ou use SSH)
2. No terminal do pod:

```bash
# Clonar seu repo
cd /workspace
git clone https://github.com/SEU_USER/poker-vision-ocr.git
cd poker-vision-ocr

# Instalar dependências
pip install -r requirements.txt

# Testar se Chandra OCR carrega
python3 -c "from chandra_ocr import ocr; print('OK')"

# Iniciar o servidor
chmod +x start.sh
./start.sh
```

3. Aguarde a mensagem: `🚀 Servidor rodando em http://0.0.0.0:8080`

### 4. Pegar a URL do RunPod

1. No painel do RunPod, clique **Connect** no seu pod
2. Procure a seção **HTTP Service** — a URL será algo como:
   ```
   https://[POD_ID]-8080.proxy.runpod.net
   ```
3. Teste no navegador: `https://[POD_ID]-8080.proxy.runpod.net/health`
   - Deve retornar: `{"status": "ok", "ocr_loaded": true}`

### 5. Configurar no poker-vision.html

1. Abra `poker-vision.html` no navegador
2. No campo **OCR Server URL**, cole:
   ```
   https://[POD_ID]-8080.proxy.runpod.net
   ```
3. Clique **Salvar** — o indicador deve ficar verde

### 6. Manter o Servidor Rodando (Opcional)

Para que o servidor não pare quando você fecha o terminal:

```bash
# Opção A: nohup
nohup ./start.sh > server.log 2>&1 &

# Opção B: tmux (melhor)
tmux new -s poker
./start.sh
# Ctrl+B, depois D para desanexar
# tmux attach -t poker para voltar
```

## Atualizações Futuras

Quando alterar o código:

```bash
# No PyCharm: commit + push normalmente

# No terminal do RunPod:
cd /workspace/poker-vision-ocr
git pull
# Reiniciar o servidor (Ctrl+C se tiver rodando, depois:)
./start.sh
```

## Custos

| GPU | Preço/hr | VRAM | Velocidade estimada |
|-----|----------|------|-------------------|
| RTX 3090 | $0.27 | 24GB | ~2-4s por análise |
| RTX 4090 | $0.39 | 24GB | ~1-2s por análise |
| A5000 | $0.29 | 24GB | ~2-3s por análise |

Dica: Use **On-Demand** (não Spot) para não perder a instância. Pause o pod quando não estiver usando.
