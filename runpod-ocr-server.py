"""
Poker Vision OCR Server — RunPod (GPU)
Abordagem: Chandra OCR puro + coordenadas fixas + lógica completa server-side.
O HTML é apenas um viewer: captura screenshot → envia → exibe resultado.

Endpoints:
  GET  /health          → status do servidor
  POST /analyze         → análise completa da mesa
  POST /update-coords   → atualizar coordenadas (calibração)
  GET  /get-coords      → retorna coordenadas ativas
  POST /ocr-test        → testar OCR numa região específica
  POST /warmup          → pré-carregar modelo OCR
  GET  /presets          → listar presets disponíveis
  POST /presets/<name>   → carregar preset específico
"""

import os
import re
import json
import time
import base64
import io

# ── Fix cuDNN: incompatível com torch 2.5.1 downgrade no RunPod ──
# Ainda usa GPU (CUDA) para compute, só desabilita cuDNN para conv ops
import torch
torch.backends.cudnn.enabled = False

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ============ OCR ENGINE (Chandra OCR 2) — BATCH MODE ============

_ocr_manager = None   # InferenceManager singleton
_BatchInputItem = None  # Classe para batch

def get_ocr_manager():
    """Lazy-load Chandra OCR 2 (4B params) — primeiro call demora ~30s"""
    global _ocr_manager, _BatchInputItem
    if _ocr_manager is None:
        print("🔄 Carregando Chandra OCR 2 (4B)... isso leva ~30s na primeira vez")
        t0 = time.time()
        try:
            from chandra.model import InferenceManager
            from chandra.model.schema import BatchInputItem
            _BatchInputItem = BatchInputItem
            _ocr_manager = InferenceManager(method="hf")

            # Warmup com imagem dummy
            dummy = Image.new('RGB', (100, 30), (255, 255, 255))
            _ocr_manager.generate([BatchInputItem(image=dummy, prompt_type="ocr")])
            print(f"✅ Chandra OCR 2 pronto em {time.time()-t0:.1f}s")
        except ImportError as e:
            print(f"⚠️  chandra-ocr não disponível ({e})")
            _ocr_manager = None
    return _ocr_manager


def ocr_batch(images):
    """
    OCR em batch: recebe lista de PIL Images, retorna lista de strings.
    Muito mais rápido que chamar individualmente (~3-5x speedup).
    """
    manager = get_ocr_manager()
    if not manager or not _BatchInputItem or not images:
        return [''] * len(images)

    batch_items = [_BatchInputItem(image=img, prompt_type="ocr") for img in images]
    try:
        results = manager.generate(batch_items)
        return [r.raw or r.markdown or "" for r in results]
    except Exception as e:
        print(f"  Batch OCR error: {e}")
        return [''] * len(images)


def ocr_single(img):
    """OCR de uma única imagem (para endpoints simples como /ocr-test)."""
    results = ocr_batch([img])
    return results[0] if results else ''


def crop_region(img_pil, region):
    """
    Recorta região da imagem. Retorna PIL Image.
    region: {x, y, w, h} como frações (0-1) ou pixels.
    """
    W, H = img_pil.size
    x, y, w, h = region['x'], region['y'], region['w'], region['h']

    # Se coordenadas são frações (0-1), converter para pixels
    if x <= 1 and y <= 1 and w <= 1 and h <= 1:
        x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
    else:
        x, y, w, h = int(x), int(y), int(w), int(h)

    # Garantir bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    cropped = img_pil.crop((x, y, x + w, y + h))

    # Upscale se muito pequeno (OCR precisa de pelo menos ~100px de largura)
    if w < 100 or h < 30:
        scale = max(3, 100 // max(w, 1))
        cropped = cropped.resize((w * scale, h * scale), Image.LANCZOS)

    return cropped


def ocr_crop(img_pil, region, mode='text'):
    """OCR de uma região (compat — para endpoints avulsos). Usa batch de 1."""
    cropped = crop_region(img_pil, region)
    text = ocr_single(cropped)

    if mode == 'number':
        return parse_number(text)
    elif mode == 'card':
        return parse_card_text(text)
    return text


def parse_number(text):
    """Parse texto OCR para número: '1,234' → 1234, '$40.58' → 40.58"""
    if not text:
        return 0
    clean = re.sub(r'[^0-9.,]', '', text)
    if not clean:
        return 0
    # Remover separadores de milhar
    if '.' in clean and ',' in clean:
        clean = clean.replace(',', '')
    elif ',' in clean:
        parts = clean.split(',')
        if len(parts[-1]) <= 2:
            clean = clean.replace(',', '.')
        else:
            clean = clean.replace(',', '')
    # Decimal (centavos/bounty) vs milhar
    if re.match(r'^\d+\.\d{1,2}$', clean):
        return float(clean)
    clean = clean.replace('.', '')
    try:
        return float(clean) if clean else 0
    except:
        return 0


def parse_card_text(text):
    """
    Parse OCR de uma carta. GGPoker mostra rank + suit gráfico.
    Chandra pode ler: 'A♠', 'K♥', 'Ah', '10s', 'Td', etc.
    Retorna formato padronizado: 'Ah', 'Ks', 'Td', etc. ou '' se inválido.
    """
    if not text:
        return ''
    text = text.strip()

    # Mapeamento de suits (unicode/texto → letra)
    suit_map = {
        '♠': 's', '♤': 's', 'spade': 's', 'spades': 's',
        '♥': 'h', '♡': 'h', 'heart': 'h', 'hearts': 'h',
        '♦': 'd', '♢': 'd', 'diamond': 'd', 'diamonds': 'd',
        '♣': 'c', '♧': 'c', 'club': 'c', 'clubs': 'c',
    }

    # Normalizar
    t = text.replace('10', 'T').replace(' ', '').upper()

    # Tentar extrair rank + suit
    rank_match = re.search(r'([AKQJT2-9])', t)
    if not rank_match:
        return ''
    rank = rank_match.group(1)

    # Procurar suit
    suit = ''
    for symbol, letter in suit_map.items():
        if symbol in text or symbol.upper() in text:
            suit = letter
            break
    if not suit:
        suit_letter = re.search(r'[SHDCshdc]', t[rank_match.end():])
        if suit_letter:
            suit = suit_letter.group(0).lower()

    if rank and suit:
        return rank + suit
    return ''


def detect_seat_active(img_pil, region):
    """Detecta se um seat tem cartas visíveis (brilho + variância)"""
    W, H = img_pil.size
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    if x <= 1 and y <= 1:
        x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
    else:
        x, y, w, h = int(x), int(y), int(w), int(h)

    cropped = img_pil.crop((x, y, x + w, y + h))
    arr = np.array(cropped)
    if arr.size == 0:
        return False

    gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr
    mean_bright = float(np.mean(gray))
    variance = float(np.var(gray.astype(float)))

    # Cartas face-up: mais brilho e variância que seat vazio
    if len(arr.shape) == 3:
        hsv_s = np.std(arr.astype(float), axis=2)
        colorful = float(np.mean(hsv_s))
    else:
        colorful = 0

    return (mean_bright > 70 and variance > 600) or (colorful > 30 and variance > 400)


def detect_dealer_position(img_pil, dealer_regions):
    """
    Detecta em qual seat está o dealer button "D".
    Detectar círculo amarelo por cor (mais confiável que OCR para button pequeno).
    """
    best_seat = -1
    best_score = 0

    for seat_idx, region in enumerate(dealer_regions):
        if not region:
            continue

        W, H = img_pil.size
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        if x <= 1 and y <= 1:
            x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H)
        else:
            x, y, w, h = int(x), int(y), int(w), int(h)

        cropped = img_pil.crop((x, y, x + w, y + h))
        arr = np.array(cropped)

        # Detectar círculo amarelo/dourado do dealer button
        if len(arr.shape) == 3:
            r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
            # Amarelo/dourado: R alto, G alto, B baixo
            yellow_mask = (r > 150) & (g > 130) & (b < 120)
            yellow_pct = np.sum(yellow_mask) / max(1, arr.shape[0] * arr.shape[1])
            # Branco (texto "D"): R,G,B todos altos
            white_mask = (r > 200) & (g > 200) & (b > 200)
            white_pct = np.sum(white_mask) / max(1, arr.shape[0] * arr.shape[1])

            score = yellow_pct * 70 + white_pct * 30
            if score > best_score:
                best_score = score
                best_seat = seat_idx

    return best_seat if best_score > 5 else -1


# ============ CONSTANTES ============

POS_MAPS = {
    2: ['BTN','BB'],
    3: ['BTN','SB','BB'],
    4: ['BTN','SB','BB','UTG'],
    5: ['BTN','SB','BB','UTG','CO'],
    6: ['BTN','SB','BB','UTG','HJ','CO'],
    7: ['BTN','SB','BB','UTG','MP','HJ','CO'],
    8: ['BTN','SB','BB','UTG','UTG+1','MP','HJ','CO'],
    9: ['BTN','SB','BB','UTG','UTG+1','LJ','MP','HJ','CO'],
}


def _valid_region(r):
    """Retorna True se a região tem dimensões válidas (não zerada)."""
    return r and isinstance(r, dict) and r.get('w', 0) > 0 and r.get('h', 0) > 0


# ============ COORDENADAS FIXAS — PRESETS ============

# GGPoker 8-max Bounty Hunters (janela 1260x898)
# Coordenadas em fração (0-1) do tamanho da imagem
# Mapeadas a partir do layout padrão GGPoker

PRESETS = {
    'ggpoker_8max_bounty': {
        'name': 'GGPoker 8-max Bounty Hunters',
        'seats': 8,

        # ── Info da mesa (barra superior com level, blinds, prize) ──
        'blinds': {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},

        # ── Pot (centro da mesa) ──
        'pot': {'x': 0.5136, 'y': 0.3370, 'w': 0.0753, 'h': 0.0441},

        # ── Board cards (5 posições — flop/turn/river) ── calibrado
        'board': [
            {'x': 0.2956, 'y': 0.3799, 'w': 0.0760, 'h': 0.1548},
            {'x': 0.3778, 'y': 0.3788, 'w': 0.0784, 'h': 0.1582},
            {'x': 0.4631, 'y': 0.3776, 'w': 0.0784, 'h': 0.1593},
            {'x': 0.5469, 'y': 0.3810, 'w': 0.0815, 'h': 0.1605},
            {'x': 0.6315, 'y': 0.3799, 'w': 0.0815, 'h': 0.1638},
        ],

        # ── Botões de ação do Hero (Fold / Call / Raise / Check) ──
        'hero_actions': {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},

        # ── Seats com ID por posição física na mesa ──
        # ID fixo baseado na posição visual (não muda com dealer)
        # Coordenadas calibradas via calibrate.html
        'seat_regions': [
            # Seat 0: HERO (bottom center)
            {
                'id': 'hero',
                'label': 'HERO',
                'position': 'bottom_center',
                'name':    {'x': 0.4306, 'y': 0.8703, 'w': 0.1427, 'h': 0.0362},
                'stack':   {'x': 0.4337, 'y': 0.9053, 'w': 0.1334, 'h': 0.0452},
                'cards':   {'x': 0.4313, 'y': 0.7302, 'w': 0.1466, 'h': 0.1051},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 1: ↙ (bottom-left)
            {
                'id': 'bottom_left',
                'label': 'Bottom Left',
                'position': 'bottom_left',
                'name':    {'x': 0.1497, 'y': 0.7325, 'w': 0.1296, 'h': 0.0316},
                'stack':   {'x': 0.1505, 'y': 0.7652, 'w': 0.1241, 'h': 0.0373},
                'cards':   {'x': 0.1552, 'y': 0.5935, 'w': 0.1140, 'h': 0.1051},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 2: ← (left)
            {
                'id': 'left',
                'label': 'Left',
                'position': 'left',
                'name':    {'x': 0.0147, 'y': 0.4624, 'w': 0.1358, 'h': 0.0362},
                'stack':   {'x': 0.0194, 'y': 0.5019, 'w': 0.1272, 'h': 0.0316},
                'cards':   {'x': 0.0171, 'y': 0.3370, 'w': 0.1296, 'h': 0.0972},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 3: ↖ (top-left)
            {
                'id': 'top_left',
                'label': 'Top Left',
                'position': 'top_left',
                'name':    {'x': 0.1715, 'y': 0.2206, 'w': 0.1319, 'h': 0.0373},
                'stack':   {'x': 0.1715, 'y': 0.2590, 'w': 0.1303, 'h': 0.0362},
                'cards':   {'x': 0.1792, 'y': 0.1019, 'w': 0.1125, 'h': 0.0847},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 4: ↑ (top center)
            {
                'id': 'top',
                'label': 'Top',
                'position': 'top_center',
                'name':    {'x': 0.4414, 'y': 0.1663, 'w': 0.1272, 'h': 0.0328},
                'stack':   {'x': 0.4422, 'y': 0.1980, 'w': 0.1234, 'h': 0.0452},
                'cards':   {'x': 0.4492, 'y': 0.0364, 'w': 0.1086, 'h': 0.0972},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 5: ↗ (top-right)
            {
                'id': 'top_right',
                'label': 'Top Right',
                'position': 'top_right',
                'name':    {'x': 0.7168, 'y': 0.2183, 'w': 0.1303, 'h': 0.0395},
                'stack':   {'x': 0.7192, 'y': 0.2579, 'w': 0.1249, 'h': 0.0384},
                'cards':   {'x': 0.7223, 'y': 0.0929, 'w': 0.1133, 'h': 0.0983},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 6: → (right)
            {
                'id': 'right',
                'label': 'Right',
                'position': 'right',
                'name':    {'x': 0.8596, 'y': 0.4624, 'w': 0.1272, 'h': 0.0373},
                'stack':   {'x': 0.8604, 'y': 0.4997, 'w': 0.1226, 'h': 0.0395},
                'cards':   {'x': 0.8697, 'y': 0.3404, 'w': 0.1071, 'h': 0.0915},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
            # Seat 7: ↘ (bottom-right)
            {
                'id': 'bottom_right',
                'label': 'Bottom Right',
                'position': 'bottom_right',
                'name':    {'x': 0.7308, 'y': 0.7291, 'w': 0.1257, 'h': 0.0373},
                'stack':   {'x': 0.7331, 'y': 0.7641, 'w': 0.1195, 'h': 0.0418},
                'cards':   {'x': 0.7386, 'y': 0.6025, 'w': 0.1109, 'h': 0.0972},
                'bet':     {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
                'dealer':  {'x': 0.6912, 'y': 0.6658, 'w': 0.0341, 'h': 0.0475},
                'bounty':  {'x': 0.00, 'y': 0.00, 'w': 0.00, 'h': 0.00},
            },
        ],
    }
}

# Coordenadas ativas (carregadas do preset ou calibração manual)
active_coords = None
coords_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coords.json')


def load_coords():
    """Carrega coordenadas salvas ou usa preset padrão"""
    global active_coords
    if os.path.exists(coords_file):
        with open(coords_file, 'r') as f:
            active_coords = json.load(f)
        print(f"📐 Coordenadas carregadas de {coords_file}")
    else:
        active_coords = PRESETS['ggpoker_8max_bounty']
        print("📐 Usando preset GGPoker 8-max Bounty")
    return active_coords


def save_coords(coords):
    """Salva coordenadas calibradas"""
    global active_coords
    active_coords = coords
    with open(coords_file, 'w') as f:
        json.dump(coords, f, indent=2)
    print(f"💾 Coordenadas salvas em {coords_file}")


# ============ ANÁLISE COMPLETA ============

def analyze_table(img_pil):
    """
    Análise completa da mesa — BATCH OCR.
    Coleta TODAS as regiões primeiro, faz UMA chamada batch ao Chandra,
    depois parseia os resultados. ~3-5x mais rápido que OCR sequencial.
    """
    coords = active_coords or load_coords()
    num_seats = int(coords.get('seats', 8))
    seat_regions = coords.get('seat_regions', [])
    t0 = time.time()
    timings = {}

    # ── Resultado: table (dados gerais) + seats (agrupados por ID) ──
    table = {
        'pot': 0, 'board': [], 'street': 'preflop',
        'blinds': '', 'btn_seat': -1,
        'num_active': 0, 'last_action': '', 'bet_to_call': 0,
        'tournament': True
    }

    # ════════════════════════════════════════════
    # FASE 1: Dealer detection (não precisa OCR)
    # ════════════════════════════════════════════
    t1 = time.time()
    dealer_regions = [s.get('dealer') for s in seat_regions]
    btn_seat_idx = detect_dealer_position(img_pil, dealer_regions)
    table['btn_seat'] = btn_seat_idx + 1 if btn_seat_idx >= 0 else -1
    timings['dealer'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  Dealer: seat {table['btn_seat']}")

    # ════════════════════════════════════════════
    # FASE 2: Detectar seats ativos (pixel check, sem OCR)
    # ════════════════════════════════════════════
    t1 = time.time()
    seat_active = []
    for i, sr in enumerate(seat_regions):
        is_active = True
        if sr.get('cards'):
            is_active = detect_seat_active(img_pil, sr['cards'])
        seat_active.append(is_active)
    timings['detect'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ════════════════════════════════════════════
    # FASE 3: Coletar TODAS as regiões para batch OCR
    # ════════════════════════════════════════════
    t1 = time.time()
    batch_images = []  # lista de PIL Images
    batch_keys = []    # lista de (tipo, índice) para mapear resultado

    # 3a. Blinds
    if _valid_region(coords.get('blinds')):
        batch_images.append(crop_region(img_pil, coords['blinds']))
        batch_keys.append(('blinds', 0))

    # 3b. Pot
    if _valid_region(coords.get('pot')):
        batch_images.append(crop_region(img_pil, coords['pot']))
        batch_keys.append(('pot', 0))

    # 3c. Board cards (só se região tem conteúdo visível)
    board_indices = []
    for i, card_region in enumerate(coords.get('board', [])):
        if _valid_region(card_region) and detect_seat_active(img_pil, card_region):
            batch_images.append(crop_region(img_pil, card_region))
            batch_keys.append(('board', i))
            board_indices.append(i)

    # 3d. Seats ativos: nome, stack, cards (2x), bet, bounty
    for i, sr in enumerate(seat_regions):
        if not seat_active[i]:
            continue

        if _valid_region(sr.get('name')):
            batch_images.append(crop_region(img_pil, sr['name']))
            batch_keys.append(('name', i))

        if _valid_region(sr.get('stack')):
            batch_images.append(crop_region(img_pil, sr['stack']))
            batch_keys.append(('stack', i))

        if _valid_region(sr.get('cards')):
            cr = sr['cards']
            half_w = cr['w'] / 2
            c1_r = {'x': cr['x'], 'y': cr['y'], 'w': half_w, 'h': cr['h']}
            c2_r = {'x': cr['x'] + half_w, 'y': cr['y'], 'w': half_w, 'h': cr['h']}
            batch_images.append(crop_region(img_pil, c1_r))
            batch_keys.append(('card1', i))
            batch_images.append(crop_region(img_pil, c2_r))
            batch_keys.append(('card2', i))

        if _valid_region(sr.get('bet')):
            batch_images.append(crop_region(img_pil, sr['bet']))
            batch_keys.append(('bet', i))

        if _valid_region(sr.get('bounty')):
            batch_images.append(crop_region(img_pil, sr['bounty']))
            batch_keys.append(('bounty', i))

    timings['crop'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  Batch: {len(batch_images)} regiões coletadas")

    # ════════════════════════════════════════════
    # FASE 4: OCR BATCH — UMA chamada para tudo
    # ════════════════════════════════════════════
    t1 = time.time()
    batch_results = ocr_batch(batch_images)
    timings['ocr_batch'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  OCR batch: {len(batch_results)} resultados em {(time.time()-t1)*1000:.0f}ms")

    # ════════════════════════════════════════════
    # FASE 5: Parsear resultados do batch
    # ════════════════════════════════════════════
    t1 = time.time()

    # Mapear resultados por chave
    result_map = {}
    for idx, (key, seat_idx) in enumerate(batch_keys):
        text = batch_results[idx].strip() if idx < len(batch_results) else ''
        result_map.setdefault(key, {})[seat_idx] = text

    # Blinds
    if 'blinds' in result_map:
        raw = result_map['blinds'].get(0, '')
        m = re.search(r'(\d[\d,.]*)\s*[|/]\s*(\d[\d,.]*)', raw)
        if m:
            table['blinds'] = f"{m.group(1)}/{m.group(2)}"
        elif raw:
            table['blinds'] = raw
        print(f"  Blinds: {table['blinds']}")

    # Pot
    if 'pot' in result_map:
        raw = result_map['pot'].get(0, '').replace(' ', '')
        m = re.search(r'(\d[\d,.]+)', raw)
        if m:
            table['pot'] = parse_number(m.group(1))
        print(f"  Pot: {table['pot']}")

    # Board
    for i in board_indices:
        raw = result_map.get('board', {}).get(i, '')
        card = parse_card_text(raw)
        if card:
            table['board'].append(card)
            print(f"  Board {i+1}: {card}")
    n_board = len(table['board'])
    table['street'] = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}.get(n_board, 'preflop')

    # Seats
    pos_map = POS_MAPS.get(num_seats, POS_MAPS[8])
    seats = []

    for i, sr in enumerate(seat_regions):
        seat_num = i + 1
        is_hero = (i == 0)
        # ID de posição física vem do preset/calibração
        phys_id = sr.get('id', f'seat_{seat_num}')
        phys_label = sr.get('label', f'Seat {seat_num}')
        seat = {
            'seat': seat_num,           # número sequencial (1-8)
            'id': phys_id,              # ID fixo de posição na mesa (hero, left, top_right...)
            'label': phys_label,        # label legível (HERO, Left, Top Right...)
            'name': '', 'stack': 0, 'cards': ['', ''],
            'position': '?',            # posição poker dinâmica (BTN, SB, BB, UTG...)
            'bet': 0, 'bounty': '',
            'is_hero': is_hero, 'active': seat_active[i], 'action': '',
        }

        if not seat_active[i]:
            seats.append(seat)
            continue

        # Nome
        raw_name = result_map.get('name', {}).get(i, '')
        if raw_name and len(raw_name) > 1:
            seat['name'] = raw_name
        else:
            seat['name'] = 'Hero' if is_hero else f'Player{seat_num}'
        if 'sitting' in seat['name'].lower() or 'sit out' in seat['name'].lower():
            seat['active'] = False
            seats.append(seat)
            continue
        print(f"  [{phys_id}] name: {seat['name']}")

        # Stack
        raw_stack = result_map.get('stack', {}).get(i, '')
        seat['stack'] = parse_number(raw_stack)
        print(f"  [{phys_id}] stack: {seat['stack']}")

        # Cards
        c1_raw = result_map.get('card1', {}).get(i, '')
        c2_raw = result_map.get('card2', {}).get(i, '')
        seat['cards'] = [parse_card_text(c1_raw), parse_card_text(c2_raw)]
        if seat['cards'][0] or seat['cards'][1]:
            print(f"  [{phys_id}] cards: {seat['cards']}")

        # Bet
        raw_bet = result_map.get('bet', {}).get(i, '')
        seat['bet'] = parse_number(raw_bet)
        if seat['bet'] > 0:
            print(f"  [{phys_id}] bet: {seat['bet']}")

        # Bounty
        seat['bounty'] = result_map.get('bounty', {}).get(i, '')

        # Posição poker
        if btn_seat_idx >= 0:
            seat['position'] = pos_map[(num_seats - btn_seat_idx + i) % num_seats]

        seats.append(seat)

    timings['parse'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── Deduzir ações ──
    active_seats = [s for s in seats if s['active']]
    table['num_active'] = len(active_seats)

    max_bet = max((s['bet'] for s in active_seats), default=0)
    if max_bet > 0:
        bettor = next((s for s in active_seats if s['bet'] == max_bet), None)
        table['last_action'] = f"Bet {int(max_bet):,}"
        table['bet_to_call'] = max_bet
        if bettor:
            bettor['action'] = f"Bet {int(max_bet):,}"

    # ── Ordenar seats por posição poker ──
    if btn_seat_idx >= 0:
        for s in seats:
            s['_po'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
        seats.sort(key=lambda s: s['_po'])
        for s in seats:
            s.pop('_po', None)

    # ── Log final ──
    elapsed = time.time() - t0
    print(f"\n✅ Análise completa em {elapsed:.1f}s ({len(batch_images)} OCR batch)")
    print(f"   Board: {table['board']} ({table['street']})")
    print(f"   Pot: {table['pot']} | Blinds: {table['blinds']} | BTN: seat {table['btn_seat']}")
    print(f"   Seats ativos: {table['num_active']}")
    for s in seats:
        if s['active']:
            hero_tag = ' ★HERO' if s['is_hero'] else ''
            print(f"     [{s['id']:13s}] {s['position']:6s} {s['name']:15s} stack={s['stack']:>8,} bet={s['bet']:>6,} cards={s['cards']}{hero_tag}")

    return {'table': table, 'seats': seats, 'timings': timings}, elapsed


# ============ FLASK SERVER ============

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get('OCR_API_KEY', '')


def check_auth():
    if not API_KEY:
        return True
    key = request.headers.get('X-API-Key', '') or request.args.get('key', '')
    return key == API_KEY


def base64_to_pil(b64):
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


@app.route('/health', methods=['GET'])
def health():
    coords = active_coords or load_coords()
    gpu_name = 'N/A'
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    except:
        pass
    # Extrair IDs das posições configuradas
    seat_ids = [sr.get('id', f'seat_{i+1}') for i, sr in enumerate(coords.get('seat_regions', []))]
    return jsonify({
        'status': 'ok',
        'ocr_engine': 'Chandra OCR 2',
        'gpu': gpu_name,
        'preset': coords.get('name', 'custom'),
        'seats': coords.get('seats', 8),
        'seat_ids': seat_ids,
        'ocr_ready': _ocr_manager is not None,
    })


@app.route('/warmup', methods=['POST'])
def warmup():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    t0 = time.time()
    get_ocr_manager()
    return jsonify({'status': 'ready', 'elapsed': f"{time.time()-t0:.1f}s"})


@app.route('/analyze', methods=['POST'])
def analyze():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image'}), 400

    try:
        img_pil = base64_to_pil(data['image'])
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400

    print(f"\n{'='*60}")
    print(f"📸 Nova análise — imagem {img_pil.size[0]}x{img_pil.size[1]}")

    analysis, elapsed = analyze_table(img_pil)

    # Override dealer se frontend enviou manualmente
    if 'btn_seat_idx' in data and data['btn_seat_idx'] >= 0:
        coords_data = active_coords or load_coords()
        num_seats = int(coords_data.get('seats', 8))
        btn_idx = int(data['btn_seat_idx'])
        pos_map = POS_MAPS.get(num_seats, POS_MAPS[8])
        analysis['table']['btn_seat'] = btn_idx + 1
        for s in analysis['seats']:
            i = int(s['seat']) - 1  # 'seat' é o número sequencial (int), não 'id' (string)
            s['position'] = pos_map[(num_seats - btn_idx + i) % num_seats]
        for s in analysis['seats']:
            s['_po'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
        analysis['seats'].sort(key=lambda s: s['_po'])
        for s in analysis['seats']:
            s.pop('_po', None)
        print(f"  ⚠️  BTN manual override: seat {btn_idx + 1}")

    hero = next((s for s in analysis['seats'] if s['is_hero']), None)
    hero_cards = hero['cards'] if hero else []
    has_hero_cards = len([c for c in hero_cards if c]) == 2

    return jsonify({
        'success': has_hero_cards,
        'table': analysis['table'],
        'seats': analysis['seats'],
        'timings': analysis.get('timings', {}),
        'elapsed': f"{elapsed:.1f}",
    })


# IDs padrão por posição física (8-max, anti-horário a partir do hero)
DEFAULT_SEAT_IDS = [
    {'id': 'hero',         'label': 'HERO',         'position': 'bottom_center'},
    {'id': 'bottom_left',  'label': 'Bottom Left',  'position': 'bottom_left'},
    {'id': 'left',         'label': 'Left',         'position': 'left'},
    {'id': 'top_left',     'label': 'Top Left',     'position': 'top_left'},
    {'id': 'top',          'label': 'Top',          'position': 'top_center'},
    {'id': 'top_right',    'label': 'Top Right',    'position': 'top_right'},
    {'id': 'right',        'label': 'Right',        'position': 'right'},
    {'id': 'bottom_right', 'label': 'Bottom Right', 'position': 'bottom_right'},
]


@app.route('/update-coords', methods=['POST'])
def update_coords():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing coords data'}), 400

    # Injetar IDs de posição se não vieram da calibração
    for i, sr in enumerate(data.get('seat_regions', [])):
        if 'id' not in sr and i < len(DEFAULT_SEAT_IDS):
            sr.update(DEFAULT_SEAT_IDS[i])

    save_coords(data)
    return jsonify({'status': 'ok', 'message': 'Coordenadas salvas'})


@app.route('/get-coords', methods=['GET'])
def get_coords():
    coords = active_coords or load_coords()
    return jsonify(coords)


@app.route('/ocr-test', methods=['POST'])
def ocr_test():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    if not data or 'image' not in data or 'region' not in data:
        return jsonify({'error': 'Missing image or region'}), 400
    try:
        img_pil = base64_to_pil(data['image'])
        mode = data.get('mode', 'text')
        cropped = crop_region(img_pil, data['region'])
        result = ocr_single(cropped)
        return jsonify({'text': str(result), 'mode': mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/presets', methods=['GET'])
def list_presets():
    return jsonify({k: {'name': v['name'], 'seats': v['seats']}
                    for k, v in PRESETS.items()})


@app.route('/presets/<name>', methods=['POST'])
def load_preset(name):
    if name not in PRESETS:
        return jsonify({'error': f'Preset "{name}" not found'}), 404
    save_coords(PRESETS[name])
    return jsonify({'status': 'ok', 'preset': name})


# ============ MAIN ============

if __name__ == '__main__':
    print("🃏 Poker Vision OCR Server")
    print("=" * 40)

    load_coords()

    print("\n🔄 Pré-carregando Chandra OCR 2...")
    get_ocr_manager()

    port = int(os.environ.get('PORT', 8080))
    print(f"\n🚀 Servidor rodando em http://0.0.0.0:{port}")
    print(f"   Endpoints: /health, /analyze, /update-coords, /ocr-test, /warmup")
    app.run(host='0.0.0.0', port=port, debug=False)
