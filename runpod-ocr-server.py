"""
Poker Vision OCR Server — RunPod (RTX 3090)
Abordagem: Chandra OCR puro + coordenadas fixas + lógica completa server-side.
O HTML é apenas um viewer: captura screenshot → envia → exibe resultado.

Endpoints:
  GET  /health          → status do servidor
  POST /analyze         → análise completa da mesa
  POST /update-coords   → atualizar coordenadas (calibração)
  POST /ocr-test        → testar OCR numa região específica
"""

import os
import re
import json
import time
import base64
import io
import torch
torch.backends.cudnn.enabled = False  # compatibilidade cu124 no RunPod
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ============ OCR ENGINE (Chandra OCR 2) ============

_ocr_model = None

def get_ocr():
    """Lazy-load Chandra OCR 2 (4B params) — primeiro call demora ~30s"""
    global _ocr_model
    if _ocr_model is None:
        print("🔄 Carregando Chandra OCR 2 (4B)... isso leva ~30s na primeira vez")
        t0 = time.time()
        try:
            from chandra.model import InferenceManager
            from chandra.model.schema import BatchInputItem

            manager = InferenceManager(method="hf")

            def ocr_fn(img):
                results = manager.generate([BatchInputItem(image=img, prompt_type="ocr")])
                return results[0].raw or results[0].markdown or ""

            # Warmup com imagem dummy
            dummy = Image.new('RGB', (100, 30), (255, 255, 255))
            ocr_fn(dummy)
            _ocr_model = ocr_fn
            print(f"✅ Chandra OCR 2 pronto em {time.time()-t0:.1f}s")
        except ImportError as e:
            print(f"⚠️  chandra-ocr não disponível ({e}), usando fallback pytesseract")
            try:
                import pytesseract
                _ocr_model = lambda img: pytesseract.image_to_string(img).strip()
                print("✅ Fallback pytesseract ativo")
            except ImportError:
                print("❌ Nenhum OCR disponível!")
                _ocr_model = lambda img: ""
    return _ocr_model


def ocr_crop(img_pil, region, mode='text'):
    """
    Recorta região da imagem e faz OCR.
    region: {x, y, w, h} em pixels OU {x, y, w, h} como frações (0-1)
    mode: 'text' | 'number' | 'card'
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

    ocr_fn = get_ocr()
    try:
        result = ocr_fn(cropped)
        text = result if isinstance(result, str) else str(result)
        text = text.strip()
    except Exception as e:
        print(f"  OCR error: {e}")
        return '' if mode == 'text' else 0

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
    Tenta OCR em cada região de dealer — a que retornar "D" é o BTN.
    Fallback: detectar círculo amarelo por cor.
    """
    ocr_fn = get_ocr()
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

        # Método 1: Detectar círculo amarelo/dourado do dealer button
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


# ============ COORDENADAS FIXAS — PRESETS ============

# GGPoker 8-max Bounty Hunters (janela 1260x898)
# Coordenadas em fração (0-1) do tamanho da imagem
# Mapeadas a partir do layout padrão GGPoker

PRESETS = {
    'ggpoker_8max_bounty': {
        'name': 'GGPoker 8-max Bounty Hunters',
        'seats': 8,

        # ── Info da mesa (barra superior com level, blinds, prize) ──
        'blinds': {'x': 0.00, 'y': 0.00, 'w': 0.40, 'h': 0.07},

        # ── Pot (centro da mesa) ──
        'pot': {'x': 0.33, 'y': 0.28, 'w': 0.34, 'h': 0.06},

        # ── Board cards (5 posições — flop/turn/river) ──
        # Mapeado do screenshot 1: 7h Qd 5h ocupam ~30-56% horizontal, ~33-55% vertical
        'board': [
            {'x': 0.29, 'y': 0.33, 'w': 0.09, 'h': 0.19},
            {'x': 0.38, 'y': 0.33, 'w': 0.09, 'h': 0.19},
            {'x': 0.47, 'y': 0.33, 'w': 0.09, 'h': 0.19},
            {'x': 0.56, 'y': 0.33, 'w': 0.09, 'h': 0.19},
            {'x': 0.65, 'y': 0.33, 'w': 0.09, 'h': 0.19},
        ],

        # ── Botões de ação do Hero (Fold / Call / Raise / Check) ──
        'hero_actions': {'x': 0.53, 'y': 0.87, 'w': 0.45, 'h': 0.07},

        # ── Seats (anti-horário a partir do Hero, como GGPoker) ──
        'seat_regions': [
            # Seat 1: HERO — TurnZero (bottom center)
            {
                'name':    {'x': 0.42, 'y': 0.87, 'w': 0.16, 'h': 0.04},
                'stack':   {'x': 0.42, 'y': 0.91, 'w': 0.16, 'h': 0.05},
                'cards':   {'x': 0.42, 'y': 0.70, 'w': 0.16, 'h': 0.17},
                'bet':     {'x': 0.44, 'y': 0.65, 'w': 0.12, 'h': 0.04},
                'dealer':  {'x': 0.40, 'y': 0.67, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.45, 'y': 0.67, 'w': 0.10, 'h': 0.03},
            },
            # Seat 2: ↙ — terryu7575! (bottom-left)
            {
                'name':    {'x': 0.14, 'y': 0.77, 'w': 0.15, 'h': 0.04},
                'stack':   {'x': 0.14, 'y': 0.81, 'w': 0.15, 'h': 0.05},
                'cards':   {'x': 0.12, 'y': 0.62, 'w': 0.14, 'h': 0.14},
                'bet':     {'x': 0.22, 'y': 0.58, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.27, 'y': 0.69, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.14, 'y': 0.61, 'w': 0.08, 'h': 0.03},
            },
            # Seat 3: ← — Munkh_ (left)
            {
                'name':    {'x': 0.01, 'y': 0.53, 'w': 0.13, 'h': 0.04},
                'stack':   {'x': 0.01, 'y': 0.57, 'w': 0.13, 'h': 0.05},
                'cards':   {'x': 0.00, 'y': 0.37, 'w': 0.11, 'h': 0.14},
                'bet':     {'x': 0.13, 'y': 0.53, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.12, 'y': 0.48, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.02, 'y': 0.36, 'w': 0.08, 'h': 0.03},
            },
            # Seat 4: ↖ — ProCheatano (top-left)
            {
                'name':    {'x': 0.16, 'y': 0.22, 'w': 0.16, 'h': 0.04},
                'stack':   {'x': 0.16, 'y': 0.26, 'w': 0.16, 'h': 0.05},
                'cards':   {'x': 0.16, 'y': 0.08, 'w': 0.14, 'h': 0.14},
                'bet':     {'x': 0.28, 'y': 0.31, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.30, 'y': 0.27, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.18, 'y': 0.06, 'w': 0.08, 'h': 0.03},
            },
            # Seat 5: ↑ — Dabrowski (top center)
            {
                'name':    {'x': 0.42, 'y': 0.12, 'w': 0.16, 'h': 0.04},
                'stack':   {'x': 0.42, 'y': 0.16, 'w': 0.16, 'h': 0.05},
                'cards':   {'x': 0.43, 'y': 0.00, 'w': 0.14, 'h': 0.12},
                'bet':     {'x': 0.44, 'y': 0.22, 'w': 0.12, 'h': 0.04},
                'dealer':  {'x': 0.41, 'y': 0.20, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.47, 'y': 0.03, 'w': 0.08, 'h': 0.03},
            },
            # Seat 6: ↗ — PetinhoShowtime (top-right)
            {
                'name':    {'x': 0.68, 'y': 0.22, 'w': 0.18, 'h': 0.04},
                'stack':   {'x': 0.68, 'y': 0.26, 'w': 0.18, 'h': 0.05},
                'cards':   {'x': 0.72, 'y': 0.08, 'w': 0.14, 'h': 0.14},
                'bet':     {'x': 0.62, 'y': 0.31, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.66, 'y': 0.27, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.74, 'y': 0.06, 'w': 0.08, 'h': 0.03},
            },
            # Seat 7: → — zhouxiaoyy (right)
            {
                'name':    {'x': 0.82, 'y': 0.53, 'w': 0.16, 'h': 0.04},
                'stack':   {'x': 0.82, 'y': 0.57, 'w': 0.16, 'h': 0.05},
                'cards':   {'x': 0.84, 'y': 0.37, 'w': 0.14, 'h': 0.14},
                'bet':     {'x': 0.76, 'y': 0.53, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.82, 'y': 0.48, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.86, 'y': 0.36, 'w': 0.08, 'h': 0.03},
            },
            # Seat 8: ↘ — Jiaobi (bottom-right)
            {
                'name':    {'x': 0.72, 'y': 0.77, 'w': 0.16, 'h': 0.04},
                'stack':   {'x': 0.72, 'y': 0.81, 'w': 0.16, 'h': 0.05},
                'cards':   {'x': 0.74, 'y': 0.62, 'w': 0.14, 'h': 0.14},
                'bet':     {'x': 0.65, 'y': 0.58, 'w': 0.10, 'h': 0.04},
                'dealer':  {'x': 0.70, 'y': 0.69, 'w': 0.04, 'h': 0.05},
                'bounty':  {'x': 0.76, 'y': 0.61, 'w': 0.08, 'h': 0.03},
            },
        ],
    }
}

# Coordenadas ativas (carregadas do preset ou calibração manual)
active_coords = None
coords_file = os.path.join(os.path.dirname(__file__), 'coords.json')


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
    Análise completa da mesa de poker.
    Retorna dados agrupados por seat ID (1-8).
    Estrutura: { table: {...}, seats: [{id:1, ...}, {id:2, ...}, ...] }
    """
    coords = active_coords or load_coords()
    num_seats = coords.get('seats', 8)
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
    seats = []  # [{id, name, stack, cards, position, bet, bounty, is_hero, active, action}]

    # ── 1. Blinds ──
    t1 = time.time()
    if coords.get('blinds'):
        raw = ocr_crop(img_pil, coords['blinds'], 'text')
        m = re.search(r'(\d[\d,.]*)\s*[|/]\s*(\d[\d,.]*)', raw)
        if m:
            table['blinds'] = f"{m.group(1)}/{m.group(2)}"
        elif raw:
            table['blinds'] = raw
        print(f"  Blinds: {table['blinds']}")
    timings['blinds'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── 2. Pot ──
    t1 = time.time()
    if coords.get('pot'):
        raw = ocr_crop(img_pil, coords['pot'], 'text')
        m = re.search(r'(\d[\d,.]+)', raw.replace(' ', ''))
        if m:
            table['pot'] = parse_number(m.group(1))
        print(f"  Pot: {table['pot']}")
    timings['pot'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── 3. Board ──
    t1 = time.time()
    for i, card_region in enumerate(coords.get('board', [])):
        if not card_region:
            continue
        if detect_seat_active(img_pil, card_region):
            card = ocr_crop(img_pil, card_region, 'card')
            if card:
                table['board'].append(card)
                print(f"  Board {i+1}: {card}")
    n_board = len(table['board'])
    table['street'] = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}.get(n_board, 'preflop')
    timings['board'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── 4. Dealer detection ──
    t1 = time.time()
    dealer_regions = [s.get('dealer') for s in seat_regions]
    btn_seat_idx = detect_dealer_position(img_pil, dealer_regions)
    table['btn_seat'] = btn_seat_idx + 1 if btn_seat_idx >= 0 else -1  # 1-indexed
    timings['dealer'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  Dealer: seat {table['btn_seat']}")

    # ── 5. OCR de todos os seats (1-8) ──
    t1 = time.time()
    pos_maps = {
        2: ['BTN','BB'], 3: ['BTN','SB','BB'], 4: ['BTN','SB','BB','UTG'],
        5: ['BTN','SB','BB','UTG','CO'], 6: ['BTN','SB','BB','UTG','HJ','CO'],
        7: ['BTN','SB','BB','UTG','MP','HJ','CO'],
        8: ['BTN','SB','BB','UTG','UTG+1','MP','HJ','CO'],
        9: ['BTN','SB','BB','UTG','UTG+1','LJ','MP','HJ','CO'],
    }
    pos_map = pos_maps.get(num_seats, pos_maps[8])

    for i, sr in enumerate(seat_regions):
        seat_id = i + 1  # 1-indexed
        is_hero = (i == 0)

        seat = {
            'id': seat_id,
            'name': '',
            'stack': 0,
            'cards': ['', ''],
            'position': '?',
            'bet': 0,
            'bounty': '',
            'is_hero': is_hero,
            'active': False,
            'action': '',
        }

        # Detectar se seat está ativo
        is_active = True
        if sr.get('cards'):
            is_active = detect_seat_active(img_pil, sr['cards'])
        seat['active'] = is_active

        if not is_active:
            seats.append(seat)
            continue

        # Nome
        if sr.get('name'):
            raw = ocr_crop(img_pil, sr['name'], 'text')
            if raw and len(raw) > 1:
                seat['name'] = raw
            else:
                seat['name'] = 'Hero' if is_hero else f'Player{seat_id}'
            # Sitting Out → inativo
            if 'sitting' in seat['name'].lower() or 'sit out' in seat['name'].lower():
                seat['active'] = False
                seats.append(seat)
                continue
            print(f"  Seat {seat_id} name: {seat['name']}")

        # Stack
        if sr.get('stack'):
            seat['stack'] = ocr_crop(img_pil, sr['stack'], 'number')
            print(f"  Seat {seat_id} stack: {seat['stack']}")

        # Cards (dividir região ao meio: carta 1 e carta 2)
        if sr.get('cards') and is_active:
            cr = sr['cards']
            half_w = cr['w'] / 2
            c1_r = {'x': cr['x'], 'y': cr['y'], 'w': half_w, 'h': cr['h']}
            c2_r = {'x': cr['x'] + half_w, 'y': cr['y'], 'w': half_w, 'h': cr['h']}
            c1 = ocr_crop(img_pil, c1_r, 'card')
            c2 = ocr_crop(img_pil, c2_r, 'card')
            seat['cards'] = [c1, c2]
            if c1 or c2:
                print(f"  Seat {seat_id} cards: {c1} {c2}")

        # Bet
        if sr.get('bet'):
            seat['bet'] = ocr_crop(img_pil, sr['bet'], 'number')
            if seat['bet'] > 0:
                print(f"  Seat {seat_id} bet: {seat['bet']}")

        # Bounty
        if sr.get('bounty'):
            seat['bounty'] = ocr_crop(img_pil, sr['bounty'], 'text')

        # Posição poker (baseado no dealer)
        if btn_seat_idx >= 0:
            seat['position'] = pos_map[(num_seats - btn_seat_idx + i) % num_seats]

        seats.append(seat)

    timings['seats'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── 6. Deduzir ações ──
    active_seats = [s for s in seats if s['active']]
    table['num_active'] = len(active_seats)

    # Maior bet = last_action
    max_bet = max((s['bet'] for s in active_seats), default=0)
    if max_bet > 0:
        bettor = next((s for s in active_seats if s['bet'] == max_bet), None)
        table['last_action'] = f"Bet {int(max_bet):,}"
        table['bet_to_call'] = max_bet
        if bettor:
            bettor['action'] = f"Bet {int(max_bet):,}"

    # ── 7. Ordenar seats por posição poker (BTN→SB→BB→UTG...) mantendo o ID ──
    if btn_seat_idx >= 0:
        for s in seats:
            if s['position'] != '?':
                s['pos_order'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
            else:
                s['pos_order'] = 99
        seats.sort(key=lambda s: s['pos_order'])
        # Remover campo auxiliar
        for s in seats:
            s.pop('pos_order', None)

    # ── 8. Timing e log ──
    elapsed = time.time() - t0

    print(f"\n✅ Análise completa em {elapsed:.1f}s")
    print(f"   Board: {table['board']} ({table['street']})")
    print(f"   Pot: {table['pot']} | Blinds: {table['blinds']} | BTN: seat {table['btn_seat']}")
    print(f"   Seats ativos: {table['num_active']}")
    for s in seats:
        if s['active']:
            hero_tag = ' ★HERO' if s['is_hero'] else ''
            print(f"     [{s['id']}] {s['position']:6s} {s['name']:15s} stack={s['stack']:>8,} bet={s['bet']:>6,} cards={s['cards']}{hero_tag}")

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
    return jsonify({
        'status': 'ok',
        'ocr_engine': 'Chandra OCR 2',
        'gpu': 'RTX 3090',
        'preset': coords.get('name', 'custom'),
        'seats': coords.get('seats', 8),
        'ocr_ready': _ocr_model is not None,
    })


@app.route('/warmup', methods=['POST'])
def warmup():
    """Pré-carregar modelo OCR (primeiro request demora ~30s)"""
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    t0 = time.time()
    get_ocr()
    return jsonify({'status': 'ready', 'elapsed': f"{time.time()-t0:.1f}s"})


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Análise completa. Recebe: {image: base64}
    Opcionalmente: {image: base64, btn_seat_idx: 3} para forçar posição do dealer
    """
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
        num_seats = coords_data.get('seats', 8)
        btn_idx = data['btn_seat_idx']
        pos_maps = {
            2: ['BTN','BB'], 3: ['BTN','SB','BB'], 4: ['BTN','SB','BB','UTG'],
            5: ['BTN','SB','BB','UTG','CO'], 6: ['BTN','SB','BB','UTG','HJ','CO'],
            7: ['BTN','SB','BB','UTG','MP','HJ','CO'],
            8: ['BTN','SB','BB','UTG','UTG+1','MP','HJ','CO'],
            9: ['BTN','SB','BB','UTG','UTG+1','LJ','MP','HJ','CO'],
        }
        pos_map = pos_maps.get(num_seats, pos_maps[8])
        analysis['table']['btn_seat'] = btn_idx + 1
        for s in analysis['seats']:
            i = s['id'] - 1  # 0-indexed
            s['position'] = pos_map[(num_seats - btn_idx + i) % num_seats]
        # Re-sort by position
        for s in analysis['seats']:
            s['_po'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
        analysis['seats'].sort(key=lambda s: s['_po'])
        for s in analysis['seats']:
            s.pop('_po', None)
        print(f"  ⚠️  BTN manual override: seat {btn_idx + 1}")

    # Hero cards para checar sucesso
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


@app.route('/update-coords', methods=['POST'])
def update_coords():
    """Atualizar coordenadas (calibração). Recebe JSON com novo mapeamento."""
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing coords data'}), 400
    save_coords(data)
    return jsonify({'status': 'ok', 'message': 'Coordenadas salvas'})


@app.route('/get-coords', methods=['GET'])
def get_coords():
    """Retorna coordenadas ativas (para o frontend de calibração)"""
    coords = active_coords or load_coords()
    return jsonify(coords)


@app.route('/ocr-test', methods=['POST'])
def ocr_test():
    """Testar OCR numa região específica. Útil para calibrar."""
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    if not data or 'image' not in data or 'region' not in data:
        return jsonify({'error': 'Missing image or region'}), 400
    try:
        img_pil = base64_to_pil(data['image'])
        mode = data.get('mode', 'text')
        result = ocr_crop(img_pil, data['region'], mode)
        return jsonify({'text': str(result), 'mode': mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/presets', methods=['GET'])
def list_presets():
    """Listar presets disponíveis"""
    return jsonify({k: {'name': v['name'], 'seats': v['seats']}
                    for k, v in PRESETS.items()})


@app.route('/presets/<name>', methods=['POST'])
def load_preset(name):
    """Carregar um preset específico"""
    if name not in PRESETS:
        return jsonify({'error': f'Preset "{name}" not found'}), 404
    save_coords(PRESETS[name])
    return jsonify({'status': 'ok', 'preset': name})


# ============ MAIN ============

if __name__ == '__main__':
    print("🃏 Poker Vision OCR Server")
    print("=" * 40)

    # Carregar coordenadas
    load_coords()

    # Pré-carregar OCR na startup
    print("\n🔄 Pré-carregando Chandra OCR 2...")
    get_ocr()

    port = int(os.environ.get('PORT', 8080))
    print(f"\n🚀 Servidor rodando em http://0.0.0.0:{port}")
    print(f"   Endpoints: /health, /analyze, /update-coords, /ocr-test, /warmup")
    app.run(host='0.0.0.0', port=port, debug=False)
