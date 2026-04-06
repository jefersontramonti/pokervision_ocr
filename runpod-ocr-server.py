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
import math

# ── Fix cuDNN: incompatível com torch 2.5.1 downgrade no RunPod ──
# Ainda usa GPU (CUDA) para compute, só desabilita cuDNN para conv ops
import torch
torch.backends.cudnn.enabled = False

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ============ CONSTANTES ============

# Mapa de posições poker por número de seats (constante global, sem duplicação)
POS_MAPS = {
    2: ['BTN','BB'], 3: ['BTN','SB','BB'], 4: ['BTN','SB','BB','UTG'],
    5: ['BTN','SB','BB','UTG','CO'], 6: ['BTN','SB','BB','UTG','HJ','CO'],
    7: ['BTN','SB','BB','UTG','MP','HJ','CO'],
    8: ['BTN','SB','BB','UTG','UTG+1','MP','HJ','CO'],
    9: ['BTN','SB','BB','UTG','UTG+1','LJ','MP','HJ','CO'],
}


def _valid_region(region):
    """Verifica se uma região de coordenadas é válida (não zerada)."""
    if not region or not isinstance(region, dict):
        return False
    return region.get('w', 0) > 0 and region.get('h', 0) > 0


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


# ============ OCR ENGINE — EasyOCR (rápido, GPU) ============

_easyocr_reader = None

def get_easyocr_reader():
    """Lazy-load EasyOCR com GPU. ~5s no primeiro call, depois instantâneo."""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("🔄 Carregando EasyOCR (GPU)…")
        t0 = time.time()
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print(f"✅ EasyOCR pronto em {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"⚠️  EasyOCR não disponível: {e}")
            _easyocr_reader = None
    return _easyocr_reader


def _easyocr_read(pil_img):
    """Lê texto de uma imagem PIL com EasyOCR. Retorna string limpa."""
    reader = get_easyocr_reader()
    if reader is None:
        return ''
    try:
        arr = np.array(pil_img.convert('RGB'))
        results = reader.readtext(arr, detail=0, paragraph=False)
        return ' '.join(str(r) for r in results).strip()
    except Exception as e:
        print(f"  EasyOCR erro: {e}")
        return ''


# ── Detecção de carta: rank via EasyOCR + suit via cor/forma ─────────

_RANK_VALID = set('AKQJT98765432')


def _card_rank(card_img):
    """Lê o rank da carta (canto superior esquerdo)."""
    w, h = card_img.size
    corner = card_img.crop((0, 0, max(1, int(w * 0.40)), max(1, int(h * 0.50))))
    raw = _easyocr_read(corner).upper().replace(' ', '')
    if '10' in raw or 'IO' in raw or '1O' in raw:
        return 'T'
    for ch in raw:
        if ch in _RANK_VALID:
            return ch
    return ''


def _card_suit(card_img):
    """
    Detecta o naipe por análise de cor + forma (sem modelo).
    Suporta deck 2 cores (♥♦=vermelho, ♠♣=preto)
    e deck 4 cores GGPoker (♦=azul, ♣=verde).
    """
    arr = np.array(card_img.convert('RGB')).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    n = float(max(arr.shape[0] * arr.shape[1], 1))

    red_pct   = float(np.sum((r > 160) & (g < 110) & (b < 110)) / n)
    blue_pct  = float(np.sum((b > 140) & (r < 120) & (g < 160)) / n)
    green_pct = float(np.sum((g > 140) & (r < 120) & (b < 120)) / n)
    dark_pct  = float(np.sum((r < 80)  & (g < 80)  & (b < 80))  / n)

    # Deck 4 cores GGPoker (diamante=azul, paus=verde)
    if blue_pct > 0.015 and blue_pct > red_pct:
        return 'd'
    if green_pct > 0.015 and green_pct > red_pct and green_pct > dark_pct:
        return 'c'

    if red_pct > 0.015 and red_pct >= dark_pct:
        # Vermelho: Hearts vs Diamonds
        mask = (r > 160) & (g < 110) & (b < 110)
        rows = [int(np.sum(mask[i])) for i in range(mask.shape[0])]
        rows_nz = [w for w in rows if w > 0]
        if not rows_nz:
            return 'h'
        top_w = rows_nz[0]
        max_w = max(rows_nz)
        # Diamond: topo pontiagudo (top_w << max_w)
        return 'd' if top_w < max_w * 0.20 else 'h'

    if dark_pct > 0.015:
        # Preto: Spades vs Clubs
        try:
            import cv2
            mask = ((arr[:, :, 0] < 80) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)).astype(np.uint8)
            h_img = mask.shape[0]
            top   = mask[:int(h_img * 0.70), :] * 255
            n_labels, _ = cv2.connectedComponents(top)
            # Clubs: 3 círculos = 3+ componentes; Spades: 1-2
            return 'c' if (n_labels - 1) >= 3 else 's'
        except Exception:
            return 's'

    return ''


def detect_card_fast(card_img):
    """Detecta uma carta (rank + suit). Ex: 'Ah', 'Td', ''."""
    rank = _card_rank(card_img)
    suit = _card_suit(card_img)
    return (rank + suit) if (rank and suit) else ''


# ============ ANÁLISE RÁPIDA — EasyOCR + pixel analysis ============

def analyze_table_fast(img_pil):
    """
    Análise rápida: EasyOCR (GPU) para texto + análise de cor/forma para cartas.
    Alvo: ~2-4s na RTX 4090 vs 25s do mosaico Chandra.

    Fluxo:
      1. Detecção pixel: dealer button + seats ativos
      2. Recorte das regiões calibradas
      3. EasyOCR em paralelo nos crops de texto (pot, stacks, nomes)
      4. detect_card_fast para cada carta (rank EasyOCR + suit pixel)
      5. Montagem do resultado
    """
    coords       = active_coords or load_coords()
    num_seats    = int(coords.get('seats', 8))
    seat_regions = coords.get('seat_regions', [])

    t0 = time.time()
    timings = {}

    # ── Fase 1: Dealer + seats ativos (pixel, sem OCR) ─────────────
    t1 = time.time()
    dealer_regions = [s.get('dealer') for s in seat_regions]
    btn_seat_idx   = detect_dealer_position(img_pil, dealer_regions)
    timings['dealer'] = f"{(time.time()-t1)*1000:.0f}ms"

    t1 = time.time()
    seat_active = [
        detect_seat_active(img_pil, sr.get('cards', {}), is_hero=(i == 0))
        for i, sr in enumerate(seat_regions)
    ]
    timings['detect'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── Fase 2: Recortes ────────────────────────────────────────────
    t1 = time.time()
    pot_img = crop_region(img_pil, coords['pot']) if _valid_region(coords.get('pot')) else None

    board_imgs = []
    for r in coords.get('board', []):
        if _valid_region(r) and detect_seat_active(img_pil, r, is_hero=True):
            board_imgs.append(crop_region(img_pil, r))
        else:
            board_imgs.append(None)

    seat_imgs = []
    for i, sr in enumerate(seat_regions):
        act = seat_active[i]
        seat_imgs.append({
            'name':  crop_region(img_pil, sr['name'])  if act and _valid_region(sr.get('name'))  else None,
            'stack': crop_region(img_pil, sr['stack']) if act and _valid_region(sr.get('stack')) else None,
            'cards': crop_region(img_pil, sr['cards']) if act and _valid_region(sr.get('cards')) else None,
        })
    timings['crop'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── Fase 3: OCR + detecção de cartas ───────────────────────────
    t1 = time.time()
    ocr_data = {}

    # Texto (EasyOCR): pot, nomes, stacks
    reader = get_easyocr_reader()
    if reader:
        text_pairs = []
        if pot_img:
            text_pairs.append(('pot', pot_img))
        for i, si in enumerate(seat_imgs):
            if si.get('name'):
                text_pairs.append((f'nm{i}', si['name']))
            if si.get('stack'):
                text_pairs.append((f'st{i}', si['stack']))

        for key, img in text_pairs:
            ocr_data[key] = _easyocr_read(img)

    # Cartas do board (rank EasyOCR + suit pixel)
    board_cards = [detect_card_fast(img) if img else None for img in board_imgs]

    # Cartas do hero: cortar ao meio (c1=esquerda, c2=direita)
    hero_c1, hero_c2 = '', ''
    hero_img = seat_imgs[0].get('cards') if seat_imgs else None
    if hero_img:
        w, h = hero_img.size
        hero_c1 = detect_card_fast(hero_img.crop((0, 0, w // 2, h)))
        hero_c2 = detect_card_fast(hero_img.crop((w // 2, 0, w, h)))

    timings['ocr'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── Fase 4: Montar resultado ────────────────────────────────────
    t1 = time.time()
    pos_map = POS_MAPS.get(num_seats, POS_MAPS[8])

    pot     = parse_number(ocr_data.get('pot', ''))
    board   = [c for c in board_cards if c]
    n_board = len(board)
    street  = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}.get(n_board, 'preflop')

    seats = []
    for i, sr in enumerate(seat_regions):
        is_hero  = (i == 0)
        active   = seat_active[i]
        name     = ocr_data.get(f'nm{i}', '').strip() or ('Hero' if is_hero else f'Player{i+1}')
        st_text  = ocr_data.get(f'st{i}', '')
        stack    = parse_number(st_text)
        all_in   = bool(re.search(r'all.?in', st_text, re.IGNORECASE))

        if re.search(r'sit.?out|sitting', name, re.IGNORECASE):
            active = False

        position = '?'
        if btn_seat_idx >= 0:
            position = pos_map[(num_seats - btn_seat_idx + i) % num_seats]

        seats.append({
            'seat':     i + 1,
            'id':       sr.get('id',    f'seat_{i+1}'),
            'label':    sr.get('label', f'Seat {i+1}'),
            'name':     name,
            'stack':    stack,
            'all_in':   all_in,
            'cards':    [hero_c1, hero_c2] if is_hero else ['', ''],
            'position': position,
            'bet':      0,
            'bounty':   '',
            'is_hero':  is_hero,
            'active':   active,
            'action':   '',
        })

    active_seats = [s for s in seats if s['active']]

    # Ordenar por posição
    if btn_seat_idx >= 0:
        for s in seats:
            s['_po'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
        seats.sort(key=lambda s: s['_po'])
        for s in seats:
            s.pop('_po', None)

    timings['parse'] = f"{(time.time()-t1)*1000:.0f}ms"
    elapsed = time.time() - t0

    print(f"\n✅ Fast análise em {elapsed:.1f}s | Board: {board} | Pot: {pot}")
    for s in seats:
        if s['active']:
            print(f"  [{s['id']:13s}] {s['position']:6s} {s['name']:15s} stack={s['stack']} all_in={s.get('all_in')} cards={s['cards']}")

    table = {
        'pot':         pot,
        'board':       board,
        'street':      street,
        'blinds':      '',
        'btn_seat':    btn_seat_idx + 1 if btn_seat_idx >= 0 else -1,
        'num_active':  len(active_seats),
        'last_action': '',
        'bet_to_call': 0,
        'tournament':  True,
        'ocr_parsed':  {k: v[:120] for k, v in ocr_data.items()},
    }

    success = bool(board or any(
        s['name'] not in ('Hero', f"Player{s['seat']}") for s in seats
    ))
    return {'table': table, 'seats': seats, 'timings': timings}, elapsed


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
        val = float(clean)
        return val if math.isfinite(val) else 0
    clean = clean.replace('.', '')
    try:
        val = float(clean) if clean else 0
        return val if math.isfinite(val) else 0
    except:
        return 0


def parse_card_text(text):
    """
    Parse OCR de uma carta. GGPoker mostra rank + suit gráfico.
    Chandra pode ler: 'A♠', 'K♥', 'Ah', '10s', 'Td',
                      'Ten of Clubs', 'Jack of Diamonds', etc.
    Retorna formato padronizado: 'Ah', 'Ks', 'Td', etc. ou '' se inválido.
    """
    if not text:
        return ''
    text = text.strip()

    # ── Descrição em inglês: "Ten of Clubs", "Jack of Diamonds", "10 of Spades" ──
    _rank_en = {'ace':'A','king':'K','queen':'Q','jack':'J','ten':'T',
                'nine':'9','eight':'8','seven':'7','six':'6','five':'5',
                'four':'4','three':'3','two':'2'}
    _suit_en = {'spades':'s','spade':'s','hearts':'h','heart':'h',
                'diamonds':'d','diamond':'d','clubs':'c','club':'c'}
    m_en = re.search(
        r'(ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|\d+)'
        r'\s+of\s+(spades?|hearts?|diamonds?|clubs?)',
        text, re.IGNORECASE
    )
    if m_en:
        rr, sr = m_en.group(1).lower(), m_en.group(2).lower()
        rank = _rank_en.get(rr) or ('T' if rr == '10' else rr if rr.isdigit() else '')
        suit = _suit_en.get(sr, '')
        if rank and suit:
            return rank + suit

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


def parse_two_cards(text):
    """
    Extrai duas cartas do hero de texto OCR.
    Suporta 3 formatos que o Chandra pode retornar:
      1. Inglês:   "Jack of Diamonds and 6 of Clubs"  → ('Jd', '6c')
      2. Símbolos: "J♦ 6♣"                            → ('Jd', '6c')
      3. Letras:   "Jd 6c"                            → ('Jd', '6c')
    """
    if not text:
        return '', ''

    # ── 1. Descrição em inglês (Chandra descreve a imagem da carta) ──
    rank_en = {
        'ace':'A','king':'K','queen':'Q','jack':'J','ten':'T',
        'nine':'9','eight':'8','seven':'7','six':'6','five':'5',
        'four':'4','three':'3','two':'2',
    }
    suit_en = {
        'spades':'s','spade':'s','hearts':'h','heart':'h',
        'diamonds':'d','diamond':'d','clubs':'c','club':'c',
    }
    cards_en = []
    tl = text.lower()
    for m in re.finditer(
        r'(ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|\d+)'
        r'\s+of\s+(spades?|hearts?|diamonds?|clubs?)',
        tl
    ):
        rank_raw, suit_raw = m.group(1), m.group(2)
        rank = rank_en.get(rank_raw) or ('T' if rank_raw == '10' else rank_raw if rank_raw.isdigit() else '')
        suit = suit_en.get(suit_raw, '')
        if rank and suit:
            cards_en.append(rank + suit)
        if len(cards_en) == 2:
            break
    if len(cards_en) >= 2:
        return cards_en[0], cards_en[1]

    # ── 2. Símbolos unicode ou letras (5♥ 4♥  /  Jd 6c) ──
    suit_map = {
        '♠':'s','♤':'s','♥':'h','♡':'h','♦':'d','♢':'d','♣':'c','♧':'c',
        's':'s','h':'h','d':'d','c':'c',
    }
    ranks = set('AKQJT98765432')
    t = text.replace('10', 'T').strip()
    cards_sym = []
    i = 0
    while i < len(t) and len(cards_sym) < 2:
        ch = t[i].upper()
        if ch in ranks:
            suit = ''
            for j in range(i + 1, min(i + 5, len(t))):
                cand = t[j]
                if cand in suit_map:
                    suit = suit_map[cand]; i = j; break
                if cand.lower() in suit_map:
                    suit = suit_map[cand.lower()]; i = j; break
            if suit:
                cards_sym.append(ch + suit)
        i += 1

    c1 = (cards_en + cards_sym + [''])[0]
    c2 = (cards_en + cards_sym + ['', ''])[1]
    return c1, c2


def detect_seat_active(img_pil, region, is_hero=False):
    """
    Detecta se um seat tem cartas visíveis.

    Vilões: procura o VERSO VERMELHO das cartas (GGPoker usa verso vermelho).
    Se a região de cards tem bastante pixel vermelho → jogador está na mão.
    Se não tem vermelho → foldou ou seat vazio.

    Hero: cartas são face-up (brancas com texto), usa brilho + variância.
    """
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

    total_pixels = max(1, arr.shape[0] * arr.shape[1])

    if len(arr.shape) != 3:
        return False

    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)

    gray = np.mean(arr, axis=2)
    variance = float(np.var(gray))

    # Cartas face-up (hero ou showdown): região clara com conteúdo variado
    bright_mask = (r > 160) & (g > 160) & (b > 160)
    bright_pct  = float(np.sum(bright_mask) / total_pixels * 100)
    face_up     = bright_pct > 15 and variance > 500

    if is_hero:
        return bool(face_up)
    else:
        # Vilões: verso VERMELHO (normal) OU cartas face-up (showdown/all-in)
        red_mask      = (r > 120) & (g < 80) & (b < 80) & (r > g * 1.8) & (r > b * 1.8)
        dark_red_mask = (r > 80)  & (r < 180) & (g < 60) & (b < 60) & (r > g * 2)
        red_pct       = float(np.sum(red_mask)      / total_pixels * 100)
        dark_red_pct  = float(np.sum(dark_red_mask) / total_pixels * 100)
        face_down     = red_pct > 10 or dark_red_pct > 15
        return bool(face_down or face_up)


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

def build_labeled_mosaic(regions):
    """
    Constrói mosaico vertical com cada crop precedido por label.
    Labels: PRETO fundo + BRANCO texto + fonte grande → alto contraste, OCR confiável.
    """
    from PIL import ImageDraw, ImageFont

    MOSAIC_W = 320   # largura total
    LABEL_H  = 32    # altura da faixa de label (maior = fonte maior)
    GAP      = 8     # espaço entre itens

    # Fonte maior para labels legíveis pelo OCR
    try:
        font = ImageFont.load_default(size=20)
    except Exception:
        font = ImageFont.load_default()

    def crop_h(label):
        return 60       # texto uniforme — cartas já vêm pré-recortadas no canto

    total_h = sum(LABEL_H + crop_h(lbl) + GAP for lbl, _ in regions)
    mosaic  = Image.new('RGB', (MOSAIC_W, total_h), (255, 255, 255))
    draw    = ImageDraw.Draw(mosaic)

    y = 0
    for label, crop in regions:
        ch = crop_h(label)

        # Faixa PRETA com texto BRANCO — máximo contraste, OCR confiável
        draw.rectangle([(0, y), (MOSAIC_W, y + LABEL_H - 1)], fill=(0, 0, 0))
        draw.text((8, y + 6), f'[{label}]', fill=(255, 255, 255), font=font)

        # Crop: escala proporcional, centralizado
        cw_orig, ch_orig = crop.size
        scale  = min(MOSAIC_W / max(cw_orig, 1), ch / max(ch_orig, 1))
        nw, nh = max(1, int(cw_orig * scale)), max(1, int(ch_orig * scale))
        resized = crop.resize((nw, nh), Image.LANCZOS)
        x_off   = (MOSAIC_W - nw) // 2
        y_off   = y + LABEL_H + (ch - nh) // 2
        mosaic.paste(resized, (x_off, y_off))

        y += LABEL_H + ch + GAP

    return mosaic


def parse_mosaic_text(ocr_text, labels):
    """
    Parseia o texto OCR do mosaico.
    Chandra retorna HTML table: <tr><td>[LABEL]</td><td>VALOR</td></tr>
    Também suporta formato texto simples como fallback.
    Retorna dict: {label: valor_limpo}
    """
    result = {}

    # ── Formato HTML (Chandra interpreta mosaico como tabela) ──
    if '<tr>' in ocr_text or '<td>' in ocr_text:
        for m in re.finditer(
            r'<tr>\s*<td>\s*\[([^\]]+)\]\s*</td>\s*<td>(.*?)</td>\s*</tr>',
            ocr_text, re.DOTALL | re.IGNORECASE
        ):
            result[m.group(1).strip()] = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if result:
            return result
        # Às vezes sem espaços: <tr><td>[X]</td><td>val</td></tr>
        for m in re.finditer(r'<td>\[([^\]]+)\]</td><td>([^<]*)</td>', ocr_text, re.IGNORECASE):
            result[m.group(1).strip()] = m.group(2).strip()
        if result:
            return result

    # ── Normalizar: OCR confunde dígito 0 com letra O nos labels ──
    # Ex: [STO] → [ST0], [NMO] → [NM0]
    normalized = re.sub(
        r'\[([A-Z]{1,3})([0-9O]{1,2})\]',
        lambda m: '[' + m.group(1) + m.group(2).replace('O', '0').replace('o', '0') + ']',
        ocr_text
    )

    # ── Formato texto simples: [LABEL]\n\nvalor\n\n[PRÓXIMO] ──
    for i, label in enumerate(labels):
        next_label = labels[i + 1] if i + 1 < len(labels) else None
        if next_label:
            pat = r'\[' + re.escape(label) + r'\]\s*(.*?)\s*(?=\[' + re.escape(next_label) + r'\])'
        else:
            pat = r'\[' + re.escape(label) + r'\]\s*(.*?)$'
        m = re.search(pat, normalized, re.DOTALL | re.IGNORECASE)
        result[label] = m.group(1).strip() if m else ''

    return result


def analyze_table_mosaic(img_pil):
    """
    Análise via MOSAICO ROTULADO — uma única chamada OCR.

    Fluxo:
      1. Detecção visual (dealer button, seats ativos) — sem OCR, rápido
      2. Recorta regiões importantes (pot, board, hero cards, stacks)
      3. Monta UM mosaico vertical com labels [POT], [BOARD1], [STACK0]…
      4. UMA chamada Chandra OCR — lê todo o mosaico de uma vez
      5. Parseia texto por label → monta resposta estruturada

    ~3-5s esperado (vs 228s do batch original)
    """
    manager = get_ocr_manager()
    if not manager or not _BatchInputItem:
        return None, {}

    coords   = active_coords or load_coords()
    num_seats   = int(coords.get('seats', 8))
    seat_regions = coords.get('seat_regions', [])
    t0 = time.time()
    timings = {}

    # ── FASE 1: Dealer (visual, zero OCR) ──────────────────────────
    t1 = time.time()
    dealer_regions = [s.get('dealer') for s in seat_regions]
    btn_seat_idx   = detect_dealer_position(img_pil, dealer_regions)
    timings['dealer'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  Dealer visual: idx={btn_seat_idx}")

    # ── FASE 2: Seats ativos (pixel, zero OCR) ─────────────────────
    t1 = time.time()
    seat_active = []
    for i, sr in enumerate(seat_regions):
        active = True
        if _valid_region(sr.get('cards')):
            active = detect_seat_active(img_pil, sr['cards'], is_hero=(i == 0))
        seat_active.append(active)
        print(f"  [{sr.get('id','?')}] active={active}")
    timings['detect'] = f"{(time.time()-t1)*1000:.0f}ms"

    # ── FASE 3: Montar regiões para o mosaico ──────────────────────
    t1 = time.time()
    mosaic_items = []   # (label, PIL crop)
    label_map    = {}   # label → (tipo, seat_idx)

    def add(label, crop, tipo, idx=0):
        mosaic_items.append((label, crop))
        label_map[label] = (tipo, idx)

    def card_corner(img):
        """Recorta só o canto superior da carta onde ficam rank+suit (ex: A♠).
        Evita que Chandra entre em modo 'análise de imagem' (~28s → ~3s)."""
        w, h = img.size
        # Top 45% da altura cobre o símbolo de rank+suit no canto superior esquerdo
        return img.crop((0, 0, w, max(1, int(h * 0.45))))

    # Pot
    if _valid_region(coords.get('pot')):
        add('POT', crop_region(img_pil, coords['pot']), 'pot')

    # Board cards — só o canto superior (rank+suit)
    for i, r in enumerate(coords.get('board', [])):
        if _valid_region(r) and detect_seat_active(img_pil, r, is_hero=True):
            add(f'BD{i+1}', card_corner(crop_region(img_pil, r)), 'board', i)

    # Stacks de todos os seats ativos + hero cards
    for i, sr in enumerate(seat_regions):
        if not seat_active[i]:
            continue
        if _valid_region(sr.get('stack')):
            add(f'ST{i}', crop_region(img_pil, sr['stack']), 'stack', i)
        # Cartas do hero — só canto superior de cada carta
        if i == 0 and _valid_region(sr.get('cards')):
            add('HCC', card_corner(crop_region(img_pil, sr['cards'])), 'cards', 0)
        # Nomes para todos os ativos
        if _valid_region(sr.get('name')):
            add(f'NM{i}', crop_region(img_pil, sr['name']), 'name', i)

    timings['crop'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  Mosaico: {len(mosaic_items)} itens")

    if not mosaic_items:
        print("  ⚠️  Nenhuma região válida para OCR")
        return _empty_result(seat_regions, btn_seat_idx, timings, t0)

    # ── FASE 4: UMA chamada OCR no mosaico inteiro ─────────────────
    t1 = time.time()
    mosaic_img = build_labeled_mosaic(mosaic_items)
    try:
        results    = manager.generate([_BatchInputItem(image=mosaic_img, prompt_type="ocr")])
        ocr_text   = (results[0].markdown or results[0].raw or '').strip()
    except Exception as e:
        print(f"  OCR mosaico falhou: {e}")
        ocr_text = ''
    timings['ocr'] = f"{(time.time()-t1)*1000:.0f}ms"
    print(f"  OCR mosaico ({len(ocr_text)} chars):\n{ocr_text[:400]}")

    # ── FASE 5: Parse por label ────────────────────────────────────
    t1 = time.time()
    labels   = [lbl for lbl, _ in mosaic_items]
    parsed   = parse_mosaic_text(ocr_text, labels)
    print(f"  parsed labels: {list(parsed.keys())}")
    for k, v in parsed.items():
        print(f"    [{k}] = {repr(v[:80])}")

    # Montar table
    pos_map  = POS_MAPS.get(num_seats, POS_MAPS[8])
    table = {
        'pot': 0, 'board': [], 'street': 'preflop',
        'blinds': '', 'btn_seat': btn_seat_idx + 1 if btn_seat_idx >= 0 else -1,
        'num_active': 0, 'last_action': '', 'bet_to_call': 0,
        'tournament': True,
        'ocr_raw': ocr_text,  # debug
        'ocr_parsed': {},    # preenchido abaixo
    }

    if 'POT' in parsed:
        table['pot'] = parse_number(parsed['POT'])

    for lbl, val in parsed.items():
        if lbl.startswith('BD'):
            card = parse_card_text(val)
            if card:
                table['board'].append(card)

    n_board = len(table['board'])
    table['street'] = {0:'preflop', 3:'flop', 4:'turn', 5:'river'}.get(n_board, 'preflop')

    # Montar seats
    seats = []
    for i, sr in enumerate(seat_regions):
        seat_num  = i + 1
        is_hero   = (i == 0)
        phys_id   = sr.get('id',    f'seat_{seat_num}')
        phys_label= sr.get('label', f'Seat {seat_num}')
        active    = seat_active[i]

        name  = parsed.get(f'NM{i}', '').strip() or ('Hero' if is_hero else f'Player{seat_num}')
        stack = parse_number(parsed.get(f'ST{i}', ''))
        # HCC = ambas as cartas do hero em um crop só
        hcc_text = parsed.get('HCC', '') if is_hero else ''
        c1, c2 = parse_two_cards(hcc_text) if is_hero else ('', '')

        # Detectar sit-out
        if 'sitting' in name.lower() or 'sit out' in name.lower():
            active = False

        position = '?'
        if btn_seat_idx >= 0:
            position = pos_map[(num_seats - btn_seat_idx + i) % num_seats]

        seats.append({
            'seat': seat_num, 'id': phys_id, 'label': phys_label,
            'name': name, 'stack': stack, 'cards': [c1, c2],
            'position': position, 'bet': 0, 'bounty': '',
            'is_hero': is_hero, 'active': active, 'action': '',
        })

    active_seats      = [s for s in seats if s['active']]
    table['num_active'] = len(active_seats)
    table['ocr_parsed'] = {k: v[:120] for k, v in parsed.items()}

    # Ordenar por posição
    if btn_seat_idx >= 0:
        for s in seats:
            s['_po'] = pos_map.index(s['position']) if s['position'] in pos_map else 99
        seats.sort(key=lambda s: s['_po'])
        for s in seats: s.pop('_po', None)

    timings['parse'] = f"{(time.time()-t1)*1000:.0f}ms"
    elapsed = time.time() - t0
    print(f"\n✅ Mosaico análise em {elapsed:.1f}s | Board: {table['board']} | Pot: {table['pot']}")
    for s in seats:
        if s['active']:
            print(f"  [{s['id']:13s}] {s['position']:6s} {s['name']:15s} stack={s['stack']} cards={s['cards']}")
    return {'table': table, 'seats': seats, 'timings': timings}, elapsed


def _empty_result(seat_regions, btn_seat_idx, timings, t0):
    seats = [{'seat': i+1, 'id': sr.get('id',''), 'label': sr.get('label',''),
               'name': '', 'stack': 0, 'cards': ['',''], 'position': '?',
               'bet': 0, 'bounty': '', 'is_hero': i==0, 'active': False, 'action': ''}
             for i, sr in enumerate(seat_regions)]
    table = {'pot': 0, 'board': [], 'street': 'preflop', 'blinds': '',
             'btn_seat': btn_seat_idx+1 if btn_seat_idx >= 0 else -1,
             'num_active': 0, 'last_action': '', 'bet_to_call': 0, 'tournament': True}
    return {'table': table, 'seats': seats, 'timings': timings}, time.time() - t0


def analyze_table(img_pil):
    """
    Análise completa da mesa — BATCH OCR (fallback).
    Coleta TODAS as regiões primeiro, faz UMA chamada batch ao Chandra,
    depois parseia os resultados.
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
        if _valid_region(sr.get('cards')):
            is_hero = (i == 0)
            is_active = detect_seat_active(img_pil, sr['cards'], is_hero=is_hero)
        seat_active.append(is_active)
        phys_id = sr.get('id', f'seat_{i+1}')
        print(f"  [{phys_id}] active: {is_active}")
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

    # 3c. Board cards (só se região válida e tem conteúdo visível)
    board_indices = []
    for i, card_region in enumerate(coords.get('board', [])):
        if _valid_region(card_region) and detect_seat_active(img_pil, card_region, is_hero=True):
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

        # Cartas: apenas Hero (villain cards são verso fechado, OCR inútil)
        if (i == 0) and _valid_region(sr.get('cards')):
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
    if max_bet > 0 and math.isfinite(max_bet):
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
        'ocr_engine': 'EasyOCR' if _easyocr_reader else ('Chandra OCR 2' if _ocr_manager else 'none'),
        'gpu': gpu_name,
        'preset': coords.get('name', 'custom'),
        'seats': coords.get('seats', 8),
        'seat_ids': seat_ids,
        'ocr_ready': _easyocr_reader is not None or _ocr_manager is not None,
        'easyocr_ready': _easyocr_reader is not None,
        'chandra_ready': _ocr_manager is not None,
    })


@app.route('/warmup', methods=['POST'])
def warmup():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    t0 = time.time()
    get_easyocr_reader()   # carrega EasyOCR (engine principal)
    get_ocr_manager()      # carrega Chandra (fallback opcional)
    return jsonify({
        'status': 'ready',
        'elapsed': f"{time.time()-t0:.1f}s",
        'easyocr': _easyocr_reader is not None,
        'chandra': _ocr_manager is not None,
    })


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

    # Fast mode (EasyOCR + pixel): padrão
    # use_chandra=true → mosaico Chandra (mais lento mas mais robusto)
    # use_batch=true   → batch Chandra antigo (fallback)
    use_chandra = data.get('use_chandra', False)
    use_batch   = data.get('use_batch',   False)
    if use_batch:
        analysis, elapsed = analyze_table(img_pil)
    elif use_chandra:
        analysis, elapsed = analyze_table_mosaic(img_pil)
    else:
        analysis, elapsed = analyze_table_fast(img_pil)

    # Override dealer se frontend enviou manualmente
    if 'btn_seat_idx' in data and data['btn_seat_idx'] >= 0:
        coords_data = active_coords or load_coords()
        num_seats = int(coords_data.get('seats', 8))
        btn_idx = int(data['btn_seat_idx'])
        pos_map = POS_MAPS.get(num_seats, POS_MAPS[8])
        analysis['table']['btn_seat'] = btn_idx + 1
        for s in analysis['seats']:
            i = s['seat'] - 1
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
