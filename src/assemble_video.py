"""台本JSON + キャラクター素材 + VOICEVOX音声から動画を組み立てる。

使い方:
    python src/assemble_video.py scripts_templates/scripts/20260710_xxx.json
"""

import json
import math
import random
import re
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from moviepy import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx import AudioLoop, MultiplyVolume
from moviepy.video.fx import CrossFadeIn
from scipy.signal import butter, lfilter
from scipy.ndimage import label

from tts import synthesize, engine_is_alive
from description import build_description
from pattern_genes import CHARACTER_GENES, resolve_value
from ai_provider import GeminiProvider
import kobun_voice_config as kvc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = PROJECT_ROOT / "materials" / "characters"
FONTS_DIR = PROJECT_ROOT / "materials" / "fonts"
BGM_DIR = PROJECT_ROOT / "materials" / "bgm"
SE_DIR = PROJECT_ROOT / "materials" / "se"
OUTPUT_DIR = PROJECT_ROOT / "output"

BGM_TRACKS = {
    "normal": BGM_DIR / "battle_debate_loop.mp3",  # ネオロック調のアップテンポ版に変更（没入感重視）
    "climax": BGM_DIR / "tense_climax_loop.mp3",
}
BGM_VOLUME = 0.07  # セリフの聞き取りやすさを優先して控えめに
SE_FILES = {
    "attack": SE_DIR / "attack.mp3",
    "logic_ding": SE_DIR / "logic_ding.mp3",
    "stumble": SE_DIR / "stumble.mp3",
    "chaos": SE_DIR / "chaos.mp3",
    "win_fanfare": SE_DIR / "win_fanfare.mp3",
    "lose_buzzer": SE_DIR / "lose_buzzer.mp3",
}
# 効果音ごとの音量（全体的に控えめに。素材によって元音量差が大きいため個別調整）
SE_VOLUME = {
    "attack": 0.25,
    "logic_ding": 0.22,
    "stumble": 0.22,
    "chaos": 0.3,
    "win_fanfare": 0.25,
    "lose_buzzer": 0.2,
}
# セリフの頭で鳴らすか、言い終わりに合わせて鳴らすか
SE_TIMING_AT_END = {"logic_ding", "stumble", "win_fanfare", "lose_buzzer"}

# 素材フォルダ/ファイル名は「トウマ」が実際には Len 表記（Ren ではない）になっているため、それに合わせる
CHARACTER_FOLDER = {"ren": "Len", "mailo": "Mailo", "noa": "Noa", "baku": "Baku", "dady": "Dady"}
FILE_PREFIX = {"ren": "len", "mailo": "mailo", "noa": "noa", "baku": "baku", "dady": "dady"}

# pattern_genes.py（編集パターン遺伝子の土台）で管理しているキャラ別アクセントカラーを
# ここから参照する。値そのものの変更は pattern_genes.py 側で行うこと。
CHARACTER_COLOR = {char: resolve_value(genes.accent_color) for char, genes in CHARACTER_GENES.items()}
CHARACTER_LABEL = {"ren": "トウマ", "mailo": "ユズ", "noa": "ソラ", "baku": "カイ", "dady": "覆面・ダディ"}

CANVAS = {
    "shorts": (1080, 1920),
    "long": (1920, 1080),
    "compilation": (1920, 1080),
}

POP_FONT_PATH = FONTS_DIR / "RoundedMplus1c-Black.ttf"
EMOTION_FONT_PATH = FONTS_DIR / "YuseiMagic-Regular.ttf"
# 感情の起伏が強いse＝ポップな通常字幕ではなく専用フォントで表現を変える
_EMOTIONAL_SE = {"attack", "stumble", "chaos", "win_fanfare", "lose_buzzer"}

OUTLINE_WHITE_FRAC = 0.010  # 白縁の太さ（スプライト高さに対する比率）
OUTLINE_BLACK_FRAC = 0.010  # 黒縁の太さ（白縁の外側にさらに乗せる）
ANIM_T = 0.28  # キャラのスライドイン/アウトにかける秒数
CROSSFADE_T = 0.22  # 発言/非発言切り替え時のクロスフェード秒数

PAIR_SCALE = 0.62
CUTIN_SCALE = PAIR_SCALE  # 討論外キャラも通常サイズのまま。端で3〜4割見切れさせる
CUTIN_OFFSCREEN_FRAC = 0.35  # カットイン時に画面外へ食い込ませる割合
# 討論外キャラの出現コーナー（固定）。指定のないキャラ(ダディ等)は行番号で左右を振り分ける。
CUTIN_SIDE = {"noa": "left", "baku": "right"}

_sprite_cache: dict[str, Image.Image] = {}
_variant_cache: dict[tuple, Path] = {}


def sprite_path(character: str, emotion: str) -> Path:
    folder = CHARACTERS_DIR / CHARACTER_FOLDER[character]
    prefix = FILE_PREFIX[character]
    path = folder / f"{prefix}_{emotion}.png"
    if path.exists():
        return path
    return folder / f"{prefix}_base.png"


def _dilate_alpha(alpha: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return alpha
    return alpha.filter(ImageFilter.MaxFilter(px * 2 + 1))


def outlined_sprite(path: Path) -> Image.Image:
    """切り抜きの粗さを隠す、白→黒の二重縁取りを付けたスプライトを返す（キャッシュ付き）。"""
    key = str(path)
    if key in _sprite_cache:
        return _sprite_cache[key]

    img = Image.open(path).convert("RGBA")
    alpha = img.split()[3]
    white_px = max(2, int(img.height * OUTLINE_WHITE_FRAC))
    black_px = max(2, int(img.height * OUTLINE_BLACK_FRAC))

    black_alpha = _dilate_alpha(alpha, white_px + black_px)
    white_alpha = _dilate_alpha(alpha, white_px)

    black_layer = Image.new("RGBA", img.size, (0, 0, 0, 255))
    black_layer.putalpha(black_alpha)
    white_layer = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white_layer.putalpha(white_alpha)

    result = Image.alpha_composite(black_layer, white_layer)
    result = Image.alpha_composite(result, img)
    _sprite_cache[key] = result
    return result


def sprite_variant_path(
    character: str, emotion: str, active: bool, scale_ratio: float, canvas_h: int, tmp_dir: Path, mirror: bool = False
) -> tuple[Path, int, int]:
    """アウトライン付与・拡縮・非発言時の減光（・必要なら左右反転）を行ったスプライトをPNGとして保存し、パスとサイズを返す（キャッシュ付き）。"""
    key = (character, emotion, active, scale_ratio, canvas_h, mirror)
    if key in _variant_cache:
        cached = _variant_cache[key]
        with Image.open(cached) as im:
            return cached, im.width, im.height

    base = outlined_sprite(sprite_path(character, emotion))
    # 発言/非発言でサイズを変えるとクロスフェード中に大小2つのシルエットが重なって見えるため、
    # サイズは統一し、明るさと前後関係(z-order)だけで発言強調する。
    scale = (canvas_h * scale_ratio) / base.height
    w = max(1, int(base.width * scale))
    h = max(1, int(base.height * scale))
    img = base.resize((w, h), Image.LANCZOS)

    if not active:
        # オーバーレイのアルファをキャラ自身のアルファでマスクし、透明部分（余白）が
        # 暗い長方形として見えてしまうのを防ぐ
        char_alpha = img.split()[3]
        dim_alpha = char_alpha.point(lambda a: a * 55 // 255)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 255))
        overlay.putalpha(dim_alpha)
        img = Image.alpha_composite(img, overlay)

    if mirror:
        img = ImageOps.mirror(img)

    out_path = tmp_dir / f"sprite_{character}_{emotion}_{active}_{canvas_h}_{mirror}.png"
    img.save(out_path)
    _variant_cache[key] = out_path
    return out_path, w, h


def make_gradient_background(size, top=(210, 214, 222), bottom=(232, 234, 238)) -> Image.Image:
    # フォールバック用の単色グラデーション背景。ホラー系以外は明るめにする。
    w, h = size
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=color)
    return img


OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
FACE_MODEL_PATH = PROJECT_ROOT / "materials" / "models" / "face_detection_yunet.onnx"

_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None and FACE_MODEL_PATH.exists():
        import cv2

        _face_detector = cv2.FaceDetectorYN_create(str(FACE_MODEL_PATH), "", (320, 320))
    return _face_detector


def contains_face(img: Image.Image) -> bool:
    """肖像権リスクを避けるため、人物の顔が写り込んでいる画像を検出する。"""
    detector = _get_face_detector()
    if detector is None:
        return False
    import cv2
    import numpy as np

    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(arr)
    return faces is not None and len(faces) > 0


# 商標を示唆するタイトルパターン（完全ではないが簡易フィルタとして機能する）。
# 例: "Portable-Air-Conditioner_DeLonghi-PAC-T100" のような「単語-型番風」表記や、
# 家電・電子機器・食品分野でよく見る大手ブランド名を含むタイトルは避ける。
_BRAND_MODEL_PATTERN = re.compile(r"[A-Za-z][a-zA-Z]{2,}-[A-Za-z0-9]{2,}")
_KNOWN_BRANDS = [
    "samsung", "sony", "panasonic", "lg ", "sharp", "toshiba", "apple", "iphone",
    "nike", "adidas", "coca-cola", "coca cola", "pepsi", "nintendo", "canon", "nikon",
    "toyota", "honda", "nissan", "delonghi", "dyson", "philips", "braun", "zojirushi",
    "mitsubishi", "hitachi", "daikin", "xerox", "google", "microsoft", "amazon",
]


def _looks_branded(title: str) -> bool:
    """タイトルから商標・ブランドロゴが写り込んでいそうな写真を簡易的に推定する（完全ではない）。"""
    if not title:
        return False
    lowered = title.lower()
    if any(b in lowered for b in _KNOWN_BRANDS):
        return True
    return bool(_BRAND_MODEL_PATTERN.search(title))


def fetch_background_image(query: str):
    """Openverseからテーマに関連する、人物の顔・ブランドが写っていなさそうな背景写真を1枚取得する。失敗時は(None, None)。"""
    if not query:
        return None, None
    try:
        resp = requests.get(
            OPENVERSE_ENDPOINT,
            params={"q": query, "license": "by,cc0", "mature": "false", "page_size": 20},
            headers={"User-Agent": "ai-video-pipeline/0.1 (background image fetch)"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None, None

    for r in results:
        url = r.get("url")
        if not url or (r.get("width") or 0) < 600 or (r.get("height") or 0) < 600:
            continue
        if _looks_branded(r.get("title") or ""):
            continue  # 商標・ブランドロゴが写り込んでいそうなためスキップ
        try:
            img_resp = requests.get(url, timeout=10, headers={"User-Agent": "ai-video-pipeline/0.1"})
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        except Exception:
            continue

        try:
            if contains_face(img):
                continue  # 肖像権リスク回避のためスキップし、次の候補を試す
        except Exception:
            pass  # 顔検出自体に失敗した場合は安全側に倒して採用する

        credit = {
            "title": r.get("title"),
            "creator": r.get("creator"),
            "license": r.get("license"),
            "source": r.get("foreign_landing_url"),
        }
        return img, credit

    return None, None


_HORROR_KEYWORDS = [
    "ホラー", "怖い", "幽霊", "お化け", "恐怖", "呪い", "心霊",
    "horror", "ghost", "scary", "haunted", "creepy",
]


def is_horror_theme(*texts: str) -> bool:
    joined = " ".join(t.lower() for t in texts if t)
    return any(kw.lower() in joined for kw in _HORROR_KEYWORDS)


def process_background(img: Image.Image, size, dark: bool = False) -> Image.Image:
    """キャンバスにcover方式で合わせ、キャラ・字幕の視認性のためぼかし+色調整する。
    ホラー系テーマ以外は明るめのトーンにする。"""
    w, h = size
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - w) // 2, (new_h - h) // 2
    img = img.crop((left, top, left + w, top + h))
    img = img.filter(ImageFilter.GaussianBlur(6))
    if dark:
        overlay = Image.new("RGB", size, (18, 20, 26))
        return Image.blend(img, overlay, 0.45)
    overlay = Image.new("RGB", size, (235, 235, 232))
    return Image.blend(img, overlay, 0.25)


# 主張の補強画像は、商標・肖像権リスクのある実写真の取得をやめ、自前描画のピクトグラムに限定する。
# キーは generate_script.py の SCRIPT_SCHEMA / プロンプトで使うenumと一致させること。
VISUAL_AID_LABELS = {
    "thermometer": "温度",
    "chart_up": "上昇傾向",
    "chart_down": "下降傾向",
    "money": "コスト",
    "clock": "時間",
    "warning": "注意",
    "eco": "エコ",
    "house": "暮らし",
    "phone": "デジタル",
    "food": "食",
    "sleep": "睡眠",
}


def _draw_visual_aid_pictogram(draw: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, r: float, color) -> None:
    lw = max(2, int(r * 0.12))
    if kind == "thermometer":
        tube_w = r * 0.34
        draw.rounded_rectangle([cx - tube_w / 2, cy - r, cx + tube_w / 2, cy + r * 0.5], radius=tube_w / 2, outline=color, width=lw)
        draw.ellipse([cx - tube_w, cy + r * 0.1, cx + tube_w, cy + r * 1.1], fill=color)
        draw.line([cx, cy - r * 0.6, cx, cy + r * 0.3], fill=color, width=max(2, lw - 2))
    elif kind in ("chart_up", "chart_down"):
        heights = [0.35, 0.6, 0.9] if kind == "chart_up" else [0.9, 0.6, 0.35]
        bar_w = r * 0.4
        xs = [cx - r * 0.8, cx, cx + r * 0.8]
        for x, h in zip(xs, heights):
            bh = r * 1.4 * h
            draw.rectangle([x - bar_w / 2, cy + r * 0.7 - bh, x + bar_w / 2, cy + r * 0.7], fill=color)
        ay = cy - r * 0.9 if kind == "chart_up" else cy + r * 0.5
        draw.polygon([(cx + r * 0.9, ay), (cx + r * 1.3, ay + (r * 0.35 if kind == "chart_up" else -r * 0.35)), (cx + r * 0.5, ay + (r * 0.35 if kind == "chart_up" else -r * 0.35))], fill=color)
    elif kind == "money":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=lw)
        f = load_font(POP_FONT_PATH, int(r * 1.1))
        draw.text((cx, cy), "¥", font=f, fill=color, anchor="mm")
    elif kind == "clock":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=lw)
        draw.line([cx, cy, cx, cy - r * 0.6], fill=color, width=lw)
        draw.line([cx, cy, cx + r * 0.45, cy + r * 0.1], fill=color, width=lw)
    elif kind == "warning":
        draw.polygon([(cx, cy - r), (cx + r * 0.95, cy + r * 0.8), (cx - r * 0.95, cy + r * 0.8)], outline=color, width=lw)
        f = load_font(POP_FONT_PATH, int(r * 0.9))
        draw.text((cx, cy + r * 0.25), "!", font=f, fill=color, anchor="mm")
    elif kind == "eco":
        # 元は扇形+斜め線でメーター(速度計)風に見えてしまっていたため、葉っぱの形に描き直した
        draw.ellipse([cx - r * 0.62, cy - r, cx + r * 0.62, cy + r * 0.75], fill=color)
        draw.line([cx, cy + r * 0.75, cx, cy + r * 1.05], fill=color, width=lw)  # 葉柄
        draw.line([cx, cy - r * 0.6, cx, cy + r * 0.6], fill=(255, 255, 255, 255), width=max(2, lw - 2))  # 葉脈
    elif kind == "house":
        # 元は塗りつぶしの三角+矩形で単なる上向き矢印に見えてしまっていたため、
        # 輪郭線+ドアで「家」とわかる形に描き直した
        draw.polygon([(cx, cy - r), (cx + r * 0.95, cy + r * 0.1), (cx - r * 0.95, cy + r * 0.1)], outline=color, width=lw)
        draw.rectangle([cx - r * 0.62, cy + r * 0.08, cx + r * 0.62, cy + r], outline=color, width=lw)
        draw.rectangle([cx - r * 0.16, cy + r * 0.42, cx + r * 0.16, cy + r], fill=color)  # ドア
    elif kind == "phone":
        draw.rounded_rectangle([cx - r * 0.55, cy - r, cx + r * 0.55, cy + r], radius=r * 0.18, outline=color, width=lw)
        draw.ellipse([cx - r * 0.08, cy + r * 0.75, cx + r * 0.08, cy + r * 0.9], fill=color)
    elif kind == "food":
        # 丼(どんぶり)に湯気: 単なる楕円+弧では「食べ物」と伝わりにくかったため、
        # 器のシルエット+湯気で明確に「温かい料理」とわかるようにした
        rim_y = cy - r * 0.05
        draw.arc([cx - r, rim_y - r * 0.55, cx + r, cy + r * 1.05], start=8, end=172, fill=color, width=lw)
        draw.ellipse([cx - r, rim_y - r * 0.22, cx + r, rim_y + r * 0.14], outline=color, width=lw)
        for dx in (-r * 0.42, r * 0.05, r * 0.5):
            x0, y0 = cx + dx, rim_y - r * 0.35
            draw.line(
                [(x0, y0), (x0 - r * 0.12, y0 - r * 0.28), (x0 + r * 0.12, y0 - r * 0.56)],
                fill=color, width=max(2, lw - 2), joint="curve",
            )
    elif kind == "sleep":
        f = load_font(POP_FONT_PATH, int(r * 0.7))
        draw.text((cx, cy - r * 0.3), "Z", font=f, fill=color, anchor="mm")
        draw.text((cx + r * 0.55, cy + r * 0.15), "z", font=load_font(POP_FONT_PATH, int(r * 0.45)), fill=color, anchor="mm")
        draw.text((cx + r * 0.95, cy + r * 0.5), "z", font=load_font(POP_FONT_PATH, int(r * 0.3)), fill=color, anchor="mm")
    else:
        f = load_font(POP_FONT_PATH, int(r * 1.1))
        draw.text((cx, cy), "？", font=f, fill=color, anchor="mm")


def render_visual_aid_card(kind: str, box_size: int) -> Image.Image:
    """主張を補強するインセットカード。実写真ではなく自前描画のピクトグラム（商標・肖像権リスクなし）。"""
    radius = int(box_size * 0.12)
    card = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card, "RGBA")
    draw.rounded_rectangle([0, 0, box_size, box_size], radius=radius, fill=(250, 250, 248, 235))
    draw.rounded_rectangle(
        [2, 2, box_size - 2, box_size - 2], radius=radius, outline=(230, 200, 90, 255), width=max(3, box_size // 35)
    )

    accent = (60, 66, 58, 255)
    _draw_visual_aid_pictogram(draw, kind, box_size * 0.5, box_size * 0.42, box_size * 0.24, accent)

    label = VISUAL_AID_LABELS.get(kind, "")
    if label:
        f = load_font(POP_FONT_PATH, int(box_size * 0.13))
        draw.text((box_size * 0.5, box_size * 0.8), label, font=f, fill=(50, 54, 48, 255), anchor="mm")

    return card


def make_background(size, bg_query: str = "", theme: str = "") -> tuple[Image.Image, dict | None]:
    dark = is_horror_theme(bg_query, theme)
    photo, credit = fetch_background_image(bg_query)
    if photo is not None:
        return process_background(photo, size, dark), credit
    top, bottom = ((10, 8, 12), (30, 26, 34)) if dark else ((210, 214, 222), (232, 234, 238))
    return make_gradient_background(size, top, bottom), None


def load_font(path: Path, size: int):
    return ImageFont.truetype(str(path), size)


_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9&']+")
# 行頭に来ると読みにくい文字（句読点・閉じ括弧・小書き文字等）と、行末に来ると読みにくい
# 文字（開き括弧）。禁則処理で前後の行に送る対象。
_NO_LINE_START = "、。，．・：；？！ー…」』）】〉》〕〗〙〟ゝゞぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
_NO_LINE_END = "「『（【〈《〔〖〘〝"


def _tokenize_for_wrap(text: str) -> list[str]:
    """英数字の連続は1トークンにまとめ、それ以外（日本語）は1文字ずつのトークンにする。
    日本語には分かち書き（単語間スペース）が無いため文字単位が基本になるが、英単語や
    "Q&A" のような記号混じりの語だけは途中で割れないようにする。"""
    tokens = []
    i = 0
    while i < len(text):
        m = _ASCII_WORD_RE.match(text, i)
        if m:
            tokens.append(m.group())
            i = m.end()
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def wrap_text(text: str, font, max_width: float, draw: ImageDraw.ImageDraw) -> list[str]:
    """幅ベースで折り返す。英単語・記号混じりの語は途中で割らず、日本語は禁則処理
    （行頭に句読点・閉じ括弧・小書き文字、行末に開き括弧が来ないようにする）を行う。"""
    lines, cur = [], ""
    for tok in _tokenize_for_wrap(text):
        trial = cur + tok
        if cur and draw.textlength(trial, font=font) > max_width:
            lines.append(cur)
            cur = tok
        else:
            cur = trial
    if cur:
        lines.append(cur)

    for i in range(1, len(lines)):
        while lines[i] and lines[i][0] in _NO_LINE_START:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    for i in range(len(lines) - 1):
        while lines[i] and lines[i][-1] in _NO_LINE_END:
            lines[i + 1] = lines[i][-1] + lines[i + 1]
            lines[i] = lines[i][:-1]
    return [line for line in lines if line]


def render_title_card(theme: str, size, positions: list[dict] | None = None) -> Image.Image:
    """オープニングの「本日のテーマ」表示。通常の字幕ログとは別の専用レイアウト。
    positions指定時は、テーマの下に「誰がどちら派か」の簡易テロップ(VS表示)を追加する。"""
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    eyebrow_font = load_font(POP_FONT_PATH, int(h * 0.026))
    theme_font = load_font(POP_FONT_PATH, int(h * 0.048))
    max_width = w * 0.84
    outline_w = max(1, int(h * 0.0035))

    theme_lines = wrap_text(theme, theme_font, max_width, draw)
    line_height = int(h * 0.056)

    stance_font = load_font(POP_FONT_PATH, int(h * 0.024))
    vs_font = load_font(POP_FONT_PATH, int(h * 0.022))
    # 縦積み(立場ラベル/VS/立場ラベル)で組むため、フォーマット(shorts=縦長/long=横長)によらず
    # 横幅からのはみ出しを気にしなくてよい。
    stance_block_h = int(h * 0.028) * 2 + int(h * 0.036) if positions else 0

    block_height = int(h * 0.05) + line_height * len(theme_lines) + stance_block_h + int(h * 0.05)

    box_top = int(h * 0.05)
    box_bottom = box_top + block_height
    draw.rounded_rectangle([w * 0.05, box_top, w * 0.95, box_bottom], radius=int(h * 0.02), fill=(12, 14, 18, 210))

    eyebrow_text = "本日のテーマ"
    eb_w = draw.textlength(eyebrow_text, font=eyebrow_font)
    draw.text(
        ((w - eb_w) / 2, box_top + int(h * 0.02)), eyebrow_text, font=eyebrow_font, fill=(239, 166, 198),
        stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
    )

    y = box_top + int(h * 0.02) + int(h * 0.04)
    for tline in theme_lines:
        tw = draw.textlength(tline, font=theme_font)
        draw.text(
            ((w - tw) / 2, y), tline, font=theme_font, fill=(255, 255, 255, 255),
            stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
        )
        y += line_height

    if positions:
        left, right = positions[0], positions[1]
        left_color = CHARACTER_COLOR.get(left["character"], (255, 255, 255))
        right_color = CHARACTER_COLOR.get(right["character"], (255, 255, 255))
        left_text = f"{CHARACTER_LABEL.get(left['character'], '')}: {left['stance']}"
        right_text = f"{CHARACTER_LABEL.get(right['character'], '')}: {right['stance']}"
        cx = w / 2

        # 横幅にかかわらず必ず収まるよう、立場ラベル/VS/立場ラベルを縦に積む
        stance_max_width = w * 0.86
        left_lines = wrap_text(left_text, stance_font, stance_max_width, draw) or [left_text]
        right_lines = wrap_text(right_text, stance_font, stance_max_width, draw) or [right_text]
        stance_line_h = int(h * 0.028)

        yy = y + int(h * 0.01)
        for tline in left_lines:
            tw = draw.textlength(tline, font=stance_font)
            draw.text(
                (cx - tw / 2, yy), tline, font=stance_font, fill=left_color,
                stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
            )
            yy += stance_line_h

        vs_w = draw.textlength("VS", font=vs_font)
        draw.text(
            (cx - vs_w / 2, yy + int(h * 0.004)), "VS", font=vs_font, fill=(239, 166, 198),
            stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
        )
        yy += int(h * 0.036)

        for tline in right_lines:
            tw = draw.textlength(tline, font=stance_font)
            draw.text(
                (cx - tw / 2, yy), tline, font=stance_font, fill=right_color,
                stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
            )
            yy += stance_line_h

    return img


_INSIGHT_SE = {"logic_ding"}
_GLOOM_SE = {"lose_buzzer"}
_GLOOM_EMOTIONS = {"tearful", "disappointment", "3d_despair"}
_CONFUSION_EMOTIONS = {"puzzled", "guy_puzzled", "muscle_puzzled", "suspicion", "absentmindedness"}


def effect_kind_for_line(line: dict) -> str | None:
    """視聴維持を意識した軽量エフェクト（閃き/どんより/困惑）の種類を判定する。"""
    se = line.get("se", "none")
    emotion = line.get("emotion", "")
    if se in _INSIGHT_SE:
        return "insight"
    if se in _GLOOM_SE or emotion in _GLOOM_EMOTIONS:
        return "gloom"
    if emotion in _CONFUSION_EMOTIONS:
        return "confusion"
    return None


def render_effect_icon(kind: str, size_px: int) -> Image.Image:
    """閃き(電球)・どんより(雨雲+汗)・困惑(？マーク)の軽量アイコンを描画する。"""
    img = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size_px * 0.5, size_px * 0.46

    if kind == "insight":
        r = size_px * 0.22
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            x1, y1 = cx + math.cos(rad) * r * 1.25, cy + math.sin(rad) * r * 1.25
            x2, y2 = cx + math.cos(rad) * r * 1.7, cy + math.sin(rad) * r * 1.7
            draw.line([x1, y1, x2, y2], fill=(255, 210, 60, 255), width=max(2, size_px // 26))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 226, 110, 255), outline=(90, 70, 10, 255), width=max(2, size_px // 30))
        draw.rectangle([cx - r * 0.35, cy + r * 0.85, cx + r * 0.35, cy + r * 1.25], fill=(110, 110, 110, 255))
    elif kind == "gloom":
        rw, rh = size_px * 0.30, size_px * 0.17
        for dx, dy, s in [(0, 0, 1.0), (-rw * 0.7, -rh * 0.6, 0.7), (rw * 0.55, -rh * 0.5, 0.75)]:
            draw.ellipse(
                [cx + dx - rw * s, cy + dy - rh * s, cx + dx + rw * s, cy + dy + rh * s],
                fill=(158, 168, 182, 235),
            )
        dcx, dcy, dr = cx + rw * 0.95, cy + rh * 1.7, size_px * 0.065
        draw.polygon(
            [(dcx, dcy - dr * 1.7), (dcx - dr, dcy + dr * 0.5), (dcx + dr, dcy + dr * 0.5)],
            fill=(120, 185, 232, 255),
        )
        draw.ellipse([dcx - dr, dcy - dr * 0.3, dcx + dr, dcy + dr * 1.4], fill=(120, 185, 232, 255))
    elif kind == "confusion":
        f1 = load_font(POP_FONT_PATH, int(size_px * 0.5))
        f2 = load_font(POP_FONT_PATH, int(size_px * 0.34))
        draw.text((size_px * 0.10, size_px * 0.06), "？", font=f1, fill=(95, 95, 108, 255))
        draw.text((size_px * 0.52, size_px * 0.40), "？", font=f2, fill=(95, 95, 108, 210))

    return img


def render_subtitle_layer(line: dict, size, color_overrides: dict | None = None) -> Image.Image:
    """Shorts向け: 画面上部に、折り返し対応・ポップなフォントで字幕を描画する透過レイヤー。
    color_overrides指定時は、該当キャラクターの名前ラベル色をCHARACTER_COLORの既定値より
    優先する（pattern_pool.pyのバンディット選定結果を動画単位で反映するための差し替え口）。"""
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    font_size = int(h * 0.034)
    font_path = EMOTION_FONT_PATH if line.get("se") in _EMOTIONAL_SE else POP_FONT_PATH
    font = load_font(font_path, font_size)
    name_font = load_font(POP_FONT_PATH, int(h * 0.022))

    max_width = w * 0.82
    wrapped = wrap_text(line["text"], font, max_width, draw)

    line_height = int(font_size * 1.3)
    top_margin = int(h * 0.06)
    name_h = int(h * 0.035)
    block_height = name_h + line_height * len(wrapped) + int(h * 0.03)

    box_left, box_right = w * 0.06, w * 0.94
    box_top = top_margin
    box_bottom = box_top + block_height
    draw.rounded_rectangle([box_left, box_top, box_right, box_bottom], radius=int(h * 0.018), fill=(12, 14, 18, 195))

    # 視認性向上のため、名前ラベル・セリフ本文ともに黒縁取りを付ける
    # （背景ボックスが半透明で背景が透けることがあり、特にEMOTION_FONT_PATHは線が細く縁取りが効く）
    outline_w = max(1, int(h * 0.0035))

    overrides = color_overrides or {}
    color = tuple(overrides.get(line["character"], CHARACTER_COLOR.get(line["character"], (255, 255, 255))))
    draw.text(
        (box_left + w * 0.03, box_top + int(h * 0.012)), CHARACTER_LABEL.get(line["character"], ""),
        font=name_font, fill=color, stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
    )

    y = box_top + name_h + int(h * 0.01)
    for wline in wrapped:
        draw.text(
            (box_left + w * 0.03, y), wline, font=font, fill=(255, 255, 255, 255),
            stroke_width=outline_w, stroke_fill=(0, 0, 0, 255),
        )
        y += line_height

    return img


def determine_main_pair(lines):
    """トウマ・ユズが基本だが、ソラ/カイが当事者の回にも対応できるよう出現数で決める。"""
    counts = {}
    for line in lines:
        c = line["character"]
        if c == "dady":
            continue
        counts[c] = counts.get(c, 0) + 1
    ordered = sorted(counts, key=lambda k: -counts[k])
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return (ordered + ["ren", "mailo"])[:2]


def shot_chars(line: dict, main_pair) -> list[str]:
    """討論者2名は常時表示。討論に参加しないキャラの発言時は、コーナーカットインとして追加する。"""
    speaker = line["character"]
    chars = list(main_pair)
    if speaker not in chars:
        chars.append(speaker)
    return chars


def cutin_side(character: str, line_index: int) -> str:
    if character in CUTIN_SIDE:
        return CUTIN_SIDE[character]
    return "left" if line_index % 2 == 0 else "right"


def rest_and_edge_x(who: str, main_pair, w: int, sprite_w: int, line_index: int) -> tuple[int, int, str | None]:
    """restは静止位置のx座標、edgeはスライドイン/アウトの起点・終点のx座標。3つ目はカットインの場合のside。"""
    if who in main_pair:
        if who == main_pair[0]:
            return int(w * 0.27 - sprite_w / 2), -sprite_w, None
        return int(w * 0.73 - sprite_w / 2), w, None

    side = cutin_side(who, line_index)
    if side == "left":
        rest_x = -int(sprite_w * CUTIN_OFFSCREEN_FRAC)
        return rest_x, -sprite_w, side
    rest_x = w - int(sprite_w * (1 - CUTIN_OFFSCREEN_FRAC))
    return rest_x, w, side


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_in(p: float) -> float:
    return p ** 3


def build_position_fn(rest_x: int, edge_x: int, entering: bool, exiting: bool, dur: float, y: int):
    anim_t = min(ANIM_T, dur * 0.35) if (entering or exiting) else 0

    def pos(t):
        x = rest_x
        if entering and anim_t > 0 and t < anim_t:
            frac = ease_out(t / anim_t)
            x = edge_x + (rest_x - edge_x) * frac
        elif exiting and anim_t > 0 and t > dur - anim_t:
            frac = ease_in((t - (dur - anim_t)) / anim_t)
            x = rest_x + (edge_x - rest_x) * frac
        return (x, y)

    return pos


def assemble(script_path: Path) -> Path:
    if not engine_is_alive():
        raise RuntimeError(
            "VOICEVOXエンジンに接続できません。tools/windows-cpu/run.exe が起動しているか確認してください。"
        )

    data = json.loads(script_path.read_text(encoding="utf-8"))
    size = CANVAS.get(data.get("format", "shorts"), CANVAS["shorts"])
    w, h = size
    lines = data["lines"]

    # 編集パターンのバンディット選定結果（pattern_pool.py）を反映する差し替え色。
    # generate_script.pyが台本生成時に選定し data["selected_patterns"] に保存している。
    subtitle_color_overrides = {
        char: sel["genes"]["accent_color"]
        for char, sel in data.get("selected_patterns", {}).items()
        if "accent_color" in sel.get("genes", {})
    }
    main_pair = determine_main_pair(lines)
    shots = [shot_chars(l, main_pair) for l in lines]

    # オープニングの主張テロップ用: debate_positionsを主要2名(main_pair)の順序に合わせて並べる。
    # 旧台本（debate_positionsが無い）やキャラ不一致の場合は表示しない（安全に握りつぶす）。
    positions_by_char = {p["character"]: p for p in data.get("debate_positions", [])}
    title_positions = [positions_by_char[c] for c in main_pair if c in positions_by_char]
    if len(title_positions) != 2:
        title_positions = None

    tmp_dir = Path(tempfile.mkdtemp(prefix="ytsys_"))
    bg_path = tmp_dir / "bg.png"
    bg_image, bg_credit = make_background(size, data.get("bg_query", ""), data.get("theme", ""))
    bg_image.save(bg_path)
    if bg_credit:
        print(
            f"[背景クレジット] \"{bg_credit['title']}\" by {bg_credit['creator']} "
            f"({bg_credit['license']}) - {bg_credit['source']}",
            file=sys.stderr,
        )

    clips = []
    aid_credits = []
    for i, line in enumerate(lines):
        wav_bytes = synthesize(line["text"], line["character"], line["emotion"], line.get("se", "none"))
        wav_path = tmp_dir / f"line_{i:02d}.wav"
        wav_path.write_bytes(wav_bytes)
        audio_clip = AudioFileClip(str(wav_path))
        dur = audio_clip.duration

        cur_chars = shots[i]
        prev_chars = shots[i - 1] if i > 0 else []
        next_chars = shots[i + 1] if i < len(lines) - 1 else cur_chars
        prev_line = lines[i - 1] if i > 0 else None

        layer_clips = [ImageClip(str(bg_path)).with_duration(dur)]

        # 発言者のレイヤーが常に手前(最後に描画)になるよう、非発言→発言の順に並べ替える
        ordered_chars = sorted(cur_chars, key=lambda c: c == line["character"])

        for who in ordered_chars:
            is_pair_member = who in main_pair
            active = who == line["character"]
            emotion = line["emotion"] if active else "base"
            scale_ratio = PAIR_SCALE if is_pair_member else CUTIN_SCALE
            side = None if is_pair_member else cutin_side(who, i)
            # 立ち絵のデフォルトは左向き。右側カットインはそのままで内側(左)を向くが、
            # 左側カットインは反転させて内側(右)を向かせる。
            mirror = side == "left"

            png_path, sw, sh = sprite_variant_path(who, emotion, active, scale_ratio, h, tmp_dir, mirror)
            rest_x, edge_x, _ = rest_and_edge_x(who, main_pair, w, sw, i)
            y = h - sh - int(h * (0.12 if is_pair_member else 0.025))
            entering = who not in prev_chars
            exiting = who not in next_chars
            pos_fn = build_position_fn(rest_x, edge_x, entering, exiting, dur, y)

            # 主要ペアの発言/非発言が切り替わる瞬間は、急なカットではなくクロスフェードで繋ぐ
            was_active = is_pair_member and prev_line is not None and prev_line["character"] == who and who in prev_chars
            need_crossfade = is_pair_member and not entering and (was_active != active)

            if need_crossfade:
                cross_t = min(CROSSFADE_T, dur * 0.4)
                prev_emotion = prev_line["emotion"] if was_active else "base"
                old_png, old_sw, old_sh = sprite_variant_path(who, prev_emotion, was_active, scale_ratio, h, tmp_dir, mirror)
                old_rest_x, _, _ = rest_and_edge_x(who, main_pair, w, old_sw, i)
                old_y = h - old_sh - int(h * 0.12)
                # 旧カットは新カットのフェードインに合わせて消え、それ以降は残さない（二重表示防止）
                old_clip = ImageClip(str(old_png)).with_duration(cross_t).with_position((old_rest_x, old_y))
                new_clip = (
                    ImageClip(str(png_path))
                    .with_duration(dur)
                    .with_position(pos_fn)
                    .with_effects([CrossFadeIn(cross_t)])
                )
                layer_clips.append(old_clip)
                layer_clips.append(new_clip)
            else:
                char_clip = ImageClip(str(png_path)).with_duration(dur).with_position(pos_fn)
                layer_clips.append(char_clip)

        # 発言者(=ordered_charsの最後に処理したキャラ)の頭上に、視聴維持を意識した軽量エフェクトを添える
        effect_kind = effect_kind_for_line(line)
        if effect_kind:
            icon_size = int(h * 0.10)
            icon_x = rest_x + int(sw * 0.55)
            icon_y = max(0, y - int(icon_size * 0.55))
            icon_path = tmp_dir / f"fx_{i:02d}.png"
            render_effect_icon(effect_kind, icon_size).save(icon_path)
            show_t = min(1.1, dur * 0.55)
            pop_t = min(0.14, show_t * 0.4) or 0.01

            def icon_scale_fn(t, pop_t=pop_t):
                if t >= pop_t:
                    return 1.0
                return 0.5 + 0.5 * ease_out(t / pop_t)

            icon_clip = (
                ImageClip(str(icon_path))
                .with_duration(show_t)
                .with_position((icon_x, icon_y))
                .resized(icon_scale_fn)
            )
            layer_clips.append(icon_clip)

        visual_aid_kind = line.get("visual_aid", "none")
        if visual_aid_kind in VISUAL_AID_LABELS:
            if w > h:
                # 横長(長尺)フォーマット: 幅基準で出すと肥大化しキャラと被るため、
                # 高さ基準サイズ・中央の隙間に配置してキャラと重ならないようにする
                box_size = int(h * 0.30)
                aid_x, aid_y = int(w * 0.5 - box_size / 2), int(h * 0.32)
            else:
                box_size = int(w * 0.28)
                aid_x, aid_y = int(w * 0.66), int(h * 0.30)
            card_path = tmp_dir / f"aid_{i:02d}.png"
            render_visual_aid_card(visual_aid_kind, box_size).save(card_path)
            pop_t = min(0.16, dur * 0.3) or 0.01

            def aid_scale_fn(t, pop_t=pop_t):
                if t >= pop_t:
                    return 1.0
                return 0.6 + 0.4 * ease_out(t / pop_t)

            aid_clip = (
                ImageClip(str(card_path))
                .with_duration(dur)
                .with_position((aid_x, aid_y))
                .resized(aid_scale_fn)
            )
            layer_clips.append(aid_clip)

        subtitle_path = tmp_dir / f"subtitle_{i:02d}.png"
        if i == 0:
            # オープニングの最初の一言のみ専用タイトルカード（お題そのもの＋主張テロップ）を表示し、
            # 実際の読み上げセリフの字幕は出さない
            render_title_card(data["theme"], size, title_positions).save(subtitle_path)

            # 冒頭3秒のインパクト演出: タイトルカードをズームインしながらポップインさせ、
            # 開始直後に一瞬の白フラッシュを重ねる（掴みの強化。視聴維持率対策）
            pop_t = min(0.35, dur * 0.5)

            def _title_pop_scale(t, pop_t=pop_t):
                if t >= pop_t:
                    return 1.0
                return 1.18 - 0.18 * ease_out(t / pop_t)

            def _title_pop_pos(t, pop_t=pop_t, w=w, h=h):
                s = _title_pop_scale(t, pop_t)
                return ((w - w * s) / 2, (h - h * s) / 2)

            title_clip = (
                ImageClip(str(subtitle_path))
                .with_duration(dur)
                .resized(_title_pop_scale)
                .with_position(_title_pop_pos)
            )
            layer_clips.append(title_clip)

            flash_path = tmp_dir / "flash_open.png"
            if not flash_path.exists():
                Image.new("RGBA", size, (255, 255, 255, 200)).save(flash_path)
            flash_t = min(0.09, dur * 0.3)
            layer_clips.append(ImageClip(str(flash_path)).with_duration(flash_t).with_position((0, 0)))
        else:
            render_subtitle_layer(line, size, subtitle_color_overrides).save(subtitle_path)
            layer_clips.append(ImageClip(str(subtitle_path)).with_duration(dur).with_position((0, 0)))

        se_name = line.get("se", "none")
        line_audio = audio_clip
        if se_name in SE_FILES:
            vol = SE_VOLUME.get(se_name, 0.2)
            se_clip = AudioFileClip(str(SE_FILES[se_name])).with_effects([MultiplyVolume(vol)])
            if se_name in SE_TIMING_AT_END:
                # セリフの言い終わりに合わせて鳴らす（間延び・ズレ防止のため末尾に寄せる）
                start_t = max(0.0, dur - se_clip.duration)
                se_clip = se_clip.with_start(start_t)
            line_audio = CompositeAudioClip([audio_clip, se_clip]).subclipped(0, dur)

        composite = CompositeVideoClip(layer_clips, size=size).with_duration(dur).with_audio(line_audio)
        clips.append(composite)

    final = concatenate_videoclips(clips, method="compose")

    bgm_key = "climax" if data.get("dady_appearance") else "normal"
    bgm_path = BGM_TRACKS.get(bgm_key)
    if bgm_path and bgm_path.exists():
        bgm = AudioFileClip(str(bgm_path)).with_effects(
            [AudioLoop(duration=final.duration), MultiplyVolume(BGM_VOLUME)]
        )
        final = final.with_audio(CompositeAudioClip([final.audio, bgm]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{script_path.stem}.mp4"
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac", logger=None)

    description = build_description(data, bg_credit, aid_credits)
    desc_path = OUTPUT_DIR / f"{script_path.stem}.description.txt"
    desc_path.write_text(description, encoding="utf-8")

    return out_path


# ==========================================================================
# 古文チャンネル「5コマ漫画+VOICEVOXアフレコ」形式 (kobun_5koma)
#
# 1コマ(panel)につき、背景・キャラクター・空の吹き出し(文字なし)を含む縦スクロール
# Webtoon風の1枚絵をGeminiに一括生成させる(2026-09-15、週1〜2ペース運用に合わせて
# 生成回数を1コマあたり画像1回+吹き出し位置検出1回まで削減)。コマ内の各セリフ(line)は
# 同じ1枚絵のまま、対応する吹き出しへセリフの文字だけをアフレコに合わせて順番に重ねる。
# コマとコマの間は縦スクロール風のトランジションでつなぐ。詳細な設計判断は
# 詳細な設計判断は別途ドキュメント化している。
# ==========================================================================


def remove_white_bg(img: Image.Image) -> Image.Image:
    """Gemini等で生成した白背景のキャラクター立ち絵を透過PNGに変換する。

    コーナーから連結した白領域を背景として除去したうえで、残った不透明領域のうち
    最大の連結成分(=キャラクター本体)以外は、生成時に付与されがちな淡いドロップ
    シャドウ等のノイズとみなして併せて透過する。
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].astype(int)
    # 近似白は連結の有無に関わらずすべて透過にする(脚の間などキャラ本体に囲まれた孤立した
    # 白領域は、コーナーからの連結だけを見ていると取り逃し、うっすら白いノイズとして残っていた)
    is_white = np.all(np.abs(rgb - 255) <= 18, axis=2)
    arr[:, :, 3] = np.where(is_white, 0, arr[:, :, 3])

    opaque = arr[:, :, 3] > 0
    comp_labeled, n = label(opaque)
    if n > 1:
        sizes = np.bincount(comp_labeled.ravel())
        sizes[0] = 0
        main_label = sizes.argmax()
        noise_mask = opaque & (comp_labeled != main_label)
        arr[:, :, 3] = np.where(noise_mask, 0, arr[:, :, 3])

    out = Image.fromarray(arr)
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)
    return out


# 行頭に来てはいけない文字(日本語の禁則処理の簡易版)。これらの文字の直前では
# 改行せず、はみ出してでも現在行に含める(フィードバック2026-09-16: 違和感のある
# 位置での改行への対応)。
_NO_LINE_START_CHARS = set("、。，．！？!?…・ー」』）)ゝゞぁぃぅぇぉっゃゅょァィゥェォッャュョ")


def _fit_bubble_text(box_w: int, box_h: int, text: str):
    """box_w×box_hの吹き出し(検出されたのは楕円/丸みのある形の外接矩形)に収まる
    最大のフォントサイズ・折り返し行・行間・余白を探す。

    フィードバック(2026-09-16、3回目)で「セリフが吹き出しからはみ出す」との指摘が
    続いたため、外接矩形いっぱいを使うのをやめ、楕円に内接する矩形(各辺を約1/√2倍)を
    実際の文字使用領域にした。これにより丸みのある吹き出しの輪郭にかからなくなる。
    """
    margin = max(4, int(min(box_w, box_h) * 0.04))
    inscribed_w = box_w / 1.42
    inscribed_h = box_h / 1.42
    max_width = max(1, int(inscribed_w) - margin * 2)
    max_height = max(1, int(inscribed_h) - margin * 2)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    font = lines = None
    line_h = 0
    for size in range(64, 6, -1):
        font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", size)
        lines, cur = [], ""
        for ch in text:
            test = cur + ch
            bbox = probe.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_width and cur and ch not in _NO_LINE_START_CHARS:
                lines.append(cur)
                cur = ch
            else:
                cur = test
        if cur:
            lines.append(cur)
        line_h = int(size * 1.3)
        # 禁則処理(_NO_LINE_START_CHARS)は行末の句読点を意図的にはみ出させて改行するため、
        # 高さだけでなく各行の実測幅もmax_width以内かを確認してからこのサイズを確定する
        # (フィードバック2026-09-19: 句読点のはみ出しでも吹き出し外に文字が出ていた)。
        max_line_w = max(
            (probe.textbbox((0, 0), line, font=font)[2] for line in lines), default=0
        )
        if (line_h * len(lines) <= max_height and max_line_w <= max_width) or size <= 7:
            break
    return font, lines, line_h, margin


def _ink_bbox(text: str, font) -> tuple[int, int, int, int]:
    """textを実際に描画して、透明でない(=インクがある)ピクセルだけの外接矩形を返す。

    ImageDraw.textbbox()は「。」「！」等の全角記号で、実際に見えるインクよりずっと
    右側まで幅を返すことがある(フォントの送り幅に、見た目には空白の余白が含まれるため)。
    この幅をそのまま中央寄せに使うと、見た目のインクが左に寄って見える
    (フィードバック2026-09-19: 「心して走れ。」のような句点で終わる短いセリフで、
    左右の余白が大きく非対称になる不具合が実測で確認された)。
    """
    probe_bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    pad = 4
    w = max(1, probe_bbox[2] - probe_bbox[0]) + pad * 2
    h = max(1, probe_bbox[3] - probe_bbox[1]) + pad * 2
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((pad - probe_bbox[0], pad - probe_bbox[1]), text, font=font, fill=(0, 0, 0, 255))
    ink = tmp.getchannel("A").getbbox()
    if ink is None:
        return (0, 0, 0, 0)
    x0, y0, x1, y1 = ink
    return (x0 - pad, y0 - pad, x1 - pad, y1 - pad)


def _render_bubble_lines(
    box_w: int, box_h: int, font, lines: list[str], line_h: int, margin: int, char_limit: int | None = None
) -> Image.Image:
    """折り返し済みのlinesを描画する。char_limit指定時は先頭からその文字数分だけ描画する
    (タイプライター表示用。行の折り返し位置・中央寄せの基準は常に完成文で計算するため、
    文字が増えていっても行がガタつかない)。"""
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    total_h = line_h * len(lines)
    y = max(margin, (box_h - total_h) // 2)
    remaining = char_limit if char_limit is not None else sum(len(line) for line in lines)
    for line in lines:
        shown = line if char_limit is None else line[: max(0, remaining)]
        remaining -= len(line)
        # 中央寄せは常に完成した行のインク幅を基準にする(タイプライターで途中まで
        # しか表示していなくても位置がガタつかないよう、full_wはlineから計算する)。
        ink_x0, _, ink_x1, _ = _ink_bbox(line, font)
        ink_w = ink_x1 - ink_x0
        x_target = max(margin, (box_w - ink_w) // 2)
        x = x_target - ink_x0
        if shown:
            draw.text((x, y), shown, font=font, fill=(15, 15, 15, 255))
        y += line_h
    return img


def draw_bubble_text(box_w: int, box_h: int, text: str) -> Image.Image:
    """AIが描いた空の吹き出しの上に重ねる、セリフ文字だけの透過画像を作る
    (吹き出し自体の輪郭・しっぽはパネル画像側に既に描かれているため、ここでは文字だけを描く)。"""
    font, lines, line_h, margin = _fit_bubble_text(box_w, box_h, text)
    return _render_bubble_lines(box_w, box_h, font, lines, line_h, margin)


# これを下回るフォントサイズはもう読みにくいと判断し、1つの吹き出しに収めるのを諦めて
# 複数ページ(同じ吹き出し内で文が入れ替わる)に分割する(フィードバック2026-09-19:
# 現代語訳のような長い1文が、吹き出しに対して極小フォントで押し込まれていた)。
READABLE_MIN_FONT_SIZE = 20


def _split_sentences(text: str) -> list[str]:
    """句点・感嘆符・疑問符・三点リーダの直後で文を分割する(長すぎるセリフを
    複数ページに分けるための最小単位。禁則処理と違い、ここでは行ではなく文単位)。"""
    parts = re.split(r"(?<=[。！？…])", text)
    return [p for p in parts if p]


def _fit_bubble_pages(box_w: int, box_h: int, text: str) -> list[tuple]:
    """1つの吹き出しに表示する内容を、(font, lines, line_h, margin)のページ列として返す。

    読める最小サイズ(READABLE_MIN_FONT_SIZE)で全文が収まればページは1つ。
    収まらない場合は文単位で複数ページに分割し、各ページを個別にフィットさせる
    (フィードバック2026-09-19: 「セリフが長すぎる場合は吹き出しを増やして分割」)。
    """
    fitted = _fit_bubble_text(box_w, box_h, text)
    if fitted[0].size >= READABLE_MIN_FONT_SIZE:
        return [fitted]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [fitted]  # 文が1つしかなく、これ以上分割しようがない

    pages: list[str] = []
    buf = ""
    for s in sentences:
        candidate = buf + s
        candidate_font, _, _, _ = _fit_bubble_text(box_w, box_h, candidate)
        if not buf or candidate_font.size >= READABLE_MIN_FONT_SIZE:
            buf = candidate
        else:
            pages.append(buf)
            buf = s
    if buf:
        pages.append(buf)
    return [_fit_bubble_text(box_w, box_h, p) for p in pages]


def _typewriter_single_page(
    box_w: int, box_h: int, font, lines: list[str], line_h: int, margin: int,
    duration: float, tmp_dir: Path, tag: str,
) -> tuple:
    """1ページ分のタイプライター演出クリップと、完成形(全文字表示済み)の画像パスを返す。"""
    full_path = tmp_dir / f"{tag}_full.png"
    _render_bubble_lines(box_w, box_h, font, lines, line_h, margin).save(full_path)

    total_chars = sum(len(line) for line in lines)
    if total_chars == 0:
        return ImageClip(str(full_path)).with_duration(duration), full_path

    reveal_span = max(0.05, min(duration * 0.85, max(0.05, duration - 0.1)))
    per_char = reveal_span / total_chars
    frame_clips = []
    for k in range(1, total_chars + 1):
        frame_path = tmp_dir / f"{tag}_{k}.png"
        _render_bubble_lines(box_w, box_h, font, lines, line_h, margin, char_limit=k).save(frame_path)
        frame_clips.append(ImageClip(str(frame_path)).with_duration(per_char))

    hold = duration - per_char * total_chars
    if hold > 0.01:
        frame_clips.append(ImageClip(str(full_path)).with_duration(hold))

    return concatenate_videoclips(frame_clips, method="compose"), full_path


def draw_bubble_typewriter_clip(box_w: int, box_h: int, text: str, duration: float, tmp_dir: Path, tag: str):
    """セリフが左から一文字ずつ表示されていくタイプライター演出のクリップを作る
    (フィードバック2026-09-16)。durationの85%程度をかけて全文字を出し切り、
    残りは完成した文を静止表示する。

    吹き出しに対して長すぎるセリフ(現代語訳等)は`_fit_bubble_pages`で複数ページに
    分割し、durationを文字数比例で配分してページを順番に表示する
    (フィードバック2026-09-19)。戻り値は(クリップ, 最終ページの完成画像パス)のタプルで、
    呼び出し側が「セリフ表示をコマの終わりまで延長する」際に完成画像を再利用できるようにする。
    """
    pages = _fit_bubble_pages(box_w, box_h, text)
    if len(pages) == 1:
        font, lines, line_h, margin = pages[0]
        return _typewriter_single_page(box_w, box_h, font, lines, line_h, margin, duration, tmp_dir, tag)

    char_counts = [sum(len(l) for l in lines) for (_, lines, _, _) in pages]
    total = sum(char_counts) or 1
    clips = []
    final_path = None
    for idx, ((font, lines, line_h, margin), n_chars) in enumerate(zip(pages, char_counts)):
        page_dur = max(0.35, duration * n_chars / total)
        clip, final_path = _typewriter_single_page(
            box_w, box_h, font, lines, line_h, margin, page_dur, tmp_dir, f"{tag}_p{idx}"
        )
        clips.append(clip)
    return concatenate_videoclips(clips, method="compose"), final_path


IMAGE_MODEL_NAME = "gemini-3.1-flash-image"
VISION_TEXT_MODEL_NAME = "gemini-flash-lite-latest"
KOBUN_SCROLL_TRANSITION_SECONDS = 0.45
KOBUN_BGM_VOLUME = 0.08  # セリフの聞き取りやすさを優先して控えめに(ch1のBGM_VOLUMEと同じ考え方)

# トウマ/イロハ/先生のビジュアル設定。
# 1コマの合成マンガをGeminiに一度に描かせるためのキャラ外見の正本で、ここを直接編集すれば
# 全コマの生成に反映される。
KOBUN_CHARACTER_APPEARANCE = {
    "ren": (
        "男子高校生「トウマ」。輪郭は丸みのある卵型、やや垂れ気味の一重〜奥二重の焦茶色の瞳(黒目がちで"
        "まつ毛は短め)、眉は太めで直線的。黒髪の癖毛ショートヘア(毛先が跳ねる立体的なシルエット、"
        "前髪は重めで眉の上あたりまで、右側だけ少し長い毛束が一房落ちる)。肌はやや小麦色。"
        "黒の詰襟学生服(スタンドカラー、金ボタン5個を縦一列、襟の左側に長方形の校章ピン)、黒ズボン、"
        "黒革ローファー。身長はイロハよりやや高い程度の平均的な高校生体型"
    ),
    "iroha": (
        "女子高校生「イロハ」。輪郭はやや面長でシャープな卵型、大きめの二重の焦茶色の瞳(まつ毛長め、"
        "目尻に軽くアイラインのような描線)、眉は細めで弧を描く。茶色のロングヘア(胸の下まで届く長さ、"
        "毛先は緩やかに内巻き)、左サイドだけ細い三つ編みにして赤いリボンでまとめ、後頭部でハーフアップに"
        "結ぶ。紺のブレザー(金ボタン2個)、白シャツ、襟元に赤いリボンタイ、グレーのプリーツスカート"
        "(膝上丈)、黒タイツ、白と黒のツートンのメリージェーン風ローファー。細身の体型"
    ),
    "sensei": (
        "古文教師「先生」。輪郭は角のある面長、細い目(黒縁の四角い眼鏡の奥、目尻は少し下がり気味で"
        "温和な印象)、眉は太めで水平。黒髪ショート(七三分け寄りに整えた短髪、清潔感のあるビジネス"
        "ヘア)。濃紺〜黒のビジネススーツ(細いストライプ入り)、白シャツ、ネイビーのストライプネクタイ、"
        "黒革ベルト、黒革ローファー。生徒たちより頭一つ分背が高い、痩せ型で姿勢がよい"
    ),
}


def _erase_regions(
    img: Image.Image,
    boxes_xyxy: list[tuple[int, int, int, int]],
    pad: int = 8,
    dark_only: bool = False,
    dark_threshold: int = 140,
) -> Image.Image:
    """指定した矩形領域をcv2.inpaintで自然に塗りつぶして消去する
    (使われなかった余分な吹き出しを画像から取り除くために使う)。

    dark_only=Trueの場合、矩形全体ではなく矩形内の暗いピクセル(文字インク相当、
    輝度dark_threshold未満)だけをマスクする。吹き出し内側の予防的消去
    (used_interiors_px)は、検出された吹き出しの外接矩形がしっぽ分だけ実際の
    楕円より縦に長いことがあり、矩形全体を塗りつぶすとしっぽ周辺の背景(空や
    旗の色)まで巻き込んでinpaintがにじんだ雲のような不自然な色を作ってしまう
    (フィードバック2026-09-20で実機確認)。文字インクは黒に近い色で塗っているため、
    暗いピクセルだけを対象にすれば、白い吹き出し内部や矩形がはみ出した背景部分は
    触らずに済む。"""
    arr = np.array(img.convert("RGB"))
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    for x0, y0, x1, y1 in boxes_xyxy:
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        if dark_only:
            region = arr[y0:y1, x0:x1].astype(np.float32)
            lum = region[..., 0] * 0.299 + region[..., 1] * 0.587 + region[..., 2] * 0.114
            local_mask = (lum < dark_threshold).astype(np.uint8) * 255
            # 文字の縁のアンチエイリアス部分まで消し切るため軽く膨張させる
            local_mask = cv2.dilate(local_mask, np.ones((5, 5), np.uint8))
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local_mask)
        else:
            mask[y0:y1, x0:x1] = 255
    if not mask.any():
        return img
    inpainted = cv2.inpaint(arr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def _filter_plausible_bubbles(
    by_speaker: dict[str, list[list[int]]],
    panel_img: Image.Image,
    w: int,
    h: int,
    min_white_frac: float = 0.3,
) -> dict[str, list[list[int]]]:
    """検出された吹き出し候補から、実際には白い吹き出し内部を含んでいない
    (=誤検出。キャラの顔・服・背景等を指してしまっている)ものを除外する。

    フィードバック(2026-09-20): 必要数ちょうどの吹き出しが検出されているにも関わらず、
    そのうち1つの座標が実際にはキャラの首元の陰や背景を指しており、そこにセリフ文字と
    dark_only消去が乗って「顔(周辺)にモザイク+セリフ」になり、代わりに本来セリフが
    入るべき正しい吹き出しが「未使用」として消去される事故が実際の生成画像
    (panel11.png、ren2個目の検出box)で確認された。個数さえ合っていれば通っていた
    従来のshortfallチェック(発話者ごとの検出数のみを見る)では検知できないため、
    座標の中身(白い塗りつぶしピクセルの割合)を見て機械的にふるい落とす。吹き出しは
    常に白地に黒縁で描かせているため、本物の吹き出しなら大部分が白いピクセルになる。
    """
    arr = np.array(panel_img.convert("RGB"))
    filtered: dict[str, list[list[int]]] = {}
    for speaker_key, boxes in by_speaker.items():
        kept = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            x0, y0 = int(xmin / 1000 * w), int(ymin / 1000 * h)
            x1, y1 = int(xmax / 1000 * w), int(ymax / 1000 * h)
            if x1 <= x0 or y1 <= y0:
                continue
            region = arr[y0:y1, x0:x1]
            white_frac = float(np.all(region > 200, axis=-1).mean()) if region.size else 0.0
            if white_frac < min_white_frac:
                print(
                    f"[警告] 検出された吹き出し候補(発話者={speaker_key}, box={box})は"
                    f"白い内部が少なく(白率{white_frac:.2f})吹き出しと考えにくいため除外します。",
                    file=sys.stderr,
                )
                continue
            kept.append(box)
        filtered[speaker_key] = kept
    return filtered


def _assign_bubble_boxes(
    by_speaker: dict[str, list[list[int]]],
    chars_in_panel: list[dict],
    bubble_line_indices: list[int],
    bubble_speaker_idx: list[int],
    w: int,
    h: int,
    panel_id,
    panel_img: Image.Image | None = None,
) -> tuple[dict[int, tuple[int, int, int, int]], dict[str, int]]:
    """発話者キーごとの検出済み吹き出し(by_speaker、0-1000正規化のbox_2dのリスト)を、
    台本のlines内での位置(line_index)に割り当て、ピクセル座標のbubble_boxesを返す。

    `_generate_panel_image`での新規検出直後と、既存の検出結果(サイドカーJSON)を
    読み込んでの再利用の両方から呼べるよう、割り当てロジックを共通化した
    (2026-09-19、長尺/ショート分割で画像を使い回す機能の追加に伴い切り出し)。
    戻り値の2つ目(cursor_by_speaker)は、話者ごとに何個消費したかを表し、
    呼び出し側で「使われなかった吹き出し」を特定するのに使う。

    panel_img指定時は`_filter_plausible_bubbles`で明らかに吹き出しでない候補
    (キャラの顔・背景等の誤検出)を先に除外する(2026-09-20、サイドカーJSON経由の
    再利用時にも同じ安全策を効かせるための引数。新規検出直後は`_generate_panel_image`
    側で既に同じフィルタを適用済みだが、古いサイドカーJSON(このフィルタ導入前に
    保存されたもの)を読み込む場合はここが最後の砦になる)。
    """
    if panel_img is not None:
        by_speaker = _filter_plausible_bubbles(by_speaker, panel_img, w, h)

    cursor_by_speaker: dict[str, int] = {}
    bubble_boxes: dict[int, tuple[int, int, int, int]] = {}
    n_bubbles = len(bubble_line_indices)

    # フィードバック(2026-09-19): 吹き出しが足りない(検出漏れ)時のフォールバック位置が、
    # 読み順だけを見て均等に帯を割り振っていたため、既に検出できている別の吹き出しと
    # 重なってしまい「文字の挿入箇所がなくなる」ことがあった。実際に検出できている
    # 全吹き出しの位置(occupied_bands)を避けるように、フォールバック位置を動的に探す。
    occupied_bands: list[tuple[float, float]] = [
        (box[0], box[2]) for boxes in by_speaker.values() for box in boxes
    ]

    def _pick_fallback_band(order: int) -> tuple[float, float]:
        band_h = 1000 / (n_bubbles + 2)
        candidate = band_h * (order + 1)
        for _ in range(n_bubbles + 4):
            y0, y1 = candidate - band_h / 2, candidate + band_h / 2
            if all(y1 <= a or y0 >= b for a, b in occupied_bands):
                return y0, y1
            candidate += band_h  # 埋まっていたら1帯分ずらして再チェック
        # 最後まで空きが見つからなくても(吹き出しが極端に多いコマ等)、これ以上遅延させず
        # ずらし続けた末尾の位置をそのまま使う(はみ出しより「重なって読めない」方が害が大きい)
        return candidate - band_h / 2, candidate + band_h / 2

    for order, (line_idx, speaker_idx) in enumerate(zip(bubble_line_indices, bubble_speaker_idx)):
        speaker_key = chars_in_panel[speaker_idx - 1]["key"]
        boxes = by_speaker.get(speaker_key, [])
        cursor = cursor_by_speaker.get(speaker_key, 0)
        if cursor < len(boxes):
            ymin, xmin, ymax, xmax = boxes[cursor]
            cursor_by_speaker[speaker_key] = cursor + 1
        else:
            # 検出漏れ: 再生成を試みても、要求どおりの個数を毎回描けるとは限らないため、
            # クラッシュさせず、既存の吹き出しと重ならない帯にフォールバック配置する
            print(
                f"[警告] panel{panel_id}: 吹き出し{order}番目(人物{speaker_key})が検出できませんでした。"
                "フォールバック位置を使用します。",
                file=sys.stderr,
            )
            ymin, ymax = _pick_fallback_band(order)
            xmin, xmax = 150, 850
            occupied_bands.append((ymin, ymax))
        bubble_boxes[line_idx] = (
            int(xmin / 1000 * w), int(ymin / 1000 * h),
            int(xmax / 1000 * w), int(ymax / 1000 * h),
        )
    return bubble_boxes, cursor_by_speaker


def _generate_panel_image(
    panel: dict,
    image_provider: GeminiProvider,
    vision_provider: GeminiProvider,
    asset_dir: Path,
    reference_images: list[tuple[bytes, str]] | None = None,
    aspect_ratio: str = "9:16",
) -> tuple[Image.Image, dict[int, tuple[int, int, int, int]]]:
    """1コマ分の縦スクロールWebtoon風イラストを1枚だけ生成し、セリフ(吹き出しを持つ行のみ)
    に対応する空吹き出し(文字なし)のピクセルboxを、lines内でのインデックスをキーにして返す。

    フィードバック(2026-09-15)により、背景・キャラクター・吹き出しを分離して生成する方式から、
    1枚の絵として一括生成する方式に変更した(頻度を週1〜2に落とす代わりに、1コマあたりの
    生成回数も1回(+吹き出し位置検出の軽いテキスト呼び出し1回)まで削減する)。
    吹き出しの中の文字は空のまま生成させ、実際のセリフ文字は編集側(draw_bubble_text)で
    VOICEVOXのアフレコに合わせて後から重ねる。

    lines のうち type が "caption"(校内放送など、キャラクターの吹き出しを持たない行)の
    ものは吹き出しの生成対象から除外する(kobun_scene_video形式で使用)。

    reference_images(フィードバック2026-09-16): 既に生成済みの同キャラクターの絵を
    参照画像として渡すと、テキストのみの外見指定より顔立ち・体格のブレが小さくなる
    (kobun_scene_video側で、最初に生成したコマ+直前のコマを渡す運用を想定)。
    """
    chars_in_panel = panel["characters_in_panel"]
    lines = panel["lines"]
    bubble_line_indices = [i for i, line in enumerate(lines) if line.get("type", "dialogue") != "caption"]
    bubble_speaker_idx = [
        next(idx for idx, c in enumerate(chars_in_panel, start=1) if c["key"] == lines[line_idx]["character_key"])
        for line_idx in bubble_line_indices
    ]

    char_lines = []
    for idx, c in enumerate(chars_in_panel, start=1):
        appearance = KOBUN_CHARACTER_APPEARANCE[c["key"]]
        line = f"{idx}. {appearance}。{c['pose_hint']}"
        if c.get("effect_hint"):
            line += f"(効果記号: {c['effect_hint']})"
        char_lines.append(line)

    def _size_hint(n_chars: int) -> str:
        if n_chars <= 8:
            return "短め"
        if n_chars <= 18:
            return "中くらい"
        return "長め(十分大きな吹き出しにすること)"

    bubble_lines = [
        f"{i + 1}番目: 上記{speaker_idx}番目の人物が発話。セリフの分量は{_size_hint(len(lines[line_idx]['text']))}"
        f"(目安{len(lines[line_idx]['text'])}文字)"
        for i, (line_idx, speaker_idx) in enumerate(zip(bubble_line_indices, bubble_speaker_idx))
    ]

    panel_prompt = (
        "韓国のWebtoon(縦スクロールのフルカラーデジタル漫画)でよく見る、コマ枠のない一枚絵の"
        "イラストとして描いてください。実際に配信されているWebtoon作品の1シーンのような密度・"
        "画力にすること。「キャラクター設定資料(全身が直立したモデルシート)」のような構図には"
        "絶対にしないこと。\n\n"
        "**構図の指示: バストアップ〜ウエストアップを基本にした、大胆にトリミングされたカメラワークにすること。"
        "感情の強さに応じて顔のクローズアップにしてもよい。ただし顔・頭部だけは画面の上下左右の端で"
        "絶対に切らず、表情が常にはっきり見える範囲に収めること(切ってよいのは肩から下の体の一部のみ)。"
        "直立不動ではなく、机に伏せる/顔を寄せ合う/手を伸ばす等、動きや感情の込もったポーズにすること。**\n"
        "日本の少女漫画・Webtoonで定番の効果記号(汗、集中線、スパークル、衝撃線、怒りマーク、涙等)を、"
        "各キャラクターの感情に合わせて効果的に添えること。\n\n"
        f"場面: {panel['scene_description']}\n\n"
        "登場人物(画面内、向かって左から右の順に記載の順番で並んでいる):\n"
        + "\n".join(char_lines)
        + (
            (
                f"\n\n吹き出し(合計{len(bubble_lines)}個。中に文字・記号は絶対に描かないこと。輪郭と発話者を指す"
                "しっぽだけの空の吹き出しにすること):\n"
                + "\n".join(bubble_lines)
                + "\n\n**吹き出しに関する重要な注意点:**\n"
                "1. 各吹き出しは、対応する発話キャラクターの頭のすぐ近くに配置し、しっぽの先端がそのキャラクターの"
                "顔や頭を明確に指すこと。複数の人物が近くにいる場合でも、しっぽの向きで発話者が一目で分かるように"
                "し、他のキャラクターの吹き出しと絶対に混同されないようにすること。\n"
                "2. 吹き出しの形は必ず横長の楕円にすること(縦長は禁止)。横幅は高さの1.6〜2.2倍程度にすること。\n"
                "3. 各吹き出しはセリフの分量(上記の目安文字数)に応じて、文字が余裕を持って収まる十分な大きさに"
                "すること。小さすぎる吹き出しは禁止。\n"
                "4. どの吹き出しも、画面の上下左右の端に接したりはみ出したりしないよう、余白を持たせて"
                "完全に画面内に収めること。\n"
                "5. 吹き出しの個数は必ずちょうど"
                f"{len(bubble_lines)}個にすること(多すぎても少なすぎてもいけない)。\n"
                "6. 画面の上から下へ読む順番が、上記の1番目→"
                f"{len(bubble_lines)}番目の順に一致するように並べること。"
            )
            if bubble_lines
            else "\n\n吹き出しは描かないこと。"
        )
        + (
            f"\n\n縦長構図({aspect_ratio}、スマートフォン動画向け)。"
            if aspect_ratio == "9:16"
            else f"\n\n横長構図({aspect_ratio}、長尺動画向け)。"
        )
        + "日本のアニメ・Webtoon調のセルルック彩色。"
        + (
            "\n\n参考画像として、これまでに生成した同じキャラクターたちの絵を添付します。"
            "参考画像に写っているキャラクターの顔立ち・髪型・髪色・体格・服装のディテールを、"
            "厳密に同一のキャラクターとして一致させてください(ポーズ・表情・構図は上記の指示に従って"
            "新しく描き直してよい。参考画像に写っていないキャラクターは、テキストの外見指定に従って"
            "新規に描いてください)。"
            if reference_images
            else ""
        )
    )
    box_schema = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
    bubble_schema = {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": [c["key"] for c in chars_in_panel]},
                        "box_2d": box_schema,
                    },
                    "required": ["key", "box_2d"],
                },
            },
            "bubbles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": box_schema,
                        "tail_tip": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["box_2d", "tail_tip"],
                },
            },
        },
        "required": ["characters", "bubbles"],
    }

    def _bbox_center(box_2d):
        ymin, xmin, ymax, xmax = box_2d
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    def _tail_point(bubble: dict):
        # フィードバック(2026-09-19): 吹き出しの外接矩形の中心だけで発話者を推定すると、
        # 複数人が近接する構図で誤って隣のキャラクターに割り当たることがあった。しっぽの
        # 先端(発話者を明確に指す一番細い点)の座標を明示的に検出させ、それを発話者判定の
        # 基準にする方が、外接矩形中心より人間の読み方に近く頑健(自己申告方式より機械的)。
        tail = bubble.get("tail_tip")
        if tail:
            y, x = tail
            return (x, y)
        return _bbox_center(bubble["box_2d"])

    by_speaker: dict[str, list[list[int]]] = {}
    panel_img = w = h = None
    max_attempts = 2
    for attempt in range(max_attempts):
        panel_bytes = image_provider.generate_image(panel_prompt, reference_images=reference_images, aspect_ratio=aspect_ratio)
        panel_path = asset_dir / f"panel{panel['id']}.png"
        panel_path.write_bytes(panel_bytes)
        panel_img = Image.open(panel_path).convert("RGB")
        w, h = panel_img.size

        if not bubble_lines:
            return panel_img, {}

        # フィードバック(2026-09-16、3回目): AI自身に「どの人物を指しているか」を自己申告させる
        # 方式は取り違えが残ったため、各キャラクターの位置も同時に検出させ、各吹き出いの中心に
        # 幾何学的に最も近いキャラクターを発話者とみなす方式に変更した(自己申告より機械的で頑健)。
        bubble_result = vision_provider.generate_json_from_image(
            system_instruction="あなたは画像内の人物と、空の吹き出し(文字なし)を検出するアシスタントです。",
            user_prompt=(
                "この画像内の各人物について、key(以下のいずれか)とbox_2d([ymin,xmin,ymax,xmax]、"
                "0-1000に正規化した整数)を返してください。人物のkey: "
                + "/".join(c["key"] for c in chars_in_panel)
                + f"\n\nまた、この画像には空の吹き出し(中に文字のないもの)が最大{len(bubble_lines)}個"
                "あるはずです。見つかったものすべてについて、box_2d(吹き出し本体の輪郭、同じく"
                "0-1000正規化)に加えて、tail_tip([y,x]、0-1000正規化)として吹き出しの"
                "しっぽの先端(発話者の顔や頭を指している、一番細く尖った点)の座標も返してください。"
                "しっぽがどのキャラクターを指しているかを機械的に判定するために使うので、"
                "吹き出し本体の中心ではなく、しっぽの先端そのものの座標にしてください。"
            ),
            image_bytes=panel_bytes,
            mime_type="image/png",
            schema=bubble_schema,
        )
        char_centers = {c["key"]: _bbox_center(c["box_2d"]) for c in bubble_result.get("characters", [])}

        by_speaker = {}
        for b in bubble_result["bubbles"]:
            bx, by = _tail_point(b)
            if char_centers:
                nearest_key = min(
                    char_centers, key=lambda k: (char_centers[k][0] - bx) ** 2 + (char_centers[k][1] - by) ** 2
                )
            else:
                nearest_key = chars_in_panel[0]["key"]
            by_speaker.setdefault(nearest_key, []).append(b["box_2d"])
        for boxes in by_speaker.values():
            boxes.sort(key=lambda box: box[0])  # ymin昇順(画面の上から下)

        # フィードバック(2026-09-20): 個数は合っていても、座標が実際には吹き出しでない
        # (キャラの顔・服・背景等)ことがあるため、白い塗りつぶし面積で機械的にふるい落として
        # から不足数を判定する(そうしないと誤検出1個が「検出成功」扱いになり、後段で
        # そこにセリフとdark_only消去が乗ってしまう)。
        by_speaker = _filter_plausible_bubbles(by_speaker, panel_img, w, h)

        # フィードバック(2026-09-16、4回目): 合計数だけの判定だと、特定のキャラクターに
        # 吹き出しが偏って(例: 発話者Aに2個・Bに0個)割り当てられても「合計数は足りている」と
        # 誤判定してしまい、結果的にセリフと吹き出しの対応が崩れていた。発話者ごとに必要数を
        # 満たしているかを個別にチェックする方式に変更した。
        required_per_speaker: dict[str, int] = {}
        for speaker_idx in bubble_speaker_idx:
            key = chars_in_panel[speaker_idx - 1]["key"]
            required_per_speaker[key] = required_per_speaker.get(key, 0) + 1
        shortfall = {
            key: need - len(by_speaker.get(key, [])) for key, need in required_per_speaker.items()
        }
        shortfall = {key: n for key, n in shortfall.items() if n > 0}
        # 個数超過(セリフより吹き出しが多い)は、2026-09-19に導入した「使われなかった吹き出しを
        # 画像から消去する」処理(このあと後述)で無条件に解消できるようになったため、
        # 超過だけを理由にした再生成(API呼び出しの追加コスト)はもう発生させない。
        # 不足(shortfall、発話者ごとに必要数を満たしていない)だけが再生成の対象。
        detected_total = sum(len(v) for v in by_speaker.values())
        excess = detected_total > len(bubble_lines)
        if not shortfall:
            break
        if attempt < max_attempts - 1:
            print(
                f"[警告] panel{panel['id']}: 吹き出しが{detected_total}/{len(bubble_lines)}個"
                f"(不足している発話者: {shortfall})。コマ画像を再生成します。",
                file=sys.stderr,
            )

    bubble_boxes, cursor_by_speaker = _assign_bubble_boxes(
        by_speaker, chars_in_panel, bubble_line_indices, bubble_speaker_idx, w, h, panel["id"],
        panel_img=panel_img,
    )

    # 使われなかった吹き出し(セリフの数より多く描かれた分、または必要な発話者以外に
    # 割り当たった分)は、画面に空のまま残ると不自然なので、cv2.inpaintで消去する
    # (フィードバック2026-09-19: 「吹き出しの数がセリフの数より多い」問題への対応。
    # 再生成に頼らず、どんな原因で余分に描かれても確実に画面から消せる)。
    unused_boxes_px: list[tuple[int, int, int, int]] = []
    for speaker_key, boxes in by_speaker.items():
        consumed = cursor_by_speaker.get(speaker_key, 0)
        for box_2d in boxes[consumed:]:
            ymin, xmin, ymax, xmax = box_2d
            unused_boxes_px.append(
                (int(xmin / 1000 * w), int(ymin / 1000 * h), int(xmax / 1000 * w), int(ymax / 1000 * h))
            )

    # 使う予定の吹き出しも、内側を予防的に消去しておく(フィードバック2026-09-19:
    # 「空の吹き出しにすること」というプロンプト指示をGeminiが守らず、英数字や記号を
    # 描いてしまうことがあった。こちらのセリフ文字を後から重ねるだけでは元の文字が
    # 透けて残るため、描画予定の内側領域は原因を問わず必ずクリアしておく)。
    #
    # フィードバック(2026-09-19、2回目): 固定比率(輪郭の1/2.0等)で消すと、セリフが
    # 短い時は消しすぎて輪郭に接近し、セリフが長い時は逆に消去範囲より実際の文字が
    # はみ出て「消した範囲」と「文字を描く範囲」がずれ、境目に不自然な描画が出ていた。
    # `_render_bubble_lines`で実際に描画される文字のインク外接矩形(行間を含む行の高さ
    # ではなく実際に色が乗るピクセルだけ)をそのまま使うことで、常に過不足なく消せるようにした。
    #
    # フィードバック(2026-09-20): それでもなお、検出された吹き出しの外接矩形が
    # しっぽの分だけ縦に長く実際の楕円中心とズレることがあり、矩形消去だと
    # しっぽ周辺の背景(空や旗の色)まで巻き込んでしまっていた。そのため実際の消去
    # (_erase_regions)側でdark_only=Trueを指定し、この矩形内でも文字インク相当の
    # 暗い色のピクセルだけを対象にする(白い吹き出し内部や矩形がはみ出した背景は
    # 触らない)二重の安全策にした。
    used_interiors_px: list[tuple[int, int, int, int]] = []
    for line_idx, (bx0, by0, bx1, by1) in bubble_boxes.items():
        box_w, box_h = bx1 - bx0, by1 - by0
        text = panel["lines"][line_idx]["text"]
        pages = _fit_bubble_pages(box_w, box_h, text)
        ux0 = uy0 = ux1 = uy1 = None
        for _font, _lines, _line_h, _margin in pages:
            if not _lines:
                continue
            rendered = _render_bubble_lines(box_w, box_h, _font, _lines, _line_h, _margin)
            ink = rendered.getchannel("A").getbbox()
            if ink is None:
                continue
            ix0, iy0, ix1, iy1 = ink
            ux0 = ix0 if ux0 is None else min(ux0, ix0)
            uy0 = iy0 if uy0 is None else min(uy0, iy0)
            ux1 = ix1 if ux1 is None else max(ux1, ix1)
            uy1 = iy1 if uy1 is None else max(uy1, iy1)
        if ux0 is None:
            continue
        erase_pad = 10
        used_interiors_px.append((
            bx0 + max(0, ux0 - erase_pad), by0 + max(0, uy0 - erase_pad),
            bx0 + min(box_w, ux1 + erase_pad), by0 + min(box_h, uy1 + erase_pad),
        ))

    if unused_boxes_px:
        print(
            f"[情報] panel{panel['id']}: 使われなかった吹き出し{len(unused_boxes_px)}個を画像から消去します。",
            file=sys.stderr,
        )
        panel_img = _erase_regions(panel_img, unused_boxes_px, pad=8)
    if used_interiors_px:
        # 余白(erase_pad)は上の計算で既に個別に加味済みのため、ここでは追加しない。
        # dark_only=True: 矩形内でも文字インク相当の暗いピクセルだけを消去対象にする。
        panel_img = _erase_regions(panel_img, used_interiors_px, pad=0, dark_only=True)
    if unused_boxes_px or used_interiors_px:
        panel_img.save(panel_path)

    # 発話者キーごとの検出結果(0-1000正規化のbox_2d)をサイドカーJSONに保存しておく。
    # 台本のline_indexに紐付く前の生データなので、同じ画像を別の台本(例: 現代語訳版)
    # から参照する場合でも、話者キーの並び順さえ合っていればvision APIを再度呼ばずに
    # 同じ吹き出し位置を再現できる(2026-09-19、ユーザーからの指摘: 画像と台本の対応が
    # ずれていたことがあり、同じ画像を複数の台本で使い回す予定があるため要注意とのこと)。
    try:
        (asset_dir / f"panel{panel['id']}.detected.json").write_text(
            json.dumps({"image_size": [w, h], "by_speaker": by_speaker}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    return panel_img, bubble_boxes


def _ease_in_out_snappy(p: float, power: float = 3.0) -> float:
    """加速してから減速する(ease-in-out)動き。powerを上げるほど始まりと終わりが
    より緩やかになり、中間が速くなる(「早い加速度」を感じさせる)。"""
    if p < 0.5:
        return 0.5 * (2 * p) ** power
    return 1 - 0.5 * (2 * (1 - p)) ** power


def _scroll_transition_clip(
    from_path: Path,
    to_path: Path,
    w: int,
    h: int,
    tmp_dir: Path,
    duration: float = KOBUN_SCROLL_TRANSITION_SECONDS,
    gap: int | None = None,
    steps: int = 16,
    max_blur: float = 18.0,
):
    """コマ間の切り替えを、縦スクロールWebtoonらしい「前のコマが上へ追い出され、
    次のコマが下から追い上げてくる」動きにするトランジションクリップ。

    フィードバック(2026-09-16、2回目)により、(1)白い余白(gap)をより広く、(2)動きを
    加速→減速のease-in-outに、(3)動きの速い区間にモーションブラーを加えた。
    フィードバック(2026-09-16、3回目)により、余白の高さをコマの高さの約半分にした
    (gap未指定時はh//2を既定値にする)。

    実装方法: 個別に動く複数のクリップを重ねる方式や、位置を時間関数で連続的に動かす
    方式は、いずれかのクリップが画面から完全に外れた瞬間にmoviepyのマスク合成が
    クラッシュする不具合が再現したため、あらかじめ「from画像+白い余白+to画像」を縦に
    1枚の帯画像として合成し、そこから位置・ブラー強度が異なるstepsコマ分を静止画として
    切り出し、短い静止クリップとして繋げる方式にした(タイプライター演出と同じ考え方)。
    """
    if gap is None:
        gap = h // 2
    from_img = Image.open(from_path).convert("RGB").resize((w, h), Image.LANCZOS)
    to_img = Image.open(to_path).convert("RGB").resize((w, h), Image.LANCZOS)
    strip = Image.new("RGB", (w, h * 2 + gap), (255, 255, 255))
    strip.paste(from_img, (0, 0))
    strip.paste(to_img, (0, h + gap))

    travel = h + gap
    ys = [-travel * _ease_in_out_snappy(k / (steps - 1)) for k in range(steps)]

    speeds = []
    for k in range(steps):
        if k == 0:
            speeds.append(abs(ys[1] - ys[0]))
        elif k == steps - 1:
            speeds.append(abs(ys[k] - ys[k - 1]))
        else:
            speeds.append(abs(ys[k + 1] - ys[k - 1]) / 2)
    max_speed = max(speeds) or 1.0

    per_step = duration / steps
    frame_clips = []
    for k in range(steps):
        top = max(0, min(strip.height - h, int(round(-ys[k]))))
        crop = strip.crop((0, top, w, top + h))
        blur_radius = max_blur * (speeds[k] / max_speed)
        if blur_radius > 0.4:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        frame_path = tmp_dir / f"scroll_{from_path.stem}_{to_path.stem}_{k}.png"
        crop.save(frame_path)
        frame_clips.append(ImageClip(str(frame_path)).with_duration(per_step))

    return concatenate_videoclips(frame_clips, method="compose")


# ==========================================================================
# 古文チャンネル「Webtoonコマ+アフレコ演出付き動画」形式 (kobun_scene_video)
#
# kobun_5komaと同じ1コマ1枚絵の生成(_generate_panel_image)を再利用しつつ、
# 吹き出しの順次表示だけでなく、オノマトペのテキスト演出・雨エフェクト・画面揺れ・
# BGMの音量変化を加えた、よりシネマティックな編集を行う。2026-09-15、動画コンテ
# (息子さん向け指示書)をもとに追加。フィードバック(2026-09-16)により、コマ内での
# パン/ズーム(擬似カメラワーク)は「激しすぎる」「画像が見切れる」との指摘を受けて
# 廃止し、各コマは出力フレームに収まる静止画として表示する。
# 詳細な設計判断は別途ドキュメント化している。
# ==========================================================================


def _cover_fit_transform(img_w: int, img_h: int, w: int, h: int) -> tuple[float, float, float]:
    """画像(img_w×img_h)を、出力フレーム(w×h)いっぱいに(はみ出す分はクロップして)
    中央寄せで収めるための(scale, x, y)を返す。ズーム・パンは行わず常にこの一定値を使う。"""
    scale = max(w / img_w, h / img_h)
    x = (w - img_w * scale) / 2
    y = (h - img_h * scale) / 2
    return scale, x, y


def _rain_overlay_image(w: int, h: int, seed: int = 0) -> Image.Image:
    """雨シーン用の、斜めの雨筋を薄く重ねた半透明オーバーレイ画像を1枚作る(静止テクスチャ)。"""
    rng = random.Random(seed)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    n_drops = int(w * h / 4500)
    for _ in range(n_drops):
        x0 = rng.uniform(-h * 0.15, w)
        y0 = rng.uniform(0, h)
        length = rng.uniform(24, 55)
        x1, y1 = x0 - length * 0.25, y0 + length
        alpha = rng.randint(40, 90)
        draw.line([(x0, y0), (x1, y1)], fill=(210, 225, 240, alpha), width=1)
    return img


def _synth_page_turn_audio(fps: int = 44100) -> AudioArrayClip:
    """本のページをめくる音を、帯域フィルタしたノイズのエンベロープで代用する
    (ページをめくる効果音の実素材が無いため、他のSE(チャイム・雨音)と同じくnumpy合成で
    代用。著作権の心配がない利点もある。実際の録音素材に差し替えたい場合は要検討)。"""
    duration = 0.35
    n = int(fps * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.default_rng(1).standard_normal(n)
    b, a = butter(2, [800 / (fps / 2), 5000 / (fps / 2)], btype="band")
    filtered = lfilter(b, a, noise)
    envelope = np.exp(-6 * t) * (1 - np.exp(-40 * t))
    wave = filtered * envelope
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 0.5
    stereo = np.column_stack([wave, wave]).astype(np.float32)
    return AudioArrayClip(stereo, fps=fps)


def _kobun_text_layer(text: str, font, color=(40, 28, 15, 255)) -> Image.Image:
    """透過背景に1行のテキストだけを描いた画像を作る(オープニング/エンディングの
    ポップイン演出用の部品)。"""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    pad = 10
    img = Image.new("RGBA", (bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color)
    return img


def render_kobun_long_opening_clip(w: int, h: int, episode_title: str, tmp_dir: Path, tag: str = "opening"):
    """長尺動画共通のオープニング(約3秒、ユーザー指定のテンプレート、2026-09-19)。

    「ことば先生【古文】」→「物語で覚える古文」→各話タイトルの順に、
    古紙色の背景の上へポップインで表示する。ページをめくる効果音付き。
    今後は各話ごとにepisode_titleだけ差し替えれば使えるテンプレートとして実装した。
    """
    total_dur = 3.0

    # 簡易的な古紙の質感(中央を明るく、周辺をビネットで少し暗くする)
    paper = Image.new("RGB", (w, h), (238, 227, 201))
    vignette = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vignette)
    vd.ellipse([-w * 0.3, -h * 0.3, w * 1.3, h * 1.3], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=min(w, h) * 0.15))
    dark = Image.new("RGB", (w, h), (60, 45, 25))
    bg = Image.composite(paper, dark, vignette)
    bg_path = tmp_dir / f"{tag}_bg.png"
    bg.save(bg_path)

    title_font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.09))
    subtitle_font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.045))
    episode_font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.05))

    parts = [
        ("ことば先生【古文】", title_font, (40, 28, 15, 255), 0.42, 0.3),
        ("物語で覚える古文", subtitle_font, (70, 55, 30, 255), 0.56, 0.9),
        (episode_title, episode_font, (40, 28, 15, 255), 0.74, 1.6),
    ]
    layers = [ImageClip(str(bg_path)).with_duration(total_dur)]
    for i, (text, font, color, y_frac, start) in enumerate(parts):
        img = _kobun_text_layer(text, font, color)
        img_path = tmp_dir / f"{tag}_line{i}.png"
        img.save(img_path)
        x, y = (w - img.width) / 2, h * y_frac - img.height / 2
        pop_t = 0.25

        def scale_fn(t, pop_t=pop_t):
            return 1.0 if t >= pop_t else 0.7 + 0.3 * ease_out(t / pop_t)

        layers.append(
            ImageClip(str(img_path))
            .with_start(start)
            .with_duration(max(0.01, total_dur - start))
            .with_position((x, y))
            .resized(scale_fn)
        )

    video = CompositeVideoClip(layers, size=(w, h)).with_duration(total_dur)
    return video.with_audio(_synth_page_turn_audio().with_start(0.0))


def render_kobun_long_ending_clip(w: int, h: int, closing_line: str, tmp_dir: Path, tag: str = "ending"):
    """長尺動画共通のエンディング(約7秒、ユーザー指定のテンプレート、2026-09-19)。

    最初の2秒は各話の締めの一文(closing_line)、残り5秒は共通の次回予告画面
    「次回も、いとをかし。」「ことば先生【古文】」を表示する。共通画面は、
    YouTubeのエンドスクリーン要素(次の動画/チャンネル登録)と重ならないよう、
    テキストを中央の帯に収め左右に余白を残す。
    今後は各話ごとにclosing_lineだけ差し替えれば使えるテンプレートとして実装した。
    """
    def _draw_centered(draw_obj, text, font, cy, fill=(255, 255, 255)):
        # textbbox直値ではなく実インク幅で中央寄せする(全角句読点で右側に余分な
        # 送り幅があるとtextbbox基準では左に寄って見える。_render_bubble_linesと同じ対策)。
        ink_x0, ink_y0, ink_x1, ink_y1 = _ink_bbox(text, font)
        ink_w = ink_x1 - ink_x0
        x = (w - ink_w) / 2 - ink_x0
        draw_obj.text((x, cy - font.size / 2), text, font=font, fill=fill)

    closing_font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.05))
    closing_img = Image.new("RGB", (w, h), (18, 18, 24))
    d = ImageDraw.Draw(closing_img)
    ink_w_full = _ink_bbox(closing_line, closing_font)
    max_w = w * 0.8
    if (ink_w_full[2] - ink_w_full[0]) > max_w:
        # 長い締めの一文は句読点で1回だけ折り返す(禁則の簡易対応込み)
        mid = len(closing_line) // 2
        break_at = next(
            (i for i in range(mid, len(closing_line)) if closing_line[i] in "、。"), None
        )
        if break_at is not None:
            l1, l2 = closing_line[: break_at + 1], closing_line[break_at + 1 :]
            for i, line in enumerate([l1, l2]):
                _draw_centered(d, line, closing_font, h / 2 + (i - 0.5) * closing_font.size * 1.5 + closing_font.size / 2)
        else:
            _draw_centered(d, closing_line, closing_font, h / 2)
    else:
        _draw_centered(d, closing_line, closing_font, h / 2)
    closing_path = tmp_dir / f"{tag}_closing.png"
    closing_img.save(closing_path)

    common_font1 = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.045))
    common_font2 = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", int(h * 0.06))
    common_img = Image.new("RGB", (w, h), (18, 18, 24))
    d2 = ImageDraw.Draw(common_img)
    common_lines = [("次回も、いとをかし。", common_font1), ("ことば先生【古文】", common_font2)]
    line_hs = [int(f.size * 1.6) for _, f in common_lines]
    total_h = sum(line_hs)
    y = (h - total_h) / 2
    for (text, font), line_h in zip(common_lines, line_hs):
        # YouTubeエンドスクリーン要素(次の動画/チャンネル登録)と重ならないよう、
        # 中央60%幅に収まる前提のフォントサイズにしている(左右に約20%ずつ余白)。
        _draw_centered(d2, text, font, y + line_h / 2)
        y += line_h
    common_path = tmp_dir / f"{tag}_common.png"
    common_img.save(common_path)

    closing_dur, common_dur = 2.0, 5.0
    closing_clip = ImageClip(str(closing_path)).with_duration(closing_dur)
    common_clip = ImageClip(str(common_path)).with_duration(common_dur).with_effects([CrossFadeIn(0.4)])
    return concatenate_videoclips([closing_clip, common_clip], method="compose")


def _pick_caption_y(out_h: int, cap_h: int, occupied_ranges: list[tuple[int, int]]) -> int:
    """字幕バーのY座標を、そのコマでまだ表示され続けている吹き出しと重ならない位置に選ぶ。

    フィードバック(2026-09-20): 単語解説キャプションが2行になりバーの高さが増えた際、
    直前のセリフの吹き出し(セリフを言い終えてもコマが終わるまで表示され続ける仕様。
    2026-09-19の「セリフの表示継続」修正参照)と重なって見づらくなることがあった。
    まず既定位置(画面下寄り)を試し、重なる場合はその中で最も上にある吹き出しの
    上端のすぐ上までバーを押し上げる処理を、空きが見つかるか上限(画面上部)に
    達するまで繰り返す(`_pick_fallback_band`と同じ「まず既定→衝突したらずらす」方式)。
    """
    default_y = min(int(out_h * 0.60), out_h - cap_h - int(out_h * 0.02))
    min_y = int(out_h * 0.04)
    margin = int(out_h * 0.015)
    y = default_y
    for _ in range(len(occupied_ranges) + 2):
        conflicts = [ry0 for ry0, ry1 in occupied_ranges if y < ry1 and (y + cap_h) > ry0]
        if not conflicts:
            return y
        y = max(min_y, min(conflicts) - cap_h - margin)
        if y <= min_y:
            break
    return y


def draw_caption_bar(w: int, h: int, text: str) -> Image.Image:
    """校内放送・古文単語の意味解説など、キャラクターの吹き出しを持たない発話向けの、
    字幕バーを描く。

    フィードバック(2026-09-19): 「文字が小さくて読めない」「長い説明で見切れる」との
    指摘を受け、(1)フォントサイズを引き上げ、(2)幅に収まらなければ折り返し、
    (3)バーの高さを行数に応じて可変にするよう修正した(旧実装は固定サイズ1行のみで、
    はみ出した分は単純に画面外に切れていた)。

    フィードバック(2026-09-19、2回目): フォントサイズをh(フレーム高さ)基準にしていたため、
    ショート(9:16、h=1920)では読みやすくても、長尺(16:9、h=1080)では同じ比率でも
    絶対サイズが小さくなりすぎていた(1920→1080で高さが約半分)。縦横どちらのフレームでも
    長辺は共通(このプロジェクトでは常に1920)なので、max(w, h)基準に変更し、
    フォーマットによらず同じ絶対サイズになるようにした。
    """
    font_size = int(max(w, h) * 0.026)
    font = load_font(FONTS_DIR / "RoundedMplus1c-Black.ttf", font_size)
    pad_x = int(w * 0.06)
    max_width = w - pad_x * 2
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    lines, cur = [], ""
    for ch in text:
        test = cur + ch
        bbox = probe.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and cur and ch not in _NO_LINE_START_CHARS:
            lines.append(cur)
            cur = ch
        else:
            cur = test
    if cur:
        lines.append(cur)

    line_h = int(font_size * 1.4)
    pad_y = int(font_size * 0.6)
    bar_h = line_h * len(lines) + pad_y * 2

    img = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, bar_h], fill=(10, 10, 10, 190))
    y = pad_y
    for line in lines:
        ink_x0, _, ink_x1, _ = _ink_bbox(line, font)
        ink_w = ink_x1 - ink_x0
        x = (w - ink_w) // 2 - ink_x0
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return img


# 環境音・SE用の音源ファイルが用意できていない演出(校内放送のチャイム、雨音)を、
# numpyで直接合成して代用する(フィードバック2026-09-16: 「効果音の導入」への対応。
# 著作権の心配がない完全オリジナル音源になる利点もある)。


def _synth_chime_audio(fps: int = 44100) -> AudioArrayClip:
    """校内放送のチャイム(「ピンポンパンポーン」風の2音)を合成する。"""

    def tone(duration: float, freq: float, decay: float = 7.0):
        t = np.linspace(0, duration, int(fps * duration), endpoint=False)
        return np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)

    wave = np.concatenate([tone(0.4, 880.0), np.zeros(int(fps * 0.05)), tone(0.55, 659.25)])
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 0.5
    stereo = np.column_stack([wave, wave]).astype(np.float32)
    return AudioArrayClip(stereo, fps=fps)


def _synth_rain_audio(duration: float, fps: int = 44100, intensity: float = 0.12) -> AudioArrayClip:
    """雨音を、バンドパスフィルタをかけたホワイトノイズで合成する(環境音素材がないため代用)。"""
    n = max(1, int(fps * duration))
    noise = np.random.default_rng(0).standard_normal(n)
    b, a = butter(2, [1000 / (fps / 2), 8000 / (fps / 2)], btype="band")
    filtered = lfilter(b, a, noise)
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-9) * intensity
    stereo = np.column_stack([filtered, filtered]).astype(np.float32)
    return AudioArrayClip(stereo, fps=fps)


_SYNTH_SE = {"chime": _synth_chime_audio}


def assemble_kobun_panel_video(script_path: Path) -> Path:
    """縦スクロールWebtoon風の1枚絵コマ+VOICEVOXアフレコ形式の動画を組み立てる。

    台本JSONの各panelは、生成済みの画像パスではなく「場面description + 登場キャラとその
    ポーズ」を指定する(素材は本関数がGeminiで都度生成する)。生成した素材は
    materials/kobun_generated/<台本ファイル名>/ に保存され、以後の再実行では使い回さず
    毎回新規生成する(キャラクター画像は「毎コマ新規生成」という企画方針のため)。

    セリフは、パネル画像側に既に描かれている空の吹き出しへ、VOICEVOXのアフレコに合わせて
    1行ずつ順番に文字だけを重ねる形で表示する。コマとコマの間は、次のコマが下から追い上げて
    くる縦スクロール風のトランジションでつなぐ。
    """
    data = json.loads(script_path.read_text(encoding="utf-8"))
    panels = data["panels"]

    image_provider = GeminiProvider(IMAGE_MODEL_NAME)
    vision_provider = GeminiProvider(VISION_TEXT_MODEL_NAME)

    asset_dir = PROJECT_ROOT / "materials" / "kobun_generated" / script_path.stem
    asset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        panel_clips = []
        panel_paths = []
        w = h = None

        for panel in panels:
            panel_img, bubble_boxes = _generate_panel_image(panel, image_provider, vision_provider, asset_dir)
            w, h = panel_img.size
            panel_path = asset_dir / f"panel{panel['id']}.png"
            panel_paths.append(panel_path)

            lines = panel["lines"]
            resolved_lines = []
            t_cursor = 0.0
            for i, line in enumerate(lines):
                char_key = line["character_key"]
                speaker = kvc.resolve_speaker(char_key, line["emotion"])
                wav_bytes = synthesize(
                    line["text"], character=None, emotion=None,
                    speaker=speaker, emotion_params=kvc.emotion_to_params(line["emotion"]),
                )
                wav_path = tmp_dir / f"kobun_{panel['id']}_{i}.wav"
                wav_path.write_bytes(wav_bytes)
                audio_clip = AudioFileClip(str(wav_path))
                dur = audio_clip.duration
                resolved_lines.append(
                    {**line, "line_index": i, "audio_clip": audio_clip, "dur": dur, "start": t_cursor}
                )
                t_cursor += dur + kvc.LINE_GAP_SECONDS

            panel_dur = t_cursor - kvc.LINE_GAP_SECONDS  # 最後の行の後ろの余分な間は詰める
            layer_clips = [ImageClip(str(panel_path)).with_duration(panel_dur)]

            # 吹き出しの輪郭・しっぽはパネル画像側に既に描かれているので、ここではセリフの
            # 文字だけを、対応する吹き出しの検出box上に、アフレコに合わせて順番に重ねる。
            for rl in resolved_lines:
                bx0, by0, bx1, by1 = bubble_boxes[rl["line_index"]]
                box_w, box_h = bx1 - bx0, by1 - by0
                text_img = draw_bubble_text(box_w, box_h, rl["text"])
                text_path = tmp_dir / f"text_{panel['id']}_{rl['start']:.3f}.png"
                text_img.save(text_path)

                pop_t = min(0.1, rl["dur"] * 0.3) or 0.01

                def text_scale_fn(t, pop_t=pop_t):
                    return 1.0 if t >= pop_t else 0.7 + 0.3 * ease_out(t / pop_t)

                layer_clips.append(
                    ImageClip(str(text_path))
                    .with_duration(rl["dur"])
                    .with_start(rl["start"])
                    .with_position((bx0, by0))
                    .resized(text_scale_fn)
                )

            panel_audio = CompositeAudioClip([rl["audio_clip"].with_start(rl["start"]) for rl in resolved_lines])
            panel_clips.append(
                CompositeVideoClip(layer_clips, size=(w, h)).with_duration(panel_dur).with_audio(panel_audio)
            )

        sequence = []
        for idx, clip in enumerate(panel_clips):
            sequence.append(clip)
            if idx < len(panel_clips) - 1:
                sequence.append(_scroll_transition_clip(panel_paths[idx], panel_paths[idx + 1], w, h, tmp_dir))

        final = concatenate_videoclips(sequence, method="compose")
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{script_path.stem}.mp4"
        final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac")
        return out_path


def assemble_kobun_scene_video(
    script_path: Path,
    *,
    out_w: int = 1080,
    out_h: int = 1920,
    image_aspect_ratio: str = "9:16",
    asset_dir_name: str | None = None,
    scene_ids: list[int] | None = None,
    output_stem: str | None = None,
    reuse_existing_images: bool = False,
    variant_label: str | None = None,
    enable_loop_transition: bool = True,
    opening_episode_title: str | None = None,
    ending_closing_line: str | None = None,
    visual_cache: dict[str, tuple[Path, dict, tuple]] | None = None,
) -> Path:
    """縦スクロールWebtoon風の1枚絵+SE演出+雨エフェクトのシネマティック版動画を
    組み立てる(kobun_scene_video形式)。

    5コマ漫画版(kobun_5koma/assemble_kobun_panel_video)と同じ1コマ1枚絵の生成
    (_generate_panel_image)を再利用しつつ、吹き出しのセリフを左から一文字ずつ表示する
    タイプライター演出・オノマトペのテキスト演出・雨エフェクト・画面揺れ・BGMの音量変化を
    加えた編集を行う。台本JSONのscenes[]にse_cues(擬音語演出+効果音)・rain(雨オーバーレイ)・
    screen_shake_after_line・bgm_volume(BGM音量の相対倍率)・final_hold_seconds(コマ末尾の
    静止ホールド秒数)を指定できる(2026-09-16、フィードバックによりコマ内カメラワークは廃止し、
    各コマは常に出力フレームいっぱいに収まる静止画として表示する)。

    キャラクターデザインのブレを抑えるため、2コマ目以降は「最初に生成したコマ」と「直前の
    コマ」を参照画像としてGeminiに渡す(_generate_panel_imageのreference_images)。

    2026-09-19、長尺(16:9)+ショート2本(9:16、現代文編/古文編)への分割出力に対応するため
    以下のキーワード専用引数を追加した(すべて省略時は従来通りの単一動画の挙動):
    - out_w/out_h/image_aspect_ratio: 出力解像度とGemini生成時のアスペクト比。長尺は
      (1920,1080,"16:9")、ショートは既定の(1080,1920,"9:16")を使う想定。
    - asset_dir_name: 素材保存先ディレクトリ名(既定はscript_path.stem)。長尺(16:9)は
      解像度が違う別素材になるため専用のディレクトリ名を渡すこと。
    - scene_ids: 指定した場合、そのidのシーンだけを指定順に抽出する(ショート抽出用。
      例: 現代文編ショートは[1,2,3,4,5,6]、古文編ショートは[1,8,9,10,11,12])。
    - output_stem: 出力mp4のファイル名(既定はscript_path.stem)。
    - reuse_existing_images: Trueの場合、asset_dir内に対象scene idの画像
      (と吹き出し検出のサイドカーJSON)が既にあれば、Gemini呼び出しをせずそのまま使う
      (ショート2本が、同じ9:16素材ディレクトリを共有して画像・検出結果を再利用するための
      フラグ。単発の動画生成では「毎回新規生成」の既定方針を保つためFalseのまま)。
    - variant_label: ショートのオープニング(先頭シーン)に、音声付きの追加キャプションとして
      挿入するラベル(例:「現代語訳編」「古文訳編」)。指定した場合、先頭シーンの
      lines[]の末尾に`{"character_key":"extra","text":variant_label,"type":"caption"}`を
      追加してから処理する(2026-09-19、フィードバック「ショートのオープニングで
      現代語訳編/古文訳編のどちらか必ずわかるようにする」への対応)。
    - enable_loop_transition: Falseにすると、末尾から先頭コマへ戻るループ演出を付けない。
      opening_episode_title/ending_closing_lineのいずれかを指定した場合は自動的にFalse扱いになる。
    - opening_episode_title: 指定すると、動画の先頭に共通オープニング
      (`render_kobun_long_opening_clip`、「ことば先生【古文】」→「物語で覚える古文」→
      この引数のタイトルの順にポップイン、ページめくり効果音付き、約3秒)を追加する。
    - ending_closing_line: 指定すると、末尾のループ演出の代わりに共通エンディング
      (`render_kobun_long_ending_clip`、最初の2秒はこの引数の締めの一文、残り5秒は
      共通の次回予告画面、約7秒)を追加する。
      (2026-09-19、ユーザー指定のオープニング/エンディングテンプレート。長尺専用の想定で、
      ショートは「テンポ優先で無しでよい」とのユーザー判断のためこの2引数は渡さない)。

    既知の制約(2026-09-16時点、テスト運用中):
    - 群衆の歓声・雨音・チャイム等の環境音素材はmaterials/se, materials/bgmに未用意のため
      未実装(SEはmaterials/se/stumble.mp3のみ対応)。
    - speaker_effect="speakerphone"は簡易近似(音量を下げるのみ)で、実際の帯域フィルタ処理は
      行っていない。

    - visual_cache: 現代語版/古文版のような視覚的に同じ場面(scene_description+
      characters_in_panel一致)の画像使い回しに使うキャッシュを、呼び出し元から
      共有辞書として渡せるようにする引数(省略時はこの呼び出し内だけで完結する
      新しい辞書を使う、従来通りの挙動)。

      フィードバック(2026-09-20、ユーザー確認): 「現代文編と古文編のShorts2本で
      同じコマの画像を使ってほしい」という要件に対し、従来は各`assemble_kobun_scene_video`
      呼び出し内のローカル変数としてvisual_cacheを持っていたため、ショート2本を別々に
      呼び出す`assemble_kobun_scene_video_all_variants`では、1本目(現代文編)で
      生成した画像を2本目(古文編)が実際には共有できていなかった(ディスク上の
      再利用判定はscene idそのものが一致するファイル名でしか探さないため、
      現代文編のscene2と古文編のscene8のように視覚的にはペアでもscene idが違う
      場合は見つけられず、結果的に古文編側で見た目の異なる画像が新規生成されて
      いた)。過去にep1で「完全一致を確認済み」としていたのは、たまたま同じ
      9:16素材ディレクトリに分割機能導入前の一括生成で全12コマが既に揃っていた
      ためで、まっさらな新規話数では再現しない偶然の一致だったと判明した。
      `assemble_kobun_scene_video_all_variants`側でショート2本に同じ辞書を
      渡すことで、本当の意味で画像を共有できるようにした。
    """
    data = json.loads(script_path.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    if scene_ids is not None:
        by_id = {s["id"]: s for s in scenes}
        scenes = [by_id[sid] for sid in scene_ids]
    OUT_W, OUT_H = out_w, out_h

    image_provider = GeminiProvider(IMAGE_MODEL_NAME)
    vision_provider = GeminiProvider(VISION_TEXT_MODEL_NAME)

    asset_dir = PROJECT_ROOT / "materials" / "kobun_generated" / (asset_dir_name or script_path.stem)
    asset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        scene_clips = []
        scene_paths = []
        scene_durations = []
        scene_bgm_volumes = []
        anchor_bytes: bytes | None = None
        prev_bytes: bytes | None = None
        # 現代語版→古文版のように、同じ場面(scene_description+characters_in_panel)が
        # 台本内で2回出てくる場合に画像を使い回すためのキャッシュ(フィードバック2026-09-19:
        # 「構成が同じ意味の内容になるので、節約のためにも使う画像は同じコマに同じ画像に
        # なるようにしてください」)。bubble_boxesはlines内での位置(line_index)をキーに
        # 持つため、セリフを持つ行の位置が完全一致する場合のみ使い回す(位置がずれる場合は
        # 誤った吹き出しに文字が乗ってしまうため、安全側に倒して新規生成し直す)。
        # 呼び出し元(assemble_kobun_scene_video_all_variants)がショート2本で
        # 共有するために渡してくる場合はそちらを使い、無ければこの呼び出し限りの
        # 辞書を新規に使う(単発動画の従来通りの挙動)。
        if visual_cache is None:
            visual_cache = {}

        for scene in scenes:
            if variant_label and scene["id"] == scenes[0]["id"]:
                # ショートの先頭シーンに「現代語訳編」「古文訳編」等のラベルを追加音声/
                # 字幕として挿入する(visual_key/bubble_positionsはlinesの中身に依存しないため
                # 画像の使い回し判定には影響しない)。
                scene = {**scene, "lines": scene["lines"] + [
                    {"character_key": "extra", "text": variant_label, "emotion": "calm", "type": "caption"}
                ]}
            visual_key = json.dumps(
                {"desc": scene["scene_description"], "chars": scene["characters_in_panel"]},
                sort_keys=True, ensure_ascii=False,
            )
            bubble_positions = tuple(
                i for i, line in enumerate(scene["lines"]) if line.get("type", "dialogue") != "caption"
            )
            cached = visual_cache.get(visual_key)
            panel_path = asset_dir / f"panel{scene['id']}.png"
            detected_path = asset_dir / f"panel{scene['id']}.detected.json"
            # 吹き出しを持つ行があるのに検出サイドカーが無い場合は再利用不可(安全側に倒して
            # 新規生成へフォールバックする。サイドカー保存前の古いキャッシュ画像だけが
            # 残っているケースでbubble_boxesが空になりKeyErrorする事故を防ぐため)。
            can_reuse_from_disk = (
                reuse_existing_images and panel_path.exists()
                and (not scene["characters_in_panel"] or detected_path.exists())
            )

            if can_reuse_from_disk:
                # ショート2本(現代文編/古文編)が同じ9:16素材ディレクトリを共有し、既存の
                # 画像・検出結果をそのまま使う場合(2026-09-19、長尺/ショート分割対応)。
                panel_img = Image.open(panel_path).convert("RGB")
                img_w, img_h = panel_img.size
                if scene["characters_in_panel"]:
                    detection = json.loads(detected_path.read_text(encoding="utf-8"))
                    bubble_line_indices = [
                        i for i, line in enumerate(scene["lines"]) if line.get("type", "dialogue") != "caption"
                    ]
                    bubble_speaker_idx = [
                        next(
                            idx for idx, c in enumerate(scene["characters_in_panel"], start=1)
                            if c["key"] == scene["lines"][li]["character_key"]
                        )
                        for li in bubble_line_indices
                    ]
                    bubble_boxes, _ = _assign_bubble_boxes(
                        detection["by_speaker"], scene["characters_in_panel"],
                        bubble_line_indices, bubble_speaker_idx, img_w, img_h, scene["id"],
                        panel_img=panel_img,
                    )
                else:
                    bubble_boxes = {}
                print(f"[情報] panel{scene['id']}: 既存画像を再利用します(reuse_existing_images)。", file=sys.stderr)
                visual_cache[visual_key] = (panel_path, bubble_boxes, bubble_positions)
            elif cached is not None and cached[2] == bubble_positions:
                cached_path, bubble_boxes, _ = cached
                panel_path.write_bytes(cached_path.read_bytes())
                cached_detected = cached_path.parent / f"{cached_path.stem}.detected.json"
                if cached_detected.exists():
                    detected_path.write_bytes(cached_detected.read_bytes())
                panel_img = Image.open(panel_path).convert("RGB")
                img_w, img_h = panel_img.size
                print(
                    f"[情報] panel{scene['id']}: {cached_path.name}と同一の場面のため画像生成をスキップし、"
                    "同じ画像を再利用します。",
                    file=sys.stderr,
                )
            else:
                if cached is not None:
                    print(
                        f"[警告] panel{scene['id']}: 同一の場面と判定されましたが吹き出しを持つ行の"
                        "位置が一致しないため、画像を使い回さず新規生成します。",
                        file=sys.stderr,
                    )
                refs = []
                if anchor_bytes is not None:
                    refs.append((anchor_bytes, "image/png"))
                if prev_bytes is not None and prev_bytes != anchor_bytes:
                    refs.append((prev_bytes, "image/png"))
                panel_img, bubble_boxes = _generate_panel_image(
                    scene, image_provider, vision_provider, asset_dir,
                    reference_images=refs or None, aspect_ratio=image_aspect_ratio,
                )
                img_w, img_h = panel_img.size
                visual_cache[visual_key] = (panel_path, bubble_boxes, bubble_positions)
            scene_paths.append(panel_path)
            panel_bytes_now = panel_path.read_bytes()
            if anchor_bytes is None:
                anchor_bytes = panel_bytes_now
            prev_bytes = panel_bytes_now

            bg_scale, bg_x, bg_y = _cover_fit_transform(img_w, img_h, OUT_W, OUT_H)

            lines = scene["lines"]
            resolved_lines = []
            t_cursor = 0.0
            for i, line in enumerate(lines):
                char_key = line["character_key"]
                speaker = kvc.resolve_speaker(char_key, line["emotion"])
                wav_bytes = synthesize(
                    line["text"], character=None, emotion=None,
                    speaker=speaker, emotion_params=kvc.emotion_to_params(line["emotion"]),
                )
                wav_path = tmp_dir / f"kobun_{scene['id']}_{i}.wav"
                wav_path.write_bytes(wav_bytes)
                audio_clip = AudioFileClip(str(wav_path))
                dur = audio_clip.duration
                if line.get("speaker_effect") == "speakerphone":
                    # 簡易近似: 帯域フィルタではなく音量を下げるだけで「放送越し」感を出す
                    audio_clip = audio_clip.with_effects([MultiplyVolume(0.55)])
                resolved_lines.append(
                    {**line, "line_index": i, "audio_clip": audio_clip, "dur": dur, "start": t_cursor}
                )
                t_cursor += dur + kvc.LINE_GAP_SECONDS

            scene_dur = t_cursor - kvc.LINE_GAP_SECONDS + scene.get("final_hold_seconds", 0.0)
            scene_durations.append(scene_dur)
            scene_bgm_volumes.append(scene.get("bgm_volume", 1.0))

            # コマ内でのパン/ズームは廃止(フィードバック2026-09-16)。常に画面いっぱいの
            # 静止画として、コマの間ずっと同じ位置・大きさで表示する
            layer_clips = [
                ImageClip(str(panel_path))
                .with_duration(scene_dur)
                .resized(bg_scale)
                .with_position((bg_x, bg_y))
            ]

            extra_audio_clips = []

            if scene.get("rain"):
                rain_path = tmp_dir / f"rain_{scene['id']}.png"
                _rain_overlay_image(OUT_W, OUT_H, seed=scene["id"]).save(rain_path)
                layer_clips.append(ImageClip(str(rain_path)).with_duration(scene_dur).with_position((0, 0)))
                extra_audio_clips.append(_synth_rain_audio(scene_dur).with_start(0))

            # キャプション配置時に「まだ画面に残っている吹き出し」を避けるための記録。
            # 吹き出しはセリフを言い終えてもコマが終わるまで表示され続けるため、このコマの
            # lines[]でこのキャプションより前に出てきた吹き出しは、キャプション表示中も
            # 必ず画面に残っている(resolved_linesはlines[]の順のままstart時刻が単調増加する)。
            shown_bubble_ranges: list[tuple[int, int]] = []

            for rl in resolved_lines:
                if rl.get("type", "dialogue") == "caption":
                    cap_path = tmp_dir / f"caption_{scene['id']}_{rl['start']:.3f}.png"
                    cap_img = draw_caption_bar(OUT_W, OUT_H, rl["text"])
                    cap_img.save(cap_path)
                    # フィードバック(2026-09-19): 画面最下部だと見づらいため、もう少し上に配置。
                    # フィードバック(2026-09-20): バーが2行になり高さが増えると、まだ表示され
                    # 続けている吹き出しと重なることがあったため、衝突回避付きの
                    # `_pick_caption_y`で位置を決める。
                    cap_y = _pick_caption_y(OUT_H, cap_img.height, shown_bubble_ranges)
                    layer_clips.append(
                        ImageClip(str(cap_path))
                        .with_duration(rl["dur"])
                        .with_start(rl["start"])
                        .with_position((0, cap_y))
                    )
                    continue

                bx0, by0, bx1, by1 = bubble_boxes[rl["line_index"]]
                # カメラが静止しているため、吹き出しの位置・サイズも背景と同じ一定の
                # 変換(bg_scale/bg_x/bg_y)で一度だけ計算すればよく、時間経過でずれない
                frame_x0 = bg_x + bx0 * bg_scale
                frame_y0 = bg_y + by0 * bg_scale
                box_w = max(1, int((bx1 - bx0) * bg_scale))
                box_h = max(1, int((by1 - by0) * bg_scale))
                pos_x = min(max(0.0, frame_x0), OUT_W - box_w)
                pos_y = min(max(0.0, frame_y0), OUT_H - box_h)
                shown_bubble_ranges.append((int(pos_y), int(pos_y + box_h)))

                tag = f"text_{scene['id']}_{rl['start']:.3f}"
                text_clip, final_text_path = draw_bubble_typewriter_clip(
                    box_w, box_h, rl["text"], rl["dur"], tmp_dir, tag
                )
                # フィードバック(2026-09-19): セリフを言い終えるとすぐ吹き出しの文字が消えて
                # いたのを、次のコマに切り替わるまで(このコマが終わるまで)表示し続けるよう変更。
                hold_extra = max(0.0, scene_dur - rl["start"] - text_clip.duration)
                if hold_extra > 0.01:
                    text_clip = concatenate_videoclips(
                        [text_clip, ImageClip(str(final_text_path)).with_duration(hold_extra)], method="compose"
                    )
                layer_clips.append(text_clip.with_start(rl["start"]).with_position((pos_x, pos_y)))

            # 効果音(SE): 対象セリフの開始時点で音声を鳴らす。オノマトペの文字演出は
            # 吹き出しの上に乗って邪魔になるとのフィードバック(2026-09-19)により廃止した
            # (直すのではなく無しにする、というユーザー判断。draw_se_textは削除済み)。
            for se in scene.get("se_cues", []):
                if se.get("at_line_index") is None or se["at_line_index"] >= len(resolved_lines):
                    continue
                se_start = resolved_lines[se["at_line_index"]]["start"]
                sound_name = se.get("sound")
                if sound_name in SE_FILES:
                    vol = SE_VOLUME.get(sound_name, 0.2)
                    extra_audio_clips.append(
                        AudioFileClip(str(SE_FILES[sound_name])).with_effects([MultiplyVolume(vol)]).with_start(se_start)
                    )
                elif sound_name in _SYNTH_SE:
                    extra_audio_clips.append(_SYNTH_SE[sound_name]().with_start(se_start))

            audio_pieces = [rl["audio_clip"].with_start(rl["start"]) for rl in resolved_lines] + extra_audio_clips
            scene_audio = CompositeAudioClip(audio_pieces)
            scene_composite = CompositeVideoClip(layer_clips, size=(OUT_W, OUT_H)).with_duration(scene_dur)

            shake_after = scene.get("screen_shake_after_line")
            if shake_after is not None:
                shake_t0 = resolved_lines[shake_after]["start"] + resolved_lines[shake_after]["dur"]
                shake_len = 0.15

                def shake_pos(t, t0=shake_t0, shake_len=shake_len):
                    if t0 <= t <= t0 + shake_len:
                        amp = 14 * (1 - (t - t0) / shake_len)
                        return (random.uniform(-amp, amp), random.uniform(-amp, amp))
                    return (0, 0)

                scene_composite = CompositeVideoClip(
                    [scene_composite.with_position(shake_pos)], size=(OUT_W, OUT_H)
                ).with_duration(scene_dur)

            scene_clips.append(scene_composite.with_audio(scene_audio))

        sequence = []
        cursor = 0.0
        if opening_episode_title:
            # 長尺共通オープニング(2026-09-19、ユーザー指定テンプレート)。BGMのセグメント
            # 計算はscene_offsetsを使うため、ここでcursorを進めておけば自動的にずれない。
            opening_clip = render_kobun_long_opening_clip(OUT_W, OUT_H, opening_episode_title, tmp_dir)
            sequence.append(opening_clip)
            cursor = opening_clip.duration

        scene_offsets = []
        for idx, clip in enumerate(scene_clips):
            scene_offsets.append(cursor)
            sequence.append(clip)
            cursor += clip.duration
            if idx < len(scene_clips) - 1:
                sequence.append(_scroll_transition_clip(scene_paths[idx], scene_paths[idx + 1], OUT_W, OUT_H, tmp_dir))
                cursor += KOBUN_SCROLL_TRANSITION_SECONDS

        # ループ再生(Shorts等での自動リピート)時に、最後のコマで唐突に途切れて最初のコマに
        # 戻るのではなく、コマ間と同じスクロール演出でつながるようにする(フィードバック2026-09-19)。
        # ただし専用エンディング(ending_closing_line)を使う場合はループ演出とは排他。
        has_loop_transition = len(scene_paths) > 1 and enable_loop_transition and not ending_closing_line
        if has_loop_transition:
            sequence.append(_scroll_transition_clip(scene_paths[-1], scene_paths[0], OUT_W, OUT_H, tmp_dir))
        elif ending_closing_line:
            # 長尺共通エンディング(2026-09-19、ユーザー指定テンプレート)。前のシーンから
            # フェードで滑らかにつなぐ(コマ間のスクロール演出とは異なる、映像の締めらしい入り方)。
            ending_clip = render_kobun_long_ending_clip(OUT_W, OUT_H, ending_closing_line, tmp_dir)
            sequence.append(ending_clip.with_effects([CrossFadeIn(0.4)]))

        final = concatenate_videoclips(sequence, method="compose")

        bgm_path = BGM_DIR / "bright_debate_loop.mp3"
        if bgm_path.exists():
            bgm = AudioFileClip(str(bgm_path)).with_effects([AudioLoop(duration=final.duration)])
            bgm_segments = []
            for idx, (start, dur, vol) in enumerate(zip(scene_offsets, scene_durations, scene_bgm_volumes)):
                is_last = idx == len(scene_clips) - 1
                has_trailing_transition = (not is_last) or has_loop_transition
                seg_end = start + dur + (KOBUN_SCROLL_TRANSITION_SECONDS if has_trailing_transition else 0.0)
                seg = (
                    bgm.subclipped(start, min(seg_end, final.duration))
                    .with_effects([MultiplyVolume(KOBUN_BGM_VOLUME * vol)])
                    .with_start(start)
                )
                bgm_segments.append(seg)
            final = final.with_audio(CompositeAudioClip([final.audio, CompositeAudioClip(bgm_segments)]))

        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{output_stem or script_path.stem}.mp4"
        final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac")
        return out_path


def assemble_kobun_scene_video_all_variants(script_path: Path) -> dict[str, Path]:
    """1本の台本(現代語版+切り替え+古文版のscenes)から、長尺(16:9、台本まま)と
    ショート2本(9:16、現代文編/古文編、切り替えシーンは含めない)をまとめて組み立てる。

    2026-09-19、ユーザー方針: 「長尺は現代文編+古文編を台本のまま。長尺公開後、
    現代文編・古文編それぞれを別のショートとして出す。ショートの2本は同じコマ画像を
    共有する。長尺とショートは画像サイズが違うのでそれぞれ別サイズで生成してよい」に対応。

    scenes[]の各要素に"variant"キー(値: "opening"(全出力共通の冒頭)/"modern"(現代語版)/
    "transition"(長尺内の切り替えのみ、ショートには含めない)/"classical"(古文版))が
    必要。台本側にこの分類が無い場合はValueErrorを送出する(誤って全シーンを1本の
    ショートに詰め込む事故を防ぐため、機械的な範囲推測はしない)。

    ショート2本は同じ9:16素材ディレクトリ(script_path.stemそのまま)を共有し、
    2本目の生成時は`reuse_existing_images=True`で1本目が生成した画像・検出結果を
    再利用する(Gemini呼び出しの節約と、見た目の完全一致を両立させるため)。

    フィードバック(2026-09-20、ユーザー確認): `reuse_existing_images=True`だけでは、
    ディスク上にそのscene idそのままのファイル(例: 古文編のscene8用panel8.png)が
    無い限り再利用判定が効かない。現代文編のscene2と古文編のscene8のように
    「視覚的には同じ場面だがscene idが違う」ペアは、まっさらな新規話数では
    2本目の生成時点でまだpanel8.pngが存在しないため見つけられず、結果的に
    現代文編と見た目の異なる画像が新規生成されてしまっていた(ep1で過去に
    「完全一致を確認済み」としていたのは、分割機能導入前の一括生成で9:16素材が
    たまたま全12コマ揃っていたことによる偶然の一致だった)。
    ショート2本の呼び出しに同じ`visual_cache`辞書を明示的に共有させることで、
    1本目(現代文編)がscene2で生成した画像を、2本目(古文編)のscene8が
    「visual_key(scene_description+characters_in_panel)が一致する」と判定して
    正しく使い回せるようにした。
    """
    data = json.loads(script_path.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    variants = {s["id"]: s.get("variant") for s in scenes}
    if any(v is None for v in variants.values()):
        missing = [sid for sid, v in variants.items() if v is None]
        raise ValueError(
            f"scenes[]に'variant'フィールドが無いシーンがあります(id={missing})。"
            "長尺/ショート分割にはopening/modern/transition/classicalの分類が必須です。"
        )

    opening_ids = [sid for sid, v in variants.items() if v == "opening"]
    modern_ids = [sid for sid, v in variants.items() if v == "modern"]
    classical_ids = [sid for sid, v in variants.items() if v == "classical"]

    ending_closing_line = data.get("ending_closing_line")
    if not ending_closing_line:
        raise ValueError(
            "台本に'ending_closing_line'(長尺エンディング冒頭2秒に表示する、この話の締めの"
            "一文)がありません。各話ごとに設定してください(例:「かくして、トウマは職員室へ"
            "連行されけり。」)。"
        )

    outputs: dict[str, Path] = {}

    print("[情報] 長尺版(16:9)を組み立てます...", file=sys.stderr)
    outputs["long"] = assemble_kobun_scene_video(
        script_path,
        out_w=1920, out_h=1080, image_aspect_ratio="16:9",
        asset_dir_name=f"{script_path.stem}_16x9",
        output_stem=f"{script_path.stem}_long",
        enable_loop_transition=False,
        opening_episode_title=data.get("title", script_path.stem),
        ending_closing_line=ending_closing_line,
    )

    # 現代文編/古文編のショート2本で、視覚的に同じ場面(scene idは異なる)の画像を
    # 確実に共有するため、2つの呼び出しに同じ辞書を渡す(2026-09-20)。
    short_visual_cache: dict[str, tuple[Path, dict, tuple]] = {}

    print("[情報] ショート版(9:16・現代文編)を組み立てます...", file=sys.stderr)
    outputs["short_modern"] = assemble_kobun_scene_video(
        script_path,
        scene_ids=opening_ids + modern_ids,
        output_stem=f"{script_path.stem}_short_modern",
        reuse_existing_images=True,
        visual_cache=short_visual_cache,
    )

    print("[情報] ショート版(9:16・古文編)を組み立てます...", file=sys.stderr)
    outputs["short_classical"] = assemble_kobun_scene_video(
        script_path,
        scene_ids=opening_ids + classical_ids,
        output_stem=f"{script_path.stem}_short_classical",
        reuse_existing_images=True,
        visual_cache=short_visual_cache,
    )

    return outputs


def main():
    if len(sys.argv) < 2:
        print("使い方: python src/assemble_video.py <台本JSONのパス>")
        sys.exit(1)
    script_path = Path(sys.argv[1])
    data = json.loads(script_path.read_text(encoding="utf-8"))
    if data.get("format") == "kobun_5koma":
        out_path = assemble_kobun_panel_video(script_path)
    elif data.get("format") == "kobun_scene_video":
        # scenes[]に"variant"分類(opening/modern/transition/classical)があれば、
        # 長尺(16:9)+ショート2本(9:16、現代文編/古文編)をまとめて組み立てる
        # (2026-09-19、長尺/ショート分割方針への対応)。無ければ従来通り単一動画。
        if all("variant" in s for s in data["scenes"]):
            out_paths = assemble_kobun_scene_video_all_variants(script_path)
            for name, p in out_paths.items():
                print(f"出力先({name}): {p}")
            return
        out_path = assemble_kobun_scene_video(script_path)
    else:
        out_path = assemble(script_path)
    print(f"出力先: {out_path}")


if __name__ == "__main__":
    main()
