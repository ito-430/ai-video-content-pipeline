"""テーマ選定エンジンの中核（[[project-theme-engine]]の当初設計を実装）。

複数ソース（Discord/季節イベント/コメント欄/YouTube急上昇/競合チャンネル）から
集めた候補テーマを、単純なFIFOキューではなく、スコアリングして最良の1件を選ぶ
「プール」として管理する。

score = trend_score*0.4 + source_bonus + diversity_bonus(直近3本と異カテゴリ+0.3)
        - staleness_penalty(14日超放置-0.2)

候補は消費（used=true）してもプールから削除せず残す（後から見返せるようにするため）。
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = PROJECT_ROOT / "scripts_templates" / "theme_pool.json"
PUBLISHED_VIDEOS_PATH = PROJECT_ROOT / "scripts_templates" / "published_videos.json"

JST = timezone(timedelta(hours=9))

# 元設計（project_theme_engine memory）のカテゴリ例を踏襲した固定タクソノミー。
CATEGORIES = ["生活習慣", "お金", "人間関係", "食べ物", "デジタル・SNS", "仕事・学校", "その他"]

# ソースごとの基本ボーナス。trendはtrend_score自体で既に評価されるためボーナス無し。
# manualは人間が直接選んだ意思を尊重し、他のソースより高めに設定する。
SOURCE_BONUS = {
    "manual": 0.5,
    "discord": 0.2,
    "comment": 0.15,
    "competitor": 0.1,
    "seasonal": 0.1,
    "trend": 0.0,
}

DIVERSITY_BONUS = 0.3
STALENESS_PENALTY = 0.2
STALENESS_DAYS = 14
RECENT_CATEGORY_WINDOW = 3


def _load_pool() -> list[dict]:
    if not POOL_PATH.exists():
        return []
    try:
        return json.loads(POOL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_pool(pool: list[dict]) -> None:
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")


def add_candidate(text: str, source: str, category: str, trend_score: float = 0.0) -> dict:
    """新しい候補テーマをプールに追加する。追加したcandidate自体を返す。"""
    if category not in CATEGORIES:
        category = "その他"
    pool = _load_pool()
    candidate = {
        "id": f"{datetime.now(JST).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        "text": text,
        "source": source,
        "category": category,
        "trend_score": max(0.0, min(1.0, trend_score)),
        "created_at": datetime.now(JST).isoformat(),
        "used": False,
    }
    pool.append(candidate)
    _save_pool(pool)
    return candidate


def recent_used_categories(n: int = RECENT_CATEGORY_WINDOW) -> list[str]:
    """直近n件の公開動画のカテゴリを新しい順に返す（多様性ガードの材料）。"""
    if not PUBLISHED_VIDEOS_PATH.exists():
        return []
    try:
        entries = json.loads(PUBLISHED_VIDEOS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    categories = [e.get("theme_category") for e in entries if e.get("theme_category")]
    return categories[-n:]


def _staleness_days(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        return 0.0
    now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
    return (now - created).total_seconds() / 86400


def score_candidate(candidate: dict, recent_categories: list[str]) -> float:
    trend_component = candidate.get("trend_score", 0.0) * 0.4
    source_component = SOURCE_BONUS.get(candidate.get("source"), 0.0)
    diversity_component = DIVERSITY_BONUS if candidate.get("category") not in recent_categories else 0.0
    staleness_component = STALENESS_PENALTY if _staleness_days(candidate.get("created_at", "")) > STALENESS_DAYS else 0.0
    return trend_component + source_component + diversity_component - staleness_component


def select_and_consume_best() -> dict | None:
    """未消費(used=false)の候補の中から最高スコアの1件を選び、used=trueにして返す。
    候補が無ければNone。"""
    pool = _load_pool()
    unused = [c for c in pool if not c.get("used")]
    if not unused:
        return None

    recent_categories = recent_used_categories()
    best = max(unused, key=lambda c: score_candidate(c, recent_categories))
    best["used"] = True
    _save_pool(pool)
    print(
        f"[theme_pool] 選定: 「{best['text']}」"
        f"(source={best['source']}, category={best['category']}, "
        f"score={score_candidate(best, recent_categories):.2f})"
    )
    return best


def count_unused() -> int:
    return sum(1 for c in _load_pool() if not c.get("used"))
