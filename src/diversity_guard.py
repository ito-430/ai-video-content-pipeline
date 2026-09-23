"""量産型コンテンツ対策: テーマ・オチ・導入パターンの多様性ガードと新規性チェック
。

YouTubeの「量産型コンテンツ」ポリシー（2025年7月改称、2026年6月にテンプレート化された
構成の明示的な収益化不可例が追加）を踏まえ、以下の機構を提供する。

1. 直近本数の構造パターン（締めのパターン等）が連続しすぎないよう重み付けで抑制する
2. 生成した台本のテーマ・切り口が直近の投稿と似すぎていないか判定し、
   閾値を超える場合は1回だけ再生成する。判定は2段階:
   a. 安価な文字列類似度（difflib）による粗い足切り。ほぼ同一の表現をLLM呼び出し
      無しで即座に検出する（コスト削減、かつLLMの判定ブレに依存しない確実な検出）
   b. a.を通過したものは、別のGemini呼び出しで意味的な類似度を自己採点させる
      （表現が違っても対立の構造・切り口が同じ場合を検出する）

2026-07-14、1日の投稿本数を1→2（探索フェーズのみ）に増やしたことに伴い、
直近本数の参照ウィンドウ(RECENT_WINDOW)を15→30に拡大した（投稿頻度が上がった分、
同じ「直近15本」でもカバーする暦日数が半分になってしまうため、実質的な検証期間を
それまでと同程度に保つ）。
"""

import difflib
import json
from pathlib import Path

from ai_provider import get_text_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts_templates" / "scripts"

RECENT_WINDOW = 30  # 直近何本を「最近の傾向」として見るか（2026-07-14: 15→30に拡大）
NOVELTY_MODEL_NAME = "gemini-flash-lite-latest"
NOVELTY_SIMILARITY_THRESHOLD = 0.75  # これ以上で「似すぎている」と判定（0〜1）
QUICK_DUPLICATE_THRESHOLD = 0.85  # 文字列としてほぼ同一とみなす閾値（LLM呼び出し前の足切り）

NOVELTY_SCHEMA = {
    "type": "object",
    "properties": {
        "max_similarity": {
            "type": "number",
            "description": "直近のテーマ一覧の中で最も似ているものとの類似度(0.0〜1.0)。0=全く違う、1=ほぼ同じ",
        },
        "most_similar_theme": {"type": "string", "description": "最も似ていたテーマ（無ければ空文字列）"},
    },
    "required": ["max_similarity", "most_similar_theme"],
}

NOVELTY_SYSTEM_PROMPT = (
    "あなたは、YouTubeディベート動画のテーマ・切り口が過去作とマンネリ化していないかを"
    "判定するアシスタントです。新しいテーマ・切り口と、直近投稿したテーマの一覧を渡すので、"
    "最も似ている過去テーマとの類似度を0.0〜1.0で判定してください。表現が違っても"
    "「対立の構造・切り口」が実質同じなら高い類似度としてください。"
)


def load_recent_scripts(n: int = RECENT_WINDOW) -> list[dict]:
    """直近n本の台本JSONを新しい順で読み込む（ファイル名が時刻プレフィックスのため、
    ファイル名の降順=投稿の新しい順になる）。"""
    if not SCRIPTS_DIR.exists():
        return []
    paths = sorted(SCRIPTS_DIR.glob("*.json"), key=lambda p: p.name, reverse=True)[:n]
    scripts = []
    for p in paths:
        try:
            scripts.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return scripts


def suppressed_choice(candidates: list[str], recent_values: list[str], max_streak: int = 2) -> list[str]:
    """recent_valuesの末尾がmax_streak回連続で同じ値の場合、その値を候補から除外する
    （抽選プールを絞ることで、同じパターンが3回以上連続するのを防ぐ）。
    全候補が除外されてしまう場合は元のcandidatesをそのまま返す（安全側）。"""
    if len(recent_values) < max_streak:
        return candidates
    tail = recent_values[-max_streak:]
    if len(set(tail)) == 1 and tail[0] in candidates:
        filtered = [c for c in candidates if c != tail[0]]
        if filtered:
            return filtered
    return candidates


def recent_closing_styles(recent_scripts: list[dict]) -> list[str]:
    """古い順に並べ直して返す（suppressed_choiceは「末尾=直近」を前提にしているため）。"""
    return [s["closing_style"] for s in reversed(recent_scripts) if s.get("closing_style")]


def quick_duplicate_check(theme: str, recent_themes: list[str]) -> str | None:
    """LLM呼び出し前の安価な文字列類似度チェック（difflib）。ほぼ同一の表現を検出したら、
    その類似テーマ文字列を返す（無ければNone）。表現の細かい違いはLLM側の意味的判定に
    任せるため、ここでは「ほぼ同一の文字列」という粗い一致だけを対象にする。"""
    for recent_theme in recent_themes:
        if not recent_theme:
            continue
        ratio = difflib.SequenceMatcher(None, theme, recent_theme).ratio()
        if ratio >= QUICK_DUPLICATE_THRESHOLD:
            return recent_theme
    return None


def check_novelty(theme: str, angle: str, recent_scripts: list[dict]) -> dict:
    """直近のテーマ一覧と比較し、最大類似度を判定する。
    まず安価な文字列類似度で足切りし、それを通過したものだけLLMで意味的に自己採点させる。"""
    recent_themes = [s.get("theme", "") for s in recent_scripts if s.get("theme")]
    if not recent_themes:
        return {"max_similarity": 0.0, "most_similar_theme": ""}

    quick_hit = quick_duplicate_check(theme, recent_themes)
    if quick_hit:
        return {"max_similarity": 1.0, "most_similar_theme": quick_hit}

    provider = get_text_provider(NOVELTY_MODEL_NAME)
    user_prompt = (
        f"新しいテーマ: {theme}\n新しい切り口: {angle}\n\n"
        f"直近のテーマ一覧:\n" + "\n".join(f"- {t}" for t in recent_themes)
    )
    return provider.generate_json(NOVELTY_SYSTEM_PROMPT, user_prompt, NOVELTY_SCHEMA, temperature=0.1)


def is_too_similar(novelty_result: dict) -> bool:
    return novelty_result.get("max_similarity", 0.0) >= NOVELTY_SIMILARITY_THRESHOLD
