"""競合チャンネルの人気動画を、着想元としてテーマプールに反映する
（当初設計の「4ソース」のうち、競合人気動画分析を担当）。

competitor_style_analysis.py が既に取得済みの競合動画タイトル
（scripts_templates/competitor_video_titles.json）を読むだけで、新規のYouTube API
呼び出しは行わない（quota消費ゼロ）。

**盗用防止が最優先**: 競合動画のタイトル・内容をそのまま転用・要約するのではなく、
その動画が扱っている一般的な話題・ジャンルから着想を得て、当チャンネル独自の
「〜べきか」形式のお題に書き改める（ingest_comment_ideas.pyのコメント転用時と同じ方針）。

GitHub Actionsで discover-competitors.yml の既存ステップ（週次）に相乗りして実行する想定。
"""

import json
import sys
from pathlib import Path

from ai_provider import AITextProvider, get_text_provider
from ingest_ideas import IDEA_SCHEMA_WITH_CATEGORY
from theme_pool import add_candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_TITLES_PATH = PROJECT_ROOT / "scripts_templates" / "competitor_video_titles.json"
STATE_PATH = PROJECT_ROOT / "scripts_templates" / "competitor_theme_state.json"
MODEL_NAME = "gemini-flash-lite-latest"

# 上位何件（再生数順）を着想元として使うか。多すぎるとAPIコストが増えるため絞る。
TOP_N_VIDEOS = 10
STATE_HISTORY_LIMIT = 300

SYSTEM_PROMPT = (
    "あなたは、他チャンネルの人気動画タイトルを『日常のくだらないことをコミカルに"
    "ディベートする動画』のお題作りの着想元として使うアシスタントです。\n"
    "**最重要**: 入力されるタイトルの内容をそのまま転用・要約・翻案することは絶対に禁止です。"
    "そのタイトルが扱っている一般的な話題・ジャンル（例: 特定の食べ物、特定の生活習慣等）だけを"
    "参考にし、当チャンネル独自の「〜べきか」「〜はどっちが正しいか」形式のお題を新しく"
    "作ってください。元動画を見た人が「これはあの動画の話だ」と特定できてしまうような"
    "固有名詞・具体的なエピソードは含めないこと。\n"
    "- 政治・宗教等の重いテーマ、実際に起きたニュース・事件そのものを主題にすることは"
    "避けてください。\n"
    "- 医療・健康法・投資/金融・法律相談に関する内容も、誤情報が広告制限につながりやすい"
    "ジャンルのため is_usable=false にしてください。\n"
    "- 着想元にできそうな一般的な話題が見つからない場合は is_usable=false にしてください。\n"
    "- themeが最も当てはまるcategoryも選んでください。"
)


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed_titles": []}


def _save_state(state: dict) -> None:
    state["processed_titles"] = state.get("processed_titles", [])[-STATE_HISTORY_LIMIT:]
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_theme(provider: AITextProvider, title: str) -> tuple[str, str] | None:
    data = provider.generate_json(SYSTEM_PROMPT, f"動画タイトル: {title}", IDEA_SCHEMA_WITH_CATEGORY)
    if data.get("is_usable") and data.get("theme"):
        return data["theme"].strip(), data.get("category", "その他")
    return None


def main():
    if not VIDEO_TITLES_PATH.exists():
        print("競合動画タイトルが見つかりません。先に competitor_style_analysis.py を実行してください。", file=sys.stderr)
        return

    videos = json.loads(VIDEO_TITLES_PATH.read_text(encoding="utf-8"))
    if not videos:
        print("競合動画タイトルが空です。")
        return

    top_videos = sorted(videos, key=lambda v: v.get("view_count", 0), reverse=True)[:TOP_N_VIDEOS]

    state = _load_state()
    processed = set(state.get("processed_titles", []))

    provider = get_text_provider(MODEL_NAME)
    added = 0

    for video in top_videos:
        title = video.get("title", "")
        if not title or title in processed:
            continue
        processed.add(title)

        try:
            result = _to_theme(provider, title)
        except Exception as e:
            print(f"[警告] Gemini呼び出しに失敗しました（{title}）: {e}", file=sys.stderr)
            continue

        if result:
            theme, category = result
            add_candidate(theme, source="competitor", category=category)
            added += 1
            print(f"[競合連動テーマ] 「{title}」から着想 →「{theme}」を追加しました。")

    state["processed_titles"] = list(processed)
    _save_state(state)
    print(f"完了: {added}件の競合連動テーマをテーマプールに追加しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "competitor_theme_source.py")
