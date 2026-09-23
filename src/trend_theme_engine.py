"""YouTube急上昇動画を、Geminiでディベートのお題に変換してテーマプールに追加する
（[[project-theme-engine]]の当初設計「4ソース」のうち、トレンド分析を担当）。

Google Trends（pytrends）は非公式APIで不安定になりやすいため採用せず、既存の
YouTube OAuth認証をそのまま使えて公式にサポートされているYouTube急上昇
（videos.list(chart="mostPopular")、1件あたり1 unitと軽量）を使う（ユーザー選択、2026-07-15）。

急上昇動画のタイトルそのものは「日常のくだらないディベートのお題」の形をしていない
ことが多いため、ingest_ideas.pyと同じis_usable+theme+categoryゲートでGeminiに
変換させる。政治・宗教・時事ニュース・実在人物の名指し等は既存方針通り除外する。

GitHub Actionsで日次実行する想定。処理済みvideo_idはtrend_theme_state.jsonに記録し、
急上昇ランキングの重複（前日と同じ動画が残っているケース）による二重変換を防ぐ。
"""

import json
import sys
from pathlib import Path

from ai_provider import AITextProvider, get_text_provider
from ingest_ideas import IDEA_SCHEMA_WITH_CATEGORY
from theme_pool import add_candidate
from youtube_upload import get_authenticated_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "scripts_templates" / "trend_theme_state.json"
MODEL_NAME = "gemini-flash-lite-latest"

MAX_TRENDING_VIDEOS = 15
# 状態ファイルの肥大化を防ぐため、直近何件分のvideo_idだけ保持するか
STATE_HISTORY_LIMIT = 300

SYSTEM_PROMPT = (
    "あなたは、YouTubeの急上昇動画のタイトル・概要を『日常のくだらないことをコミカルに"
    "ディベートする動画』のお題に変換するアシスタントです。\n"
    "入力される動画そのものの内容を要約・紹介するのではなく、そのテーマが属する日常的な"
    "ジャンル・話題（例: 食べ物、ガジェット、生活習慣等）から着想を得て、当チャンネル独自の"
    "「〜べきか」「〜はどっちが正しいか」形式のお題を新しく作ってください。\n"
    "- 特定個人（動画の投稿者・出演者含む）への言及、政治・宗教等の重いテーマ、"
    "実際に起きたニュース・事件そのものを主題にすることは避けてください。\n"
    "- 医療・健康法・投資/金融・法律相談に関する内容も、誤情報が広告制限につながりやすい"
    "ジャンルのため is_usable=false にしてください。\n"
    "- 動画の具体的な内容（誰が何をしたか等）に立ち入らず、あくまで一般化された日常の話題に"
    "留めてください。着想元にできそうな話題が無ければ is_usable=false にしてください。\n"
    "- themeが最も当てはまるcategoryも選んでください。"
)


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed_video_ids": []}


def _save_state(state: dict) -> None:
    state["processed_video_ids"] = state.get("processed_video_ids", [])[-STATE_HISTORY_LIMIT:]
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_trending_videos(youtube) -> list[dict]:
    resp = (
        youtube.videos()
        .list(part="snippet", chart="mostPopular", regionCode="JP", maxResults=MAX_TRENDING_VIDEOS)
        .execute()
    )
    return resp.get("items", [])


def _trend_score(rank: int, total: int) -> float:
    """順位ベースの正規化スコア（1位=1.0に近く、下位ほど下がる）。"""
    if total <= 1:
        return 1.0
    return round(1.0 - (rank / total), 3)


def _to_theme(provider: AITextProvider, title: str, description: str) -> tuple[str, str] | None:
    user_prompt = f"動画タイトル: {title}\n概要（先頭のみ）: {(description or '')[:200]}"
    data = provider.generate_json(SYSTEM_PROMPT, user_prompt, IDEA_SCHEMA_WITH_CATEGORY)
    if data.get("is_usable") and data.get("theme"):
        return data["theme"].strip(), data.get("category", "その他")
    return None


def main():
    youtube = get_authenticated_service()
    videos = fetch_trending_videos(youtube)
    if not videos:
        print("急上昇動画が取得できませんでした。")
        return

    state = _load_state()
    processed = set(state.get("processed_video_ids", []))

    provider = get_text_provider(MODEL_NAME)
    added = 0
    total = len(videos)

    for rank, video in enumerate(videos):
        video_id = video["id"]
        if video_id in processed:
            continue
        processed.add(video_id)

        snippet = video.get("snippet", {})
        title = snippet.get("title", "")
        if not title:
            continue

        try:
            result = _to_theme(provider, title, snippet.get("description", ""))
        except Exception as e:
            print(f"[警告] Gemini呼び出しに失敗しました（{title}）: {e}", file=sys.stderr)
            continue

        if result:
            theme, category = result
            trend_score = _trend_score(rank, total)
            add_candidate(theme, source="trend", category=category, trend_score=trend_score)
            added += 1
            print(f"[トレンドテーマ] 「{title}」→「{theme}」(score={trend_score}) を追加しました。")

    state["processed_video_ids"] = list(processed)
    _save_state(state)
    print(f"完了: {added}件のトレンド連動テーマをテーマプールに追加しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "trend_theme_engine.py")
