"""週次の人手確認サンプリング。

自動判定だけに頼らず、直近1週間に投稿した動画からランダムに1〜2本を選び、
Discordの#アラートに「人手確認をお願いします」という形で通知する。
判定・除外は行わない（あくまで人間が実際に見るきっかけを作るだけ）。

前回投稿したメッセージに✅リアクションが付いたかどうかを今回の実行時にチェックし、
「人間が実際に確認したかどうか」を human_review_log.py へ記録する（項目8: 人間関与ログ）。

GitHub Actionsで週1回呼ぶ想定。
"""

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from discord_client import get_reaction_users, post_message
from human_review_log import CONFIRM_EMOJI, record_review
from video_log import load_entries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "scripts_templates" / "human_review_state.json"

JST = timezone(timedelta(hours=9))
SAMPLE_SIZE = 2
LOOKBACK_DAYS = 7
CHANNEL_NAME = "アラート"


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _recent_entries(entries: list[dict], now: datetime) -> list[dict]:
    cutoff = now - timedelta(days=LOOKBACK_DAYS)
    recent = []
    for e in entries:
        try:
            posted_at = datetime.fromisoformat(e["posted_at"])
        except (KeyError, ValueError):
            continue
        if posted_at >= cutoff:
            recent.append(e)
    return recent


def _check_previous_review(state: dict) -> None:
    """前回サンプリング時のメッセージに✅が付いたかを確認し、人間関与ログに記録する。"""
    prev_message_id = state.get("message_id")
    prev_video_ids = state.get("video_ids", [])
    if not prev_message_id:
        return
    try:
        reactors = get_reaction_users(CHANNEL_NAME, prev_message_id, CONFIRM_EMOJI)
    except Exception as e:
        print(f"[警告] 前回サンプリングのリアクション確認に失敗しました: {e}", file=sys.stderr)
        return
    confirmed = bool(reactors)
    record_review(prev_video_ids, confirmed, note="weekly_content_sampling経由の週次人手確認")
    print(f"前回サンプリング({prev_video_ids})の確認状況を記録しました: confirmed={confirmed}")


def main():
    now = datetime.now(JST)
    state = _load_state()
    _check_previous_review(state)

    entries = load_entries()
    recent = _recent_entries(entries, now)

    if not recent:
        print("直近1週間に投稿された動画がないため、今回はサンプリングをスキップします。")
        _save_state({})
        return

    sample = random.sample(recent, k=min(SAMPLE_SIZE, len(recent)))
    lines = [
        "🔍 週次サンプリング: 以下の動画の人手確認をお願いします。"
        f"（確認したら{CONFIRM_EMOJI}でリアクションしてください。次回実行時に確認記録として残ります）"
    ]
    for e in sample:
        lines.append(f"- {e.get('title', '(タイトル不明)')} (https://youtu.be/{e['video_id']}) — {e.get('theme', '')}")

    response = post_message(CHANNEL_NAME, "\n".join(lines))
    _save_state({"message_id": response["id"], "video_ids": [e["video_id"] for e in sample]})
    print(f"{len(sample)}件をサンプリングし、Discordへ通知しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "weekly_content_sampling.py")
