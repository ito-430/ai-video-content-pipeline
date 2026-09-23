"""緊急停止・限定公開化フロー。

想定トリガー:
1. 運営者がDiscordの「システムからの要求」チャンネルに `!emergency_stop <動画IDまたはURL>` と
   投稿する（手動、本モジュールが実装する範囲）
2. 事後の再点検バッチが問題を検出した場合（NGワードチェック実装後に接続予定。項目5参照）

いずれの場合も、この関数群は「限定公開への切り替え」と「Discord通知」のみを行い、
削除や再公開の自動判断は行わない（誤検知の可能性があるため、最終判断は運営者に委ねる）。

GitHub Actionsのスケジュール実行で定期的に呼ぶ想定（ingest_ideas.py等と同じ構成）。
処理済みメッセージIDは scripts_templates/emergency_state.json に記録し、二重処理を防ぐ。
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from discord_client import add_reaction, get_recent_messages, post_message
from video_log import load_entries
from youtube_upload import set_video_privacy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "scripts_templates" / "emergency_state.json"
COMMAND_CHANNEL = "システムからの要求"
EMERGENCY_STOP_PREFIX = "!emergency_stop"

JST = timezone(timedelta(hours=9))

# youtu.be/<id> ・ watch?v=<id> ・ 生のvideo_id(11文字)のいずれにもマッチする
_VIDEO_ID_RE = re.compile(r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$")


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


def extract_video_id(text: str) -> str | None:
    """動画URL（youtu.be/xxx、watch?v=xxx）または生のvideo_idからvideo_idを取り出す。"""
    m = _VIDEO_ID_RE.search(text.strip())
    if not m:
        return None
    return m.group(1) or m.group(2)


def emergency_stop(video_id: str, reason: str) -> None:
    """該当動画を限定公開に切り替え、Discordへ即時通知する（削除・再公開はしない）。"""
    set_video_privacy(video_id, "unlisted")
    entries = {e["video_id"]: e for e in load_entries()}
    title = entries.get(video_id, {}).get("title", "(タイトル不明。ログに記録がない動画IDの可能性があります)")
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    post_message(
        "アラート",
        "🚨 緊急停止: 動画を限定公開に切り替えました。\n"
        f"動画: {title} (https://youtu.be/{video_id})\n"
        f"理由: {reason}\n"
        f"日時: {now} JST\n"
        "※ 削除・再公開はしていません。今後の対応をご判断ください。",
    )


def check_commands() -> None:
    """「システムからの要求」チャンネルの新着メッセージから!emergency_stopコマンドを検出・実行する。"""
    state = _load_state()
    last_seen = state.get("last_seen_message_id")

    messages = get_recent_messages(COMMAND_CHANNEL, after_id=last_seen, limit=50)
    if not messages:
        print("新着コマンドはありません。")
        return

    latest_id = last_seen
    for msg in messages:
        latest_id = msg["id"]
        if msg.get("author", {}).get("bot"):
            continue  # Bot自身の投稿（通知等）は無視
        content = (msg.get("content") or "").strip()
        if not content.startswith(EMERGENCY_STOP_PREFIX):
            continue

        arg = content[len(EMERGENCY_STOP_PREFIX):].strip()
        video_id = extract_video_id(arg)
        if not video_id:
            post_message(COMMAND_CHANNEL, f"⚠️ 動画ID/URLを認識できませんでした: `{content}`")
            _try_react(msg["id"], "❌")
            continue

        try:
            emergency_stop(video_id, reason=f"Discordコマンドによる手動緊急停止（実行者: {msg.get('author', {}).get('username', '不明')}）")
            _try_react(msg["id"], "✅")
        except Exception as e:
            print(f"[警告] 緊急停止の実行に失敗しました（video_id={video_id}）: {e}", file=sys.stderr)
            post_message(COMMAND_CHANNEL, f"⚠️ 緊急停止の実行に失敗しました（video_id={video_id}）: {e}")
            _try_react(msg["id"], "❌")

    state["last_seen_message_id"] = latest_id
    _save_state(state)


def _try_react(message_id: str, emoji: str) -> None:
    try:
        add_reaction(COMMAND_CHANNEL, message_id, emoji)
    except Exception:
        pass  # リアクション失敗は致命的ではないので無視


def main():
    check_commands()


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "emergency_control.py")
