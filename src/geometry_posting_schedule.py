"""物理演算バトルチャンネル(4ch目)の投稿タイミング判定。

投稿枠は「01:00/09:00/21:00 JSTの固定3枠」で確定済みであり、
1ch目のposting_schedule.py(投稿時刻そのものを探索・収束させる仕組み)とは解く問題が違う。
そのためモジュール自体は流用せず、以下の設計パターンだけを踏襲した軽量版として新規作成した:

- GitHub Actionsのscheduleトリガーは低頻度リポジトリだと数時間規模で遅延・間引きされることが
  実測で分かっている(過去の運用記録で複数回同じ現象を確認)。そのため主たる起動経路は外部cronサービスからの
  repository_dispatchとし、schedule:は保険用フォールバックとして残す。
- 判定は「ちょうどその時刻」ではなく「予定時刻に達していて、その枠がまだ未消化ならOK」という
  後追い可能な形にする(遅延・間引きされても取りこぼさない)。
- 状態ファイルで「今日何本消化したか」を記録し、二重投稿を防ぐ。
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "scripts_templates" / "geometry_posting_schedule_state.json"

# 2026-09-21、ユーザー指示「投稿時刻のロールバック検証」: 2026-09-13の米国タイムゾーン向け
# 変更(01:00/09:00/21:00 JST)以降、視聴維持率の低下トレンドが観測されたため、原因切り分けの
# ために一時的にJST向け(09:00/14:00/19:00 JST、2026-09-13以前の設定)へ戻す。数日間データを
# 収集して維持率の変化を確認する検証目的の変更であり、投稿時刻が最終的にどちらになるかは
# この検証結果を見て判断する(今回のリセットバッチの一部だが、以降の変更は1件ずつ検証する方針)。
TARGET_HOURS = [9, 14, 19]

JST = timezone(timedelta(hours=9))


def _target_hours_for(today: str) -> list[int]:
    return TARGET_HOURS


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


def should_publish_now(now: datetime | None = None) -> bool:
    """今この瞬間に投稿すべき未消化の枠があるかどうかを判定する。
    外部cronサービスからの高頻度(15〜30分おき想定)呼び出しから使う想定。"""
    now = now or datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    state = _load_state()

    if state.get("date") != today:
        state = {"date": today, "posted_count": 0}
        _save_state(state)

    target_hours = _target_hours_for(today)
    posted_count = state.get("posted_count", 0)
    if posted_count >= len(target_hours):
        return False
    next_target_hour = target_hours[posted_count]
    # GitHub Actionsのスケジュール実行は遅延・欠落することがあるため、「ちょうどその時刻」の
    # 厳密一致ではなく「予定時刻に達していればそれ以降のどの実行でも追いつける」ようにする。
    return now.hour >= next_target_hour


def mark_posted(now: datetime | None = None) -> None:
    """実際に1本投稿できたら呼ぶ。手動実行(workflow_dispatch)経由の投稿も、その日の
    消化本数としてカウントする(自動枠が後から重複投稿しないようにするため)。"""
    now = now or datetime.now(JST)
    state = _load_state()
    state["date"] = now.strftime("%Y-%m-%d")
    state["posted_count"] = state.get("posted_count", 0) + 1
    _save_state(state)
