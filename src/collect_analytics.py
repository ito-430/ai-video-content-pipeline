"""投稿から7日経過した動画のYouTube Analyticsデータを収集し、Google Sheetsに記録する。

「動画データ」タブ（7日間累計の実測パフォーマンス）と「投稿時刻実験」タブ
（24h/7d再生数・投稿時刻の実験データ）の両方を1回の収集で埋める。
推定収益(estimatedRevenue)はチャンネルが収益化されていないと取得できないため、
失敗時は空欄のまま処理を続ける。
"""

from datetime import datetime, timedelta

from pattern_pool import record_performance
from sheets_client import CHANNEL_SHEET_ID, append_row
from video_log import load_entries, save_entries
from youtube_upload import get_analytics_service

MATURITY_DAYS = 7
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


def _query_metrics(analytics, video_id: str, start_date: str, end_date: str, metrics: str) -> list:
    resp = (
        analytics.reports()
        .query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=metrics,
            filters=f"video=={video_id}",
        )
        .execute()
    )
    rows = resp.get("rows")
    if not rows:
        return [0] * len(metrics.split(","))
    return rows[0]


def _fetch_revenue(analytics, video_id: str, start_date: str, end_date: str) -> str:
    try:
        row = _query_metrics(analytics, video_id, start_date, end_date, "estimatedRevenue")
        return row[0]
    except Exception as e:
        print(f"[情報] {video_id} の推定収益は取得できませんでした（未収益化の可能性）: {e}")
        return ""


def collect_for_entry(analytics, entry: dict) -> float:
    """Sheetsへの記録に加え、7日間の平均視聴維持率(%)を返す
    （呼び出し側でパターンプールのパフォーマンスフィードバックに使う）。"""
    video_id = entry["video_id"]
    posted_at = datetime.fromisoformat(entry["posted_at"])
    day1 = (posted_at + timedelta(days=1)).strftime("%Y-%m-%d")
    day7 = (posted_at + timedelta(days=MATURITY_DAYS)).strftime("%Y-%m-%d")
    posted_date_str = posted_at.strftime("%Y-%m-%d")

    views_24h, = _query_metrics(analytics, video_id, posted_date_str, day1, "views")
    views_7d, avg_pct_7d, likes_7d, comments_7d, subs_gained_7d = _query_metrics(
        analytics, video_id, posted_date_str, day7, "views,averageViewPercentage,likes,comments,subscribersGained"
    )
    revenue = _fetch_revenue(analytics, video_id, posted_date_str, day7)

    append_row(
        CHANNEL_SHEET_ID,
        "動画データ",
        [
            video_id,
            entry["posted_at"],
            entry["theme"],
            entry["format"],
            entry["winner"],
            entry["title"],
            views_7d,
            avg_pct_7d,
            likes_7d,
            comments_7d,
            subs_gained_7d,
            revenue,
            entry.get("theme_type", "normal"),
            entry.get("theme_category", ""),
        ],
    )
    append_row(
        CHANNEL_SHEET_ID,
        "投稿時刻実験",
        [
            video_id,
            entry["posted_at"],
            WEEKDAY_JA[posted_at.weekday()],
            views_24h,
            views_7d,
            avg_pct_7d,
        ],
    )
    print(f"[分析データ収集] {video_id} をSheetsに記録しました（views_7d={views_7d}）。")
    return avg_pct_7d


def main():
    entries = load_entries()
    analytics = get_analytics_service()
    now = datetime.now().astimezone()

    updated = False
    for entry in entries:
        if entry.get("analytics_written"):
            continue
        posted_at = datetime.fromisoformat(entry["posted_at"])
        if (now - posted_at) < timedelta(days=MATURITY_DAYS):
            continue

        avg_pct_7d = collect_for_entry(analytics, entry)
        entry["analytics_written"] = True
        updated = True

        # 編集パターンのバンディット選定（pattern_pool.py）へ実測パフォーマンスをフィードバックする。
        # スコアは7日間の平均視聴維持率(%)を使う（doc: editing_pattern_engine_v1.md 4章）。
        if not entry.get("pattern_performance_recorded") and entry.get("selected_patterns"):
            try:
                score = float(avg_pct_7d or 0)
            except (TypeError, ValueError):
                score = 0.0
            for character, selection in entry["selected_patterns"].items():
                record_performance(character, selection["id"], score)
            entry["pattern_performance_recorded"] = True

    if updated:
        save_entries(entries)
    else:
        print("[分析データ収集] 収集対象（投稿から7日経過・未収集）の動画はありませんでした。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "collect_analytics.py")
