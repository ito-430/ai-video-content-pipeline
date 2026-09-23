"""物理演算バトルチャンネル(4ch目)の離脱曲線(視聴維持率カーブ)+YouTubeアルゴリズム有利指標の収集。

8-7章「離脱曲線データの取得」対応。動画単位の再生数・全体維持率だけでなく、動画内の
どの秒数(elapsedVideoTimeRatio)で離脱が集中しているかをYouTube Analytics APIから取得し、
将来の基準判定(4章)の閾値調整に使えるデータとして蓄積する。

視聴維持率データに限らず、YouTubeアルゴリズムで有利になる立ち回りを最大化するために
必要なデータを揃える狙いで、離脱曲線に加えて動画単位のサマリー指標(views/平均視聴率/
いいね/コメント/シェア/登録者増加/
インプレッション/インプレッションCTR)も併せて取得するようにした。他チャンネル
(ch1の`collect_analytics.py`)が既に集めている指標セットに、Shorts固有のフィード露出
指標(impressions/impressionClickThroughRate)を加えた形。取得した指標をどう解釈し、
どんな対策(ルール重み調整等)を打つかはここでは判断せず、週次KPT
(geometry_battle_kpt.py、ルール別に集計して提示)による
人間判断に委ねる設計を維持する。

投稿から一定日数(RETENTION_MIN_AGE_DAYS)経過し、まだ取得していない公開動画についてのみ
取得する(非公開/限定公開は十分な視聴データが集まらないため対象外。投稿直後もデータが
安定しないため待つ)。実際の閾値調整はここでは行わず、取得・蓄積までを担当する
(継続的な調整は週次KPTでの人間判断に委ねる設計)。
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from geometry_battle_video_log import load_log
from youtube_upload import get_analytics_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETENTION_PATH = PROJECT_ROOT / "scripts_templates" / "geometry_battle_retention.json"
CLIENT_SECRET_PATH = PROJECT_ROOT / "materials" / "youtube_client_secret_geometry.json"
TOKEN_PATH = PROJECT_ROOT / "materials" / "youtube_token_geometry.json"

RETENTION_MIN_AGE_DAYS = 3  # 投稿直後はデータが安定しないため、これだけ経過してから取得する

# ch1のcollect_analytics.pyが集めている指標セット(views/平均視聴率/いいね/コメント/登録者増加)に
# shares(シェア)を加えたもの。estimatedRevenue(推定収益)は未収益化チャンネルでは取得できず、
# かつ「アルゴリズム有利さ」とは直接関係が薄いためここでは対象外にしている。
SUMMARY_METRICS = "views,averageViewPercentage,likes,comments,shares,subscribersGained"
# Shorts固有のフィード露出指標。チャンネルの状況によっては取得できないことがあるため、
# SUMMARY_METRICSとは別クエリにして失敗してもそちらを道連れにしないようにする。
IMPRESSION_METRICS = "impressions,impressionClickThroughRate"


def _load_retention() -> dict:
    if RETENTION_PATH.exists():
        try:
            return json.loads(RETENTION_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_retention(data: dict) -> None:
    RETENTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETENTION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_retention_curve(video_id: str, published_at: str) -> list[dict]:
    """elapsedVideoTimeRatio(0-1の相対経過時間)ごとの視聴維持率カーブを取得する。"""
    analytics = get_analytics_service(CLIENT_SECRET_PATH, TOKEN_PATH)
    start_date = published_at[:10]
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    response = analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="audienceWatchRatio,relativeRetentionPerformance",
        dimensions="elapsedVideoTimeRatio",
        filters=f"video=={video_id}",
    ).execute()
    return [
        {"elapsed_ratio": row[0], "audience_watch_ratio": row[1], "relative_retention": row[2]}
        for row in response.get("rows", [])
    ]


def _query_video_metrics(analytics, video_id: str, start_date: str, end_date: str, metrics: str) -> list:
    """dimensionsを指定しない(=対象期間全体で動画1本ぶんに集約された)単一行を取得する。
    collect_analytics.py(ch1)の_query_metricsと同じ形。行が無ければ0埋めで返す
    (「データがまだ無い」と「指標自体が使えない」を呼び出し側で区別する必要が無い場合用)。"""
    resp = analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics=metrics,
        filters=f"video=={video_id}",
    ).execute()
    rows = resp.get("rows")
    if not rows:
        return [0] * len(metrics.split(","))
    return rows[0]


def fetch_summary_metrics(video_id: str, published_at: str) -> dict:
    """投稿日〜現在までの累計で、views/平均視聴率/エンゲージメント/インプレッションCTRを取得する。
    インプレッション系はチャンネルの状況によって取得できないことがあるため、失敗しても
    他の指標を道連れにせず、summaryにその2キーだけ含めない形で返す。"""
    analytics = get_analytics_service(CLIENT_SECRET_PATH, TOKEN_PATH)
    start_date = published_at[:10]
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    views, avg_pct, likes, comments, shares, subs_gained = _query_video_metrics(
        analytics, video_id, start_date, end_date, SUMMARY_METRICS
    )
    summary = {
        "views": views,
        "average_view_percentage": avg_pct,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "subscribers_gained": subs_gained,
    }
    try:
        impressions, impression_ctr = _query_video_metrics(
            analytics, video_id, start_date, end_date, IMPRESSION_METRICS
        )
        summary["impressions"] = impressions
        summary["impression_ctr"] = impression_ctr
    except Exception as e:
        print(f"[情報] {video_id} のインプレッション指標は取得できませんでした: {e}", file=sys.stderr)
    return summary


def main() -> None:
    retention = _load_retention()
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_MIN_AGE_DAYS)
    updated = False

    for video in load_log():
        video_id = video["video_id"]
        if video_id in retention or video.get("privacy_status") != "public":
            continue
        try:
            published_at = datetime.fromisoformat(video["published_at"])
        except (KeyError, ValueError):
            continue
        if published_at.astimezone(timezone.utc) > cutoff:
            continue

        try:
            curve = fetch_retention_curve(video_id, video["published_at"])
        except Exception as e:
            print(f"[警告] {video_id} の離脱曲線取得に失敗しました: {e}", file=sys.stderr)
            continue

        if not curve:
            print(f"[離脱曲線] {video_id}: データがまだありません(スキップ)")
            continue

        try:
            summary = fetch_summary_metrics(video_id, video["published_at"])
        except Exception as e:
            print(f"[警告] {video_id} のサマリー指標取得に失敗しました: {e}", file=sys.stderr)
            summary = {}

        retention[video_id] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "curve": curve,
            "summary": summary,
        }
        updated = True
        print(f"[離脱曲線] {video_id}: {len(curve)}ポイント取得(views={summary.get('views', '?')})")

    if updated:
        _save_retention(retention)
    else:
        print("[離脱曲線] 新規取得対象はありませんでした。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "geometry_battle_retention.py")
