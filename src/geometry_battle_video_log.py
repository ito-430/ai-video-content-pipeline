"""4ch目(幾何学物理演算バトル/SimuSphere Arena)の投稿動画ログ。

投稿パイプラインがまだ手動運用（[[project_geometry_battle_channel]]参照）のため、
Sheetsではなく軽量なローカルJSONで1本ごとの選定結果を蓄積する。
geometry_battle_kpt.pyの週次分析のinputになる。将来Sheets化する場合も、
このレコード構造をそのまま移植できるようにしている。

視聴実績(再生数等)は現状取得手段（YouTube Analytics APIとの接続）が未整備のため
含めない。将来接続した際はupdate_viewsで後から追記する設計。
"""

import json
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "scripts_templates" / "geometry_battle_video_log.json"


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return json.loads(LOG_PATH.read_text(encoding="utf-8"))


def _save_log(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def log_video(
    video_id: str,
    published_at: str,
    privacy_status: str,
    rule: str,
    shape: str,
    seed: int,
    score_breakdown: dict,
    winner_character: str | None,
    player_shape: str | None = None,
    terrain: str | None = None,
    match_type: str = "individual",
) -> None:
    """1本の動画（候補選定〜投稿）を記録する。同じvideo_idが既にあれば上書きする。
    player_shape(2026-09-08追加): プレイヤー本体の形状(円/正方形/三角形)。ユーザーから
    「四角形プレイヤー軸も同じ程度の確率で採用されるように」と要望があったため、実際の
    採用実績を週次KPTで追跡できるように記録する。
    terrain(2026-09-09追加): 新ステージ内部構造(hourglass/donut/cross/pegboard/None)。
    採用実績を週次KPTで追跡できるように記録する。
    match_type(2026-09-21追加): 対戦形式("individual"/"team"/"boss")。既存のruleとは
    独立した軸で、geometry_battle_kpt.pyのALGO_METRICS_AXESに追加すれば同じ集計に乗る。"""
    entries = [e for e in load_log() if e.get("video_id") != video_id]
    entries.append(
        {
            "video_id": video_id,
            "published_at": published_at,
            "privacy_status": privacy_status,
            "rule": rule,
            "shape": shape,
            "player_shape": player_shape,
            "terrain": terrain,
            "match_type": match_type,
            "seed": seed,
            "score": score_breakdown,
            "winner_character": winner_character,
        }
    )
    _save_log(entries)


def update_views(video_id: str, view_count: int, like_count: int | None = None) -> bool:
    """将来YouTube Data APIから取得した簡易実績(views/likes)を後から追記する。
    該当videoが見つかった場合のみTrueを返す。"""
    entries = load_log()
    found = False
    for e in entries:
        if e.get("video_id") == video_id:
            e["view_count"] = view_count
            if like_count is not None:
                e["like_count"] = like_count
            found = True
    if found:
        _save_log(entries)
    return found
