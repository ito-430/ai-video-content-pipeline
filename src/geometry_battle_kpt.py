"""物理演算バトルチャンネル(4ch目/幾何学物理演算バトル)の週次KPT(Keep/Problem/Try)。

1ch目のgenerate_kpt.py（意見交換フロー）を土台にしつつ、ch4向けに以下を変更している。
- ユーザー要望（2026-09-09）でスコープを最小化: Discordの声の集約先は「アイデア出し」
  1チャンネルのみ、投稿の実績通知（1ch目の「投稿通知」に相当する機能）は持たない。
  高頻度でDiscordを見てもらう必要はないという前提のため、チャンネル構成自体を減らしている。
- データ基盤がGoogle Sheetsではなくローカルの軽量JSON
  （geometry_battle_video_log.json / geometry_battle_character_stats.json）。
  投稿パイプラインがまだ手動運用で本数も少ないため、Sheets連携は過剰と判断した。
- 「テーマ」の代わりに rule/shape/パラメータ等のゲームデザイン軸を評価対象にする。
  before/after例も台本セリフではなく設定変更の例にしている。
- 投稿本数がまだ非常に少ないため、定量的な結論を無理に出さず「データ不足」を
  明示することをSYSTEM_PROMPTで強く指示している（実績が少ないのに断定的な
  Try提案を出すと誤った方向にチューニングしてしまうリスクがあるため）。

## 絶対厳守のガードレール
1. 永久的な収益の最大化（短期的な変化より右肩上がりの継続を優先）
2. 炎上・アカウント停止・個人情報漏洩リスクをゼロにする
Try提案はこの2点に抵触しないかをGemini自身に自己判定させ、抵触の疑いがあるものは
機械的に除外し、Discordには一切出さない。

## 意見交換フロー
Try提案は一発承認ではなく、投稿メッセージから作成したDiscordスレッドで意見交換を行う。
- ✅❌リアクション: いつでも押せば、その時点の最新案で確定（承認/却下）する。
- スレッドへの返信: geometry_battle_kpt_followup.py が改訂案を再提示する。
前週分が意見交換中（未確定）の場合、今週分の新規生成はスキップし重複を避ける。
"""

import json
import os
import sys
from datetime import datetime, timedelta

from ai_provider import get_text_provider
from discord_client import (
    add_reaction,
    channel_id,
    create_thread,
    get_recent_messages,
    post_message,
    snowflake_from_datetime,
)
from geometry_battle_characters import load_stats, win_balance_score
import geometry_battle_experiment_log as experiment_log
from geometry_battle_video_log import load_log
from shared_knowledge import extract_success_pattern

PROJECT_ROOT_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "geometry_battle_kpt_state.json")
HISTORY_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "geometry_battle_kpt_history.json")
RETENTION_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "geometry_battle_retention.json")
PEAKS_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "geometry_battle_kpi_peaks.json")

# 2026-09-21、ユーザー指示「目標設定とKPTトラッキングの修正」対応: 「平均視聴率95%」という
# 固定目標を廃止し、直近の実績が過去の最高値(ピーク)を安定して超え、段階的に引き上げて
# いるかを追跡する仕様に変更する。初期値はユーザーから提示された2026-09-21時点の実測ピーク
# (維持率=average_view_percentage 45.1%、閲覧率=impression_ctr 61.7%)をシードとして使う。
DEFAULT_PEAKS = {
    "average_view_percentage": 45.1,
    "impression_ctr": 61.7,
}
RETENTION_CHECKPOINTS = [0.25, 0.5, 0.75, 0.9]  # 8-7: 離脱曲線データを見る経過地点(動画内の相対位置)
MODEL_NAME = "gemini-flash-lite-latest"

LOOKBACK_DAYS = 7
FEEDBACK_CHANNEL = "battle_ideas"
KPT_CHANNEL = "battle_kpt_reports"
MAX_FEEDBACK_MESSAGES = 30

TRY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestion": {"type": "string"},
        "is_safe": {
            "type": "boolean",
            "description": (
                "この提案が「永久的な収益最大化を優先する」「炎上・アカウント停止・"
                "個人情報漏洩リスクをゼロにする」の2大方針に抵触しないと自己判定できるか"
            ),
        },
        "risk_level": {
            "type": "string",
            "enum": ["none", "mild"],
            "description": "「刺激的」「インパクト強化」等、炎上・賛否のリスクが少しでもあるなら mild にすること",
        },
        "before_example": {
            "type": "string",
            "description": "risk_level=mildの場合のみ、変更前の設定・パラメータの具体例（noneの場合は空文字列）",
        },
        "after_example": {
            "type": "string",
            "description": "risk_level=mildの場合のみ、変更後の設定・パラメータの具体例（noneの場合は空文字列）",
        },
    },
    "required": ["suggestion", "is_safe", "risk_level", "before_example", "after_example"],
}

KPT_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}, "description": "うまくいっている点（2〜4件）"},
        "problem": {"type": "array", "items": {"type": "string"}, "description": "課題点（2〜4件）"},
        "try_items": {
            "type": "array",
            "description": "次週以降に試す改善案（2〜4件）",
            "items": TRY_ITEM_SCHEMA,
        },
    },
    "required": ["keep", "problem", "try_items"],
}

REVISION_SCHEMA = {
    "type": "object",
    "properties": {"try_items": {"type": "array", "items": TRY_ITEM_SCHEMA}},
    "required": ["try_items"],
}

SYSTEM_PROMPT = """\
あなたは幾何学図形の物理演算バトル動画チャンネル（物理演算バトルチャンネル）のデータアナリスト兼\
ディレクターです。直近1週間の状況をもとに、週次KPT（Keep/Problem/Try）を作成してください。

## このチャンネルの前提（重要）
- 投稿パイプラインはまだ手動運用で、公開実績（再生数・視聴維持率等）はほぼ存在しない段階です。
  入力データに投稿本数が少ない、または視聴実績がないと書かれている場合、無理に定量的な結論を
  出さず、「データがまだ不足している」とKeep/Problemで正直に述べてください。母数が少ない段階で
  断定的なチューニング指示（Try）を出すことは、誤った方向に調整してしまうリスクの方が大きいです。
- 評価対象は「テーマ」ではなく、ルール(rule)・形状(shape)・特殊能力・カメラワーク・スローモーション
  等のゲームデザイン軸と、基準判定スコア（8項目）、キャラクター戦績（勝率バランス）です。

## 目標の考え方（重要、2026-09-21改訂）
このチャンネルに「平均視聴率95%」のような固定の絶対目標はありません。「■目標進捗」に
記載される直近実績と過去ピークの比較を見て、(1)直近実績が過去ピークを安定して超えて
いるか、(2)ピーク自体が段階的に引き上げられているか、の2点で評価してください。
ピークを更新できていない週が続く場合はProblemで扱い、更新できていればKeepで扱う、
という相対的な評価をしてください（絶対的な水準を持ち出さないこと）。

## 観点（データがあれば問うこと）
- 基準判定スコアの傾向（合格率・どの項目が低くなりがちか）
- キャラクター勝率バランス（特定キャラ・タイプに極端な偏りがないか）
- ルール/形状の多様性（同じ組み合わせに偏っていないか。量産型コンテンツ判定回避の核が
  「複数軸の組み合わせによる多様化」であるため、多様性の欠如は必ずProblemで扱うこと）
- YouTubeの「inauthentic content」ポリシー（テンプレート的・量産的で人間の創意工夫が見えない
  コンテンツはチャンネル単位で凍結されうる）に照らして兆候がないか
- 離脱曲線データ（8-7）: 動画内の特定の経過地点で視聴維持率が大きく落ちている傾向が見えたら、
  その原因（決着までのタメが長すぎる、逆に早すぎる等）を推測し、Problem/Tryで扱ってよい。
  ただしサンプル数が少ないうちは断定を避けること

## 絶対厳守（Try提案すべてに優先する。少しでも抵触の疑いがあればis_safeをfalseにすること）
1. 永久的な収益の最大化（短期的な変化で継続的成長を犠牲にする提案は禁止）
2. 炎上・アカウント停止・個人情報漏洩リスクをゼロにする（コンプライアンスに反する提案、
   および上記のYouTubeポリシーに抵触しうる提案は禁止）

## 刺激性のある提案への対応
「刺激的」「インパクトを強める」等、炎上・賛否のリスクが少しでもある提案には、
risk_levelを"mild"にした上で、変更前(before_example)と変更後(after_example)の
具体的な設定・パラメータ例を必ず添えること。リスクがない提案はrisk_level="none"でよい。

## Discordでのユーザーの声について
入力データには「#アイデア出し」チャンネルでの直近の生の発言が含まれる。定量データでは
見えない現場の感覚・要望・不満であるため、Keep/Problem/Tryの材料として積極的に反映すること
（データと発言が矛盾する場合は両方を併記してよい）。

## 出力方針
- Keep/Problemはデータの傾向（またはデータ不足の事実）を踏まえた具体的な言及にすること。
- Tryは次週すぐ試せる具体的なアクションにすること（例: 「rule=absorb_growthのgravity値を
  さらに下げて尺を安定させる」「特殊能力の発動間隔を調整する」等）。
"""


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_history() -> list[dict]:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(history: list[dict]) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _within_lookback(date_str: str, cutoff: datetime) -> bool:
    try:
        return datetime.fromisoformat(date_str) >= cutoff
    except (ValueError, TypeError):
        return False


def _collect_feedback(cutoff: datetime) -> str:
    """「アイデア出し」チャンネルの生の発言を、KPTの追加コンテキストとして集める。"""
    after_id = snowflake_from_datetime(cutoff)
    try:
        messages = get_recent_messages(FEEDBACK_CHANNEL, after_id=after_id, limit=100)
    except Exception as e:
        print(f"[警告] {FEEDBACK_CHANNEL}の取得に失敗しました: {e}", file=sys.stderr)
        return "(取得に失敗しました)"

    user_messages = [
        m for m in messages if not m.get("author", {}).get("bot") and (m.get("content") or "").strip()
    ]
    if not user_messages:
        return "(直近のDiscord発言はありません)"
    return "\n".join(f"- {m['content']}" for m in user_messages[:MAX_FEEDBACK_MESSAGES])


def _video_summary(cutoff: datetime) -> str:
    log = load_log()
    recent = [e for e in log if _within_lookback(e.get("published_at", ""), cutoff)]

    lines = [f"■直近{LOOKBACK_DAYS}日間の投稿本数: {len(recent)}本（累計{len(log)}本）"]
    for e in recent:
        s = e.get("score", {})
        lines.append(
            f"- video_id={e.get('video_id')} rule={e.get('rule')} shape={e.get('shape')} "
            f"player_shape={e.get('player_shape') or '不明'} "
            f"match_type={e.get('match_type') or 'individual'} "
            f"総合スコア={s.get('overall')} 判定={'合格' if s.get('passed') else '不合格'} "
            f"勝者={e.get('winner_character') or '不明'} 公開状態={e.get('privacy_status')}"
        )

    if not log:
        lines.append("(投稿実績データはまだありません。現状は技術検証・初回非公開テストの段階です)")
    elif not recent:
        lines.append(f"(直近{LOOKBACK_DAYS}日間の新規投稿はありません)")

    overall_scores = [e["score"]["overall"] for e in log if isinstance(e.get("score", {}).get("overall"), (int, float))]
    if overall_scores:
        lines.append(f"■累計{len(overall_scores)}本の基準判定スコア平均: {sum(overall_scores) / len(overall_scores):.3f}")

    rule_counts: dict[str, int] = {}
    for e in log:
        rule_counts[e.get("rule", "不明")] = rule_counts.get(e.get("rule", "不明"), 0) + 1
    if rule_counts:
        dist = "、".join(f"{r}:{c}本" for r, c in rule_counts.items())
        lines.append(f"■ルールの内訳（累計）: {dist}")

    return "\n".join(lines)


def _character_summary() -> str:
    stats = load_stats()
    if not stats:
        return "(戦績データはまだありません)"
    lines = []
    for name, e in sorted(stats.items(), key=lambda kv: -kv[1]["appearances"]):
        appearances = e.get("appearances", 0)
        wins = e.get("wins", 0)
        rate = wins / appearances if appearances else 0.0
        lines.append(f"- {name}: 登場{appearances}回 勝利{wins}回 勝率{rate:.0%}")
    lines.append(f"■勝率バランススコア（0-1、1が理想）: {win_balance_score():.3f}")
    return "\n".join(lines)


def _retention_summary() -> str:
    """8-7「離脱曲線データの取得」: geometry_battle_retention.pyが週次で蓄積したデータを要約する。
    実際の閾値調整はKPTでの議論を踏まえて開発側で判断する（ここでは提示するだけ）。
    """
    if not os.path.exists(RETENTION_PATH):
        return "(離脱曲線データはまだありません。投稿から数日経過した動画から順次取得されます)"
    try:
        with open(RETENTION_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "(離脱曲線データの読み込みに失敗しました)"
    if not data:
        return "(離脱曲線データはまだありません。投稿から数日経過した動画から順次取得されます)"

    lines = [f"（{len(data)}本の動画から取得済み。経過地点ごとの平均視聴維持率）"]
    for checkpoint in RETENTION_CHECKPOINTS:
        ratios = []
        for entry in data.values():
            curve = entry.get("curve") or []
            if not curve:
                continue
            closest = min(curve, key=lambda p: abs(p["elapsed_ratio"] - checkpoint))
            ratios.append(closest["audience_watch_ratio"])
        if ratios:
            lines.append(f"- 経過{int(checkpoint * 100)}%地点: {sum(ratios) / len(ratios):.1f}%")
    return "\n".join(lines)


# 2026-09-21、ユーザー指摘「rule以外の軸が関係している可能性も十分にあるので分析時は
# 他の軸も見てみてください」への対応。ruleだけでなく、video_logに既に記録されている
# 他の生成軸(shape=外枠の形/player_shape=プレイヤー本体の形/terrain=内部構造)についても
# 同じ集計をかける。「rule単独では説明できない差」(例: 同じruleでもterrainによって
# 視聴維持率が大きく変わる)を見逃さないようにする狙い。
ALGO_METRICS_AXES = [
    ("rule", "ルール"),
    ("shape", "外枠の形"),
    ("player_shape", "プレイヤー形状"),
    ("terrain", "内部ステージ構造"),
    ("match_type", "対戦形式"),  # 2026-09-21追加: individual/team/boss(video_log参照)
]


def _load_retention_data() -> dict | None:
    if not os.path.exists(RETENTION_PATH):
        return None
    try:
        with open(RETENTION_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data or None


def _algo_metrics_by_axis(axis: str, log: list[dict], retention: dict) -> str:
    """geometry_battle_retention.pyが集めるサマリー指標(views/平均視聴率/いいね/コメント/
    シェア/登録者増加/インプレッションCTR)を、video_logの指定軸(rule/shape/player_shape/
    terrain)の値と突き合わせて集計する。対策の判断・実施はここでは行わず、このチャンネルの
    既存方針通りKPTでの人間判断に委ねる。
    match_type軸は2026-09-21追加のため、それ以前に記録された動画にはフィールド自体が
    無い。その場合は当時の唯一の対戦形式だった"individual"として扱う(他の軸は
    未記録=文字通り「無し」で構わないため、match_typeだけ既定値を変える)。"""
    default_value = "individual" if axis == "match_type" else "無し"
    value_by_video = {e["video_id"]: e.get(axis) or default_value for e in log}

    by_value: dict[str, list[dict]] = {}
    for video_id, entry in retention.items():
        summary = entry.get("summary")
        if not summary:
            continue
        value = value_by_video.get(video_id, "不明")
        by_value.setdefault(value, []).append(summary)

    if not by_value:
        return "  (サマリー指標はまだありません)"

    lines = []
    for value, entries in sorted(by_value.items(), key=lambda kv: -len(kv[1])):
        n = len(entries)
        avg_views = sum(e.get("views", 0) for e in entries) / n
        avg_pct = sum(e.get("average_view_percentage", 0) for e in entries) / n
        engagement_rates = [
            (e.get("likes", 0) + e.get("comments", 0) + e.get("shares", 0)) / e["views"]
            for e in entries if e.get("views")
        ]
        avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0.0
        avg_subs = sum(e.get("subscribers_gained", 0) for e in entries) / n
        ctrs = [e["impression_ctr"] for e in entries if e.get("impression_ctr") is not None]
        ctr_text = f"{sum(ctrs) / len(ctrs):.2f}%" if ctrs else "データ無し"
        lines.append(
            f"  - {value}({n}本): 平均views={avg_views:.0f} 平均視聴率={avg_pct:.1f}% "
            f"エンゲージメント率={avg_engagement:.2%} 平均登録者増加={avg_subs:.2f} "
            f"インプレッションCTR={ctr_text}"
        )
    return "\n".join(lines)


def _algo_metrics_summary() -> str:
    """2026-09-21、ユーザー指示「視聴維持率データに限らず、YouTubeアルゴリズムで有利になる
    立ち回りを最大化するために必要なデータ収集・対策の考案ができるようにしたい」への対応。
    rule/shape/player_shape/terrainの4軸それぞれで、サマリー指標を集計して並べる。"""
    retention = _load_retention_data()
    if retention is None:
        return "(まだデータがありません)"

    log = load_log()
    lines = []
    for axis, label in ALGO_METRICS_AXES:
        lines.append(f"◇{label}別:")
        lines.append(_algo_metrics_by_axis(axis, log, retention))
    return "\n".join(lines)


def _load_peaks() -> dict:
    if os.path.exists(PEAKS_PATH):
        try:
            with open(PEAKS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("peaks"):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"peaks": dict(DEFAULT_PEAKS), "history": []}


def _save_peaks(data: dict) -> None:
    os.makedirs(os.path.dirname(PEAKS_PATH), exist_ok=True)
    with open(PEAKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _weighted_recent_metrics(cutoff: datetime) -> dict[str, float | None]:
    """直近(cutoff以降)に投稿された動画について、views加重平均の視聴維持率と、
    impressions加重平均のインプレッションCTRを計算する(いずれもデータが無ければNone)。"""
    retention = _load_retention_data()
    if retention is None:
        return {"average_view_percentage": None, "impression_ctr": None}
    log = load_log()
    recent_ids = {e["video_id"] for e in log if _within_lookback(e.get("published_at", ""), cutoff)}

    view_weighted_sum, view_weight_total = 0.0, 0.0
    ctr_weighted_sum, ctr_weight_total = 0.0, 0.0
    for video_id, entry in retention.items():
        if video_id not in recent_ids:
            continue
        summary = entry.get("summary") or {}
        views = summary.get("views")
        avg_pct = summary.get("average_view_percentage")
        if views and avg_pct is not None:
            view_weighted_sum += views * avg_pct
            view_weight_total += views
        impressions = summary.get("impressions")
        ctr = summary.get("impression_ctr")
        if impressions and ctr is not None:
            ctr_weighted_sum += impressions * ctr
            ctr_weight_total += impressions

    return {
        "average_view_percentage": (view_weighted_sum / view_weight_total) if view_weight_total else None,
        "impression_ctr": (ctr_weighted_sum / ctr_weight_total) if ctr_weight_total else None,
    }


_PEAK_METRIC_LABELS = {
    "average_view_percentage": "視聴維持率(views加重平均)",
    "impression_ctr": "インプレッションCTR(impressions加重平均)",
}


def _peak_progress_summary(cutoff: datetime) -> str:
    """2026-09-21、ユーザー指示「目標の相対化」対応: 固定の絶対目標(旧: 平均視聴率95%)では
    なく、直近実績が過去のピークを安定して超え、段階的に引き上げられているかを追跡する。
    ピークを更新した場合は状態ファイルに反映し、次回以降の比較基準そのものを引き上げる
    (=目標が動的に相対化される)。"""
    peaks_data = _load_peaks()
    peaks = peaks_data["peaks"]
    recent = _weighted_recent_metrics(cutoff)

    lines = []
    updated = False
    for metric_key, label in _PEAK_METRIC_LABELS.items():
        current_peak = peaks.get(metric_key, DEFAULT_PEAKS[metric_key])
        value = recent.get(metric_key)
        if value is None:
            lines.append(f"- {label}: 直近{LOOKBACK_DAYS}日間のデータ不足のため判定不能(現在のピーク: {current_peak:.1f}%)")
            continue
        if value > current_peak:
            peaks_data.setdefault("history", []).append(
                {"metric": metric_key, "old_peak": current_peak, "new_peak": round(value, 2), "date": cutoff.isoformat()}
            )
            peaks[metric_key] = round(value, 2)
            updated = True
            lines.append(f"- {label}: {value:.1f}%(★新ピーク更新、旧ピーク{current_peak:.1f}%)")
        else:
            gap = current_peak - value
            lines.append(f"- {label}: {value:.1f}%(ピーク{current_peak:.1f}%に対して-{gap:.1f}pt、未更新)")

    if updated:
        _save_peaks(peaks_data)

    return "\n".join(lines)


def build_summary() -> str:
    cutoff = datetime.now().astimezone() - timedelta(days=LOOKBACK_DAYS)

    lines = ["■動画実績", _video_summary(cutoff), ""]
    lines += ["■キャラクター戦績", _character_summary(), ""]
    lines += ["■目標進捗(固定目標ではなく直近ピークとの比較)", _peak_progress_summary(cutoff), ""]
    lines += ["■アクティブな実験(1デプロイ1仮説の原則、geometry_battle_experiment_log.py)", experiment_log.summary_for_kpt(), ""]
    lines += ["■離脱曲線データ（8-7、基準判定の尺閾値調整の参考情報）", _retention_summary(), ""]
    lines += ["■ルール別パフォーマンス指標（YouTubeアルゴリズム有利さの判断材料）", _algo_metrics_summary(), ""]
    lines += [f"■Discordでのユーザーの声（直近{LOOKBACK_DAYS}日: #アイデア出し）", _collect_feedback(cutoff)]
    return "\n".join(lines)


def generate_kpt(summary: str) -> dict:
    provider = get_text_provider(MODEL_NAME)
    return provider.generate_json(SYSTEM_PROMPT, summary, KPT_SCHEMA)


def format_try_item(item: dict) -> str:
    text = f"- {item['suggestion']}"
    if item.get("risk_level") == "mild" and (item.get("before_example") or item.get("after_example")):
        text += (
            f"\n  ⚠️炎上リスクに少し配慮が必要な提案です。\n"
            f"  変更前例: {item.get('before_example', '')}\n"
            f"  変更後例: {item.get('after_example', '')}"
        )
    return text


def format_kpt_message(week_label: str, keep: list[str], problem: list[str], safe_tries: list[dict], rejected_count: int) -> str:
    keep_text = "\n".join(f"- {k}" for k in keep)
    problem_text = "\n".join(f"- {p}" for p in problem)
    try_text = "\n".join(format_try_item(t) for t in safe_tries) or "(今週は提案なし)"

    message = (
        f"📊 物理演算バトルチャンネル 週次KPT報告（{week_label}）\n\n"
        f"■Keep\n{keep_text}\n\n■Problem\n{problem_text}\n\n■Try\n{try_text}\n"
    )
    if rejected_count:
        message += f"\n（ガードレール抵触の疑いにより{rejected_count}件の提案を自動除外しました）\n"
    message += (
        "\nこのメッセージから作成したスレッドで、意見や修正の希望を自由に返信してください。"
        "内容を踏まえて改訂案を再提示します。\n"
        "この案で進めてよければ✅、今回は見送るなら❌でリアクションしてください。"
    )
    return message


def post_and_record_kpt(summary: str, kpt: dict, state: dict) -> None:
    safe_tries = [t for t in kpt.get("try_items", []) if t.get("is_safe")]
    rejected_count = len(kpt.get("try_items", [])) - len(safe_tries)

    week_label = datetime.now().strftime("%Y-%m-%d時点の週")
    message = format_kpt_message(week_label, kpt.get("keep", []), kpt.get("problem", []), safe_tries, rejected_count)

    posted = post_message(KPT_CHANNEL, message)
    for emoji in ("✅", "❌"):
        try:
            add_reaction(KPT_CHANNEL, posted["id"], emoji)
        except Exception:
            pass

    thread_id = None
    try:
        thread = create_thread(KPT_CHANNEL, posted["id"], f"KPT意見交換_{week_label}")
        thread_id = thread["id"]
    except Exception as e:
        print(f"[警告] スレッド作成に失敗しました（意見交換なしで承認/却下のみ運用します）: {e}", file=sys.stderr)

    history = _load_history()
    history.append(
        {
            "week_label": week_label,
            "keep": kpt.get("keep", []),
            "problem": kpt.get("problem", []),
            "try_items": safe_tries,
            "summary": summary,
            "judgment": "",
            "status": "意見交換中",
        }
    )
    _save_history(history)
    history_index = len(history) - 1

    state["pending"] = {
        "message_id": posted["id"],
        "thread_id": thread_id,
        "history_index": history_index,
        "keep": kpt.get("keep", []),
        "problem": kpt.get("problem", []),
        "try_items": safe_tries,
        "last_seen_thread_msg_id": None,
        "resolved": False,
        # ✅❌の確定判定は「今一番新しく提示している案」に対して行う
        "current_reaction": {"channel_id": channel_id(KPT_CHANNEL), "message_id": posted["id"]},
    }


def main():
    state = _load_state()
    pending = state.get("pending")
    if pending and not pending.get("resolved"):
        print(
            "[battle KPT] 前回分がまだ意見交換中/未確定のため、今週の新規生成はスキップします。"
            "geometry_battle_kpt_followup.py が確定を検知するまでお待ちください。"
        )
        return

    summary = build_summary()
    kpt = generate_kpt(summary)
    post_and_record_kpt(summary, kpt, state)

    _save_state(state)
    print("物理演算バトルチャンネルの週次KPTを生成し、Discordへ投稿しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "geometry_battle_kpt.py")
