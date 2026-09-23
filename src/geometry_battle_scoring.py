"""
幾何学アニメーション物理演算バトルチャンネル(4ch目) 基準判定ロジック プロトタイプ(4章)

軽量シミュレーション結果(geometry_battle_gen.SimResult)を受け取り、企画書4章の
8項目に対応するスコアを算出し、80%以上の一致度で合格/不合格を判定する。

- インプレッション分類: 決着パターン(即決着/一方的/駆け引き/逆転劇)を分類
- サプライズ指数: 上記を連続値化したもの(衝突頻度を代理指標に使用)
- 尺: 決着までの秒数があらかじめ用意した尺カテゴリのいずれかに収まるか
- 気持ちよさスコア: 衝突頻度が適正レンジ(密集しすぎ・間延びしすぎでない)か
- 密度・視認性: 参加数が画面内で見やすい範囲か
- 停滞検知: 一定時間イベント(衝突/脱落/特殊能力)が起きない区間がないか
- 勝率バランス: geometry_battle_characters.pyの累積戦績データ(scripts_templates/
  geometry_battle_character_stats.json)から算出。登場数が少ないうちは中立スコアを返す
- ブランド一貫性: 複数動画にまたがるブランド定義がまだ存在しないため、このプロトタイプでは
  プレースホルダー(常に満点)のまま

投稿パイプラインには接続しない、技術検証用のスクリプト。
重み付けは初期値であり、企画書8章の方針通り実データを見ながら継続的に調整する前提。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

try:
    from src.geometry_battle_characters import win_balance_score
    from src.geometry_battle_gen import FPS, SimResult, summarize
except ImportError:  # 直接実行された場合
    from geometry_battle_characters import win_balance_score
    from geometry_battle_gen import FPS, SimResult, summarize

# 尺のカテゴリ(秒)。将来的に「5秒必殺シリーズ」等のサブフォーマットのブランディングにも使える。
# 2026-09-07: 8〜25秒が無所属になっていた(goal_reach等で頻出する尺が全て不合格扱いになる
# バグ)ため、quick_15sを追加してカテゴリの隙間を埋めた
DURATION_CATEGORIES = {
    "instant_5s": (0, 8),
    "quick_15s": (8, 25),
    "standard_40s": (25, 55),
    "long_70s": (55, 90),
}

# 基準判定の合否しきい値(企画書4章の「80%以上の一致度」に対応)
DEFAULT_THRESHOLD = 0.8

# 2026-09-08、ユーザー指示による暫定の無条件フィルター: 「5秒で即決着してしまうバージョン」が
# 公開されてしまう問題への対応として、ユーザーが許可するまでは決着までの尺が指定範囲の
# 候補のみを合格とする。DURATION_CATEGORIESは他項目とのバランスを見る加点方式のスコアで、
# 範囲外でも他項目が良ければ合格しうる設計だったため、それとは別に無条件の足切りとして実装する。
# (2026-09-08、同日中に20〜60秒→20〜40秒へ再調整。「60秒はやや長い」とのユーザー判断)
# (2026-09-09、ループ再生・視聴維持率の最大化を狙い20〜40秒→15〜25秒へ再調整。ユーザー指示)
PRODUCTION_DECISION_SECONDS_RANGE = (15.0, 25.0)

# 2026-09-12、ユーザー指示: 序盤(離脱の大部分が発生する0〜3秒)の情報密度を上げるため、
# 開始からこの秒数以内に最初の衝突(プレイヤー間または壁/障害物との衝突、いずれもresult.collisions
# に記録される)が発生しない候補は「序盤停滞」として無条件不合格にする(尺フィルター・
# no_interactionフィルターと同種の無条件足切り)。
FIRST_IMPACT_MAX_SECONDS = 1.5

# 2026-09-20、ユーザー指示(ドラマ評価フィルター)でdrama(0.20)を新設。他項目を比例縮小して
# 枠を確保した(density=0.10→0.075、他の非プレースホルダー項目も若干縮小)。合計は1.0を維持。
SCORE_WEIGHTS = {
    "impression": 0.125,
    "surprise": 0.125,
    "duration": 0.15,
    "satisfaction": 0.15,
    "density": 0.075,
    "stagnation": 0.125,
    "drama": 0.20,
    "win_balance": 0.025,  # プレースホルダー(戦績DB未実装)
    "brand": 0.025,  # プレースホルダー(ブランド定義未実装)
}


@dataclass
class ScoreBreakdown:
    impression_pattern: str
    impression_score: float
    surprise_score: float
    duration_score: float
    duration_category: str | None
    satisfaction_score: float
    density_score: float
    stagnation_score: float
    drama_score: float
    coverage_score: float
    comeback_score: float
    close_finish_score: float
    win_balance_score: float
    brand_score: float
    overall: float
    passed: bool
    reject_reason: str | None = None
    first_collision_frame: int | None = None

    def to_dict(self) -> dict:
        return {
            "impression_pattern": self.impression_pattern,
            "impression_score": self.impression_score,
            "surprise_score": self.surprise_score,
            "duration_score": self.duration_score,
            "duration_category": self.duration_category,
            "satisfaction_score": self.satisfaction_score,
            "density_score": self.density_score,
            "stagnation_score": self.stagnation_score,
            "drama_score": self.drama_score,
            "coverage_score": self.coverage_score,
            "comeback_score": self.comeback_score,
            "close_finish_score": self.close_finish_score,
            "win_balance_score": self.win_balance_score,
            "brand_score": self.brand_score,
            "overall": self.overall,
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "first_collision_frame": self.first_collision_frame,
        }


def _first_collision_frame(result: SimResult) -> int | None:
    """最初に衝突(プレイヤー間・壁/障害物いずれも含む、result.collisionsは種別を問わず記録される)
    が発生したフレーム番号。一度も衝突が起きなければNone。"""
    if not result.collisions:
        return None
    return min(c["frame"] for c in result.collisions)


def _classify_duration(decision_seconds: float) -> str | None:
    for name, (lo, hi) in DURATION_CATEGORIES.items():
        if lo <= decision_seconds <= hi:
            return name
    return None


def _hole_escape_dominant(result: SimResult) -> bool:
    """2026-09-20、ユーザー指示: 「ただの生き残り戦(hole_fall)以外のルールで、そのルール
    独自の脱落原因(ゾーン外/吸収/被弾/武器ダメージ等)でなく、穴や端から場外に落ちた
    脱落の方が多い試合」を無条件で不合格にする(weapon_colosseum限定ではなく全ルール対象)。
    hole_fallは「穴に落として脱落させる」こと自体がルールの核なので対象外にする。
    causeが記録されていない(古い形式の)elimination_orderエントリは判定対象から除外する
    (誤検知で不要に不合格を増やさないための安全側の扱い)。"""
    if result.rule == "hole_fall":
        return False
    escaped = sum(1 for ev in result.elimination_order if ev.get("cause") == "escaped")
    rule_specific = sum(1 for ev in result.elimination_order if ev.get("cause") not in (None, "escaped"))
    return escaped > rule_specific


# 2026-09-21、ユーザー指示「レイト・クライマックス・フィルター」対応: 「勝負の決着(最後の
# 1体/1チームの撃破)」の直前の脱落が、決着フレームのこの割合以降に起きている必要がある。
LATE_CLIMAX_THRESHOLD = 0.9


def _late_climax_ok(result: SimResult) -> bool:
    """決着直前の脱落(=最後から2番目の脱落)が、試合の大部分(LATE_CLIMAX_THRESHOLD=90%)が
    終わった時点で起きているかを確認する。これが早い段階で終わっていて、その後ずっと
    1体(1チーム)だけが残った相手を追い回すだけの展開になっている「消化試合」(勝敗自体は
    実質的にとっくに決まっているのに映像だけ続く)を弾く狙い。
    脱落が2件未満(goal_reach等、脱落自体が決着条件でないルールを含む)の場合は判定不能なため
    素通りさせる(誤検知で不要に不合格を増やさないための安全側の扱い)。"""
    if len(result.elimination_order) < 2 or not result.decided_frame:
        return True
    second_to_last_frame = sorted(ev["frame"] for ev in result.elimination_order)[-2]
    return second_to_last_frame >= LATE_CLIMAX_THRESHOLD * result.decided_frame


def _impression_pattern(result: SimResult) -> tuple[str, float]:
    """決着までの脱落タイミングの分布から、即決着/一方的/駆け引き/逆転劇に分類する。
    脱落が終盤に偏っている(=最後まで拮抗していた)ほど「逆転劇/駆け引き」寄りとみなす簡易ヒューリスティック。
    goal_reach等、脱落が発生しないルールは別軸(サプライズ指数)側で評価する。

    2026-09-09、ユーザー指示: 「プレイヤー同士が一度も干渉(衝突)しないまま、各自が
    バラバラに穴へ落ちる/脱落する」だけの決着は退屈なので、駆け引きが一切なかった
    「無干渉の自滅」として最低スコアに分類する(instant/no_eliminationより明確に低くする)。
    """
    elim_count = len(result.elimination_order)
    total_frames = result.decided_frame or 1

    if not result.collisions:
        return "no_interaction", 0.1
    if total_frames < FPS * 8:
        return "instant", 0.5
    if elim_count == 0:
        return "no_elimination", 0.55
    late_fraction = sum(1 for e in result.elimination_order if e["frame"] > total_frames * 0.6) / elim_count
    if late_fraction > 0.5:
        return "comeback", 0.9
    if late_fraction > 0.25:
        return "back_and_forth", 0.75
    return "one_sided", 0.5


def _surprise_index(result: SimResult) -> float:
    """予測の裏切り度合いの連続値。衝突回数(接戦度合いの代理指標)を正規化して使う
    プレースホルダー実装。本来は視聴者の予測モデルが必要だが、このプロトタイプ段階では
    「衝突が多い=最後まで誰が勝つか読めない=サプライズが高い」とみなす。

    2026-09-09、ユーザー指示: 「サプライズ指数のハードルを引き上げる」対応として、
    満点に必要な衝突頻度を6回/秒→9回/秒に引き上げた(単調な軌道での脱落が多い候補を
    より厳しく減点し、順位変動・衝突が豊富なケースにのみ高スコアを付与する)。
    """
    if result.decided_frame is None or result.decided_frame == 0:
        return 0.0
    seconds = result.decided_frame / FPS
    collisions_per_sec = len(result.collisions) / seconds
    return min(1.0, collisions_per_sec / 9.0)


def _satisfaction_score(result: SimResult) -> float:
    """衝突・決着のテンポの良さ(密集しすぎ・間延びしすぎでないか)を衝突頻度から推定する。
    理想レンジはおおよそ2〜8回/秒という経験則ベースのプレースホルダー。
    """
    if result.decided_frame is None or result.decided_frame == 0:
        return 0.0
    seconds = result.decided_frame / FPS
    collisions_per_sec = len(result.collisions) / seconds
    if 2 <= collisions_per_sec <= 8:
        return 1.0
    if collisions_per_sec < 2:
        return max(0.0, collisions_per_sec / 2)
    return max(0.0, 1 - (collisions_per_sec - 8) / 20)


def _density_score(result: SimResult) -> float:
    """画面内の要素数(参加人数)が多すぎず少なすぎないか。"""
    if 3 <= result.n_circles <= 8:
        return 1.0
    if result.n_circles < 3:
        return 0.6
    return max(0.3, 1 - (result.n_circles - 8) * 0.1)


def _stagnation_score(result: SimResult) -> float:
    """一定時間、目立った変化(衝突・脱落・特殊能力発動)が起きない「間延び区間」がないかを検知する。
    3秒以内なら満点、それを超える最大ギャップが長いほど減点する。
    """
    if result.decided_frame is None or result.decided_frame == 0:
        return 0.0
    events = sorted(
        [e["frame"] for e in result.collisions]
        + [e["frame"] for e in result.elimination_order]
        + [e["frame"] for e in result.ability_events]
    )
    if not events:
        return 0.3
    gaps = []
    prev = 0
    for f in events:
        gaps.append(f - prev)
        prev = f
    gaps.append(result.decided_frame - prev)
    max_gap_seconds = max(gaps) / FPS
    if max_gap_seconds <= 3:
        return 1.0
    return max(0.0, 1 - (max_gap_seconds - 3) / 10)


# 2026-09-20、ユーザー指示: 「ドラマ評価フィルター」として3項目を追加。いずれも軽量シミュレーションの
# 座標ログ(result.frames)だけから算出でき、物理演算・レンダリングには一切影響しない
# オフラインの候補選定用スコア(既存のimpression/surprise等と同じ立ち位置)。

COVERAGE_TARGET_FRACTION = 0.70  # ステージ面積のうちこの割合を移動経路が使っていれば満点
CLOSE_FINISH_WINDOW_SECONDS = 3.0  # 決着直前、この秒数だけ遡って接近度を見る
COMEBACK_BOTTOM_FRACTION = 0.30  # 中間時点でこの割合以下の順位を「劣勢」とみなす


def _coverage_score(result: SimResult) -> float:
    """全プレイヤーの全フレームの座標履歴から凸包(Convex Hull)面積を求め、
    ステージ面積のうちどれだけを移動経路が使っているかを評価する(Coverage Score)。
    一箇所に固まって動かない・ステージの一部しか使わない候補より、画面全体を
    大きく使う候補を高評価にする狙い。点が3点未満(凸包を作れない)場合は中立スコア。
    """
    points = [(e["x"], e["y"]) for frame in result.frames for e in frame]
    if len(points) < 3:
        return 0.5
    try:
        hull_area = ConvexHull(np.array(points)).volume  # 2次元入力ではvolumeが面積になる(scipyの仕様)
    except Exception:
        return 0.3  # 全点が直線上に並ぶ等、退化したケース

    if result.shape == "circle":
        stage_area = math.pi * result.arena_half_x * result.arena_half_y
    else:
        stage_area = 4 * result.arena_half_x * result.arena_half_y
    if stage_area <= 0:
        return 0.5
    return min(1.0, (hull_area / stage_area) / COVERAGE_TARGET_FRACTION)


def _close_finish_score(result: SimResult) -> float:
    """残り2体になってから決着までの接近度合いを評価する(Close Finish Score)。
    一瞬の事故死(遠く離れた状態からいきなり決着)より、最後まで距離が縮まった
    せめぎ合いが続いた候補を高評価にする。残り2体という状態が一度も発生しない
    ルール/展開(goal_reachで複数人がゴールする等)では中立スコアを返す
    (「HP/サイズ」同様、全ルールに共通する自然な指標ではないため無理に評価しない)。
    """
    if result.decided_frame is None:
        return 0.5
    two_left_frame = next((i for i, frame in enumerate(result.frames) if len(frame) == 2), None)
    if two_left_frame is None:
        return 0.5

    window_start = max(two_left_frame, result.decided_frame - int(CLOSE_FINISH_WINDOW_SECONDS * FPS))
    stage_scale = max(result.arena_half_x, result.arena_half_y)
    # 区間の平均や決着フレームそのものの距離は使わない: 実測で「残り2体が物理的に
    # 周回・接近離反を繰り返し、決着自体は環境要因(穴・銃弾等)で離れた瞬間に起きる」
    # ケースが多く見られ、どちらも「終盤にどれだけ肉薄したか」を正しく表さなかった。
    # 区間内の最小距離(=最も肉薄した瞬間)を代表値として使う方が、実際のせめぎ合いの
    # 有無を安定して捉えられる(実測5シード×3ルールで最小距離0.16〜0.52の範囲に分布)。
    min_norm_dist = None
    for frame_idx in range(window_start, min(result.decided_frame + 1, len(result.frames))):
        frame = result.frames[frame_idx]
        if len(frame) != 2:
            continue
        a, b = frame
        dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / stage_scale
        if min_norm_dist is None or dist < min_norm_dist:
            min_norm_dist = dist
    if min_norm_dist is None:
        return 0.5
    # 正規化距離0(密着)で満点、0.5倍以上に一度も肉薄しなかったら最低点
    return max(0.0, min(1.0, 1 - min_norm_dist / 0.5))


def _comeback_score(result: SimResult, impression_score: float) -> float:
    """中間時点(50%経過)で劣勢だったエンティティが最終的に勝利したかを評価する
    (Comeback Score)。既存の`_impression_pattern`(脱落タイミングの分布から
    「決着がいつまでもつれたか」を見る)とは異なり、勝者本人が劣勢から
    這い上がったかを直接判定するため、より精度が高い。

    「HP/サイズ」という指標はabsorb_growth(現在のradius=サイズ)以外には
    自然な形で存在しないため、他ルールでは安易な代理指標を導入せず、
    既存のimpression_scoreをそのまま使う(呼び出し側から渡してもらう)。
    """
    if result.rule != "absorb_growth" or result.winner_id is None or not result.frames:
        return impression_score
    mid_frame = result.frames[len(result.frames) // 2]
    if len(mid_frame) < 2:
        return impression_score
    ranked = sorted(mid_frame, key=lambda e: e["radius"])
    cutoff = max(1, int(len(ranked) * COMEBACK_BOTTOM_FRACTION))
    bottom_ids = {e["id"] for e in ranked[:cutoff]}
    return 1.0 if result.winner_id in bottom_ids else impression_score


def evaluate_candidate(result: SimResult, threshold: float = DEFAULT_THRESHOLD) -> ScoreBreakdown:
    """軽量シミュレーション結果を基準判定にかけ、合否と各項目のスコアを返す。
    90秒経っても決着しない候補(reached_max_duration_without_decision)は無条件で不合格にする。
    """
    summary = summarize(result)
    first_collision_frame = _first_collision_frame(result)

    if summary["reached_max_duration_without_decision"]:
        return ScoreBreakdown(
            impression_pattern="undecided",
            impression_score=0.0,
            surprise_score=0.0,
            duration_score=0.0,
            duration_category=None,
            satisfaction_score=0.0,
            density_score=_density_score(result),
            stagnation_score=0.0,
            drama_score=0.0,
            coverage_score=0.0,
            comeback_score=0.0,
            close_finish_score=0.0,
            win_balance_score=1.0,
            brand_score=1.0,
            overall=0.0,
            passed=False,
            reject_reason="90秒以内に決着しなかった",
            first_collision_frame=first_collision_frame,
        )

    pattern, impression_score = _impression_pattern(result)
    surprise = _surprise_index(result)
    duration_category = _classify_duration(summary["decision_seconds"])
    duration_score = 1.0 if duration_category else 0.2
    satisfaction = _satisfaction_score(result)
    density = _density_score(result)
    stagnation = _stagnation_score(result)
    coverage = _coverage_score(result)
    close_finish = _close_finish_score(result)
    comeback = _comeback_score(result, impression_score)
    drama = (coverage + close_finish + comeback) / 3
    win_balance = win_balance_score()  # scripts_templates/geometry_battle_character_stats.jsonの累積戦績から算出
    brand = 1.0  # プレースホルダー(ブランド定義未実装)

    overall = (
        impression_score * SCORE_WEIGHTS["impression"]
        + surprise * SCORE_WEIGHTS["surprise"]
        + duration_score * SCORE_WEIGHTS["duration"]
        + satisfaction * SCORE_WEIGHTS["satisfaction"]
        + density * SCORE_WEIGHTS["density"]
        + stagnation * SCORE_WEIGHTS["stagnation"]
        + drama * SCORE_WEIGHTS["drama"]
        + win_balance * SCORE_WEIGHTS["win_balance"]
        + brand * SCORE_WEIGHTS["brand"]
    )
    passed = overall >= threshold

    gate_lo, gate_hi = PRODUCTION_DECISION_SECONDS_RANGE
    within_gate = gate_lo <= summary["decision_seconds"] <= gate_hi
    reject_reason = None
    if not passed:
        reject_reason = "総合スコアがしきい値未満"
    if not within_gate:
        # スコアが高くても無条件で不合格にする(2026-09-08、ユーザー指示の暫定フィルター)
        passed = False
        reject_reason = (
            f"決着まで{summary['decision_seconds']:.1f}秒"
            f"({gate_lo:.0f}〜{gate_hi:.0f}秒の暫定フィルター範囲外)"
        )
    if pattern == "no_interaction":
        # 2026-09-09、ユーザー指示: 総合スコアが閾値を超えていても、プレイヤー同士が
        # 一度も衝突しないまま決着した「無干渉の自滅」は無条件で不採用にする
        # (尺フィルターと同種の無条件足切り。他項目の加点で相殺されて合格してしまうのを防ぐ)。
        passed = False
        reject_reason = "プレイヤー同士の衝突が一度も発生しない無干渉の自滅だった"

    max_first_impact_frame = int(FIRST_IMPACT_MAX_SECONDS * FPS)
    if first_collision_frame is None or first_collision_frame > max_first_impact_frame:
        # 2026-09-12、序盤密度向上の無条件足切り(上記と同種)。落下・漂流だけで始まる
        # 「序盤停滞」シードは、他項目のスコアが高くても不採用にする。
        passed = False
        actual = "衝突なし" if first_collision_frame is None else f"{first_collision_frame / FPS:.2f}秒"
        reject_reason = f"序盤{FIRST_IMPACT_MAX_SECONDS:.1f}秒以内に最初の衝突が発生しなかった(実際: {actual})"

    if _hole_escape_dominant(result):
        # 2026-09-20、ユーザー指示: hole_fall以外で「そのルール独自の脱落」より「穴/端からの
        # 場外脱落」の方が多い試合は無条件で不採用にする(weapon_colosseumに限らず全ルール対象)。
        passed = False
        reject_reason = "そのルール独自の脱落より、穴/端からの場外脱落の方が多かった"

    if not _late_climax_ok(result):
        # 2026-09-21、ユーザー指示「レイト・クライマックス・フィルター」: 決着直前の脱落が
        # 試合の早い段階で終わっており、その後は消化試合になっていたと判断し無条件で不採用にする。
        passed = False
        reject_reason = "決着直前の脱落が試合の早い段階で終わっており、消化試合になっていた"

    return ScoreBreakdown(
        impression_pattern=pattern,
        impression_score=round(impression_score, 3),
        surprise_score=round(surprise, 3),
        duration_score=round(duration_score, 3),
        duration_category=duration_category,
        satisfaction_score=round(satisfaction, 3),
        density_score=round(density, 3),
        stagnation_score=round(stagnation, 3),
        drama_score=round(drama, 3),
        coverage_score=round(coverage, 3),
        comeback_score=round(comeback, 3),
        close_finish_score=round(close_finish, 3),
        win_balance_score=win_balance,
        brand_score=brand,
        overall=round(overall, 3),
        passed=passed,
        reject_reason=reject_reason,
        first_collision_frame=first_collision_frame,
    )
