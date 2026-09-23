"""
幾何学アニメーション物理演算バトルチャンネル(4ch目) キャラクター性(5章) プロトタイプ

3ch目(戦士バトルチャンネル)と同じ思想を、図形キャラクターにも適用する:
- 色番号(COLORSのインデックス)ごとに名前・タイプ(相性)を固定で割り当てる
- 動画をまたいだ戦績(登場数・勝利数)をJSONファイルに蓄積する
- 蓄積した勝率を基準判定ロジック(geometry_battle_scoring.py)の勝率バランススコアに使う

タイプ相性(三すくみ)は「大型は小型に強いが高速型に弱い」という設計方針を採用し、
POWER→CONTROL→SPEED→POWERの順で有利不利が回る。absorb_growthルールの吸収判定
(通常は半径が大きい方が勝つ)に、有利側は実効半径を割り増しするボーナスとして反映する。

投稿パイプラインには接続しない、技術検証用のスクリプト。
"""

from __future__ import annotations

import json
from pathlib import Path

CHARACTER_NAMES = {
    0: "Blaze",
    1: "Frost",
    2: "Sprout",
    3: "Bolt",
    4: "Venom",
    5: "Nova",
    6: "Ember",
    7: "Boulder",
}

# タイプ(三すくみ)。POWERはCONTROLに強いがSPEEDに弱い、という関係が一周する
CHARACTER_TYPE = {
    0: "POWER",   # Blaze / shockwave
    1: "CONTROL",  # Frost / vortex(2026-09-09、旧freezeから変更)
    2: "POWER",   # Sprout / growth_surge
    3: "SPEED",   # Bolt / dash
    4: "CONTROL",  # Venom / poison_wire
    5: "SPEED",   # Nova / teleport
    6: "SPEED",   # Ember / speed_boost
    7: "POWER",   # Boulder / slam
}

TYPE_ADVANTAGE = {"POWER": "CONTROL", "CONTROL": "SPEED", "SPEED": "POWER"}
TYPE_ADVANTAGE_BONUS = 0.18  # 有利側の実効半径を何割増しとして扱うか

STATS_PATH = Path(__file__).resolve().parent.parent / "scripts_templates" / "geometry_battle_character_stats.json"


def type_matchup_multiplier(attacker_color_index: int, defender_color_index: int) -> float:
    """attacker側から見た実効サイズ倍率。有利ならボーナス、不利ならペナルティ、五分なら1.0。"""
    attacker_type = CHARACTER_TYPE.get(attacker_color_index)
    defender_type = CHARACTER_TYPE.get(defender_color_index)
    if attacker_type is None or defender_type is None or attacker_type == defender_type:
        return 1.0
    if TYPE_ADVANTAGE.get(attacker_type) == defender_type:
        return 1.0 + TYPE_ADVANTAGE_BONUS
    if TYPE_ADVANTAGE.get(defender_type) == attacker_type:
        return 1.0 - TYPE_ADVANTAGE_BONUS
    return 1.0


def load_stats() -> dict:
    if not STATS_PATH.exists():
        return {}
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def save_stats(stats: dict) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def record_result(circles: list, winner_id: int | None) -> dict:
    """1本のシミュレーション結果を戦績データベースに反映する。
    circlesはgeometry_battle_gen.simulate()が返すCircleEntityのリスト。
    """
    stats = load_stats()
    for c in circles:
        name = CHARACTER_NAMES.get(c.id % len(CHARACTER_NAMES), f"Player{c.id}")
        entry = stats.setdefault(name, {"color_index": c.id % len(CHARACTER_NAMES), "appearances": 0, "wins": 0})
        entry["appearances"] += 1
        if winner_id is not None and c.id == winner_id:
            entry["wins"] += 1
    save_stats(stats)
    return stats


def win_balance_score(min_appearances: int = 5) -> float:
    """累積戦績から勝率バランススコア(0-1)を算出する。
    登場数が少ないキャラは統計的にまだ信頼できないため評価対象から除外する。
    全員の勝率が均等に近いほど高スコア、特定キャラに極端に偏っているほど低スコアになる。
    データが不足している(min_appearances未満のキャラしかいない)場合は中立の1.0を返す
    (基準判定全体を不当に落とさないため)。
    """
    stats = load_stats()
    eligible = [e for e in stats.values() if e["appearances"] >= min_appearances]
    if len(eligible) < 2:
        return 1.0

    win_rates = [e["wins"] / e["appearances"] for e in eligible]
    expected = 1.0 / len(eligible)
    # 期待勝率からの平均絶対偏差を正規化してスコア化(偏差が大きいほど低スコア)
    deviation = sum(abs(r - expected) for r in win_rates) / len(win_rates)
    max_deviation = expected * 2  # 1人が全勝・他が全敗に近い極端なケースを下限の目安にする
    return max(0.0, 1.0 - deviation / max_deviation)
