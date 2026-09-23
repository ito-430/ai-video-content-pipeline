"""SimuSphere Arena(4ch目)の新規「組み合わせアイデア」生成エンジン。

ユーザー要望(2026-09-09)「特殊能力・ステージ・ルール等の新しいアイデアがいろいろ出てくる
仕組み」への対応。ただし実際に安全に自動化できるのは、既存の物理演算コード(simulate())が
既に受け付けるパラメータの新しい組み合わせを提案させることまでで、win-conditionそのもの
(rule)を新規のPythonロジックとして自動生成することはしない(コード生成をノーレビューで
実行に回すのは事故った時の被害が大きすぎるため)。

その代わり、simulate()が既に持つ非常に広いパラメータ空間
(rule×shape×gravity/elasticity/friction/damping×rotation_speed×wind_force×
accel_zone×trap×特殊能力の効き目(ability_params)×palette)の中から、
Geminiに「面白そうな新しい組み合わせ」を複数提案させ、以下の関門を通したものだけを
「コンボプリセット」としてプールに追加する。

1. 数値レンジのクランプ(ABILITY_PARAM_SCHEMA・本ファイルのSAFE_RANGES外の値は丸める)
2. NGワードチェック(名前・キャプション・タイトルテンプレート等の文字列項目)
3. 新規性チェック(既存プールとの完全一致は追加しない)
4. **シミュレーションによる実地検証**: 複数シードで軽量シミュレーション+基準判定を実行し、
   「30〜60秒(ごくたまに数秒の即決着もあり)で決着する」「気持ちよい」「退屈しない」という
   要件を実際に満たすかどうかを確認する。これを通らない提案は本番投入しない
   (Gemini自身の"面白そう"という主観だけを信用しない設計)。

コスト: gemini-flash-lite-latest(既存のKPT生成等と同モデル)で1回のバッチ生成につき
数百〜千数百トークン程度。既存の運用コストと比べて無視できる規模。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_provider import get_text_provider
from geometry_battle_gen import (
    ABILITY_PARAM_SCHEMA,
    _ARENA_SCALE,
    DASH_SPEED_BOOST,
    DEFAULT_DAMPING,
    DEFAULT_ELASTICITY,
    DEFAULT_FRICTION,
    DEFAULT_GRAVITY,
    GROWTH_SURGE_DURATION,
    GROWTH_SURGE_MULT,
    POISON_WIRE_DURATION,
    POISON_WIRE_LENGTH_FACTOR,
    SHOCKWAVE_IMPULSE,
    SHOCKWAVE_RADIUS,
    SLAM_BOOST,
    SPEED_BOOST_BASE,
    SPEED_BOOST_MULT,
    VORTEX_PULL_FORCE,
    VORTEX_RADIUS,
    simulate,
)
from geometry_battle_scoring import evaluate_candidate
from ng_word_filter import check_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = PROJECT_ROOT / "scripts_templates" / "geometry_battle_theme_pool.json"
MODEL_NAME = "gemini-flash-lite-latest"

RULES = ["hole_fall", "goal_reach", "area_control", "absorb_growth"]
SHAPES = ["square", "circle"]
PALETTES = ["vivid", "pastel", "neon", "sunset"]

# 数値パラメータの安全なレンジ。既存チューニング済みデフォルト(project_geometry_battle_channel
# メモリ参照)を中心に、破綻しない範囲でGeminiに提案させる。ability_paramsはABILITY_PARAM_SCHEMA
# (geometry_battle_gen.py)を流用する。gravity/windは2026-09-09の枠縮小(_ARENA_SCALE)に
# 合わせて比例縮小している。
SAFE_RANGES = {
    "gravity": (60.0 * _ARENA_SCALE, 900.0 * _ARENA_SCALE),
    "elasticity": (0.85, 1.0),
    "friction": (0.0, 0.4),
    "damping": (0.995, 1.0),
    "rotation_speed": (0.0, 0.45),
    "wind_x": (-260.0 * _ARENA_SCALE, 260.0 * _ARENA_SCALE),
    "wind_y": (-150.0 * _ARENA_SCALE, 150.0 * _ARENA_SCALE),
}

# 各能力パラメータの「中立値」(既定の効き目)。失敗した候補をデフォルト寄りに弱めて
# 再検証する際の基準点として使う(SAFE_RANGESのgravity等と同じ役割)。
ABILITY_PARAM_DEFAULTS = {
    "dash": {"speed_boost": DASH_SPEED_BOOST},
    "poison_wire": {"duration": POISON_WIRE_DURATION, "length_factor": POISON_WIRE_LENGTH_FACTOR},
    "shockwave": {"radius": SHOCKWAVE_RADIUS, "impulse": SHOCKWAVE_IMPULSE},
    "vortex": {"radius": VORTEX_RADIUS, "pull_force": VORTEX_PULL_FORCE},
    "growth_surge": {"mult": GROWTH_SURGE_MULT, "duration": GROWTH_SURGE_DURATION},
    "speed_boost": {"mult": SPEED_BOOST_MULT, "base": SPEED_BOOST_BASE},
    "slam": {"boost": SLAM_BOOST},
}
NEUTRAL_VALUES = {
    "gravity": DEFAULT_GRAVITY,
    "elasticity": DEFAULT_ELASTICITY,
    "friction": DEFAULT_FRICTION,
    "damping": DEFAULT_DAMPING,
    "rotation_speed": 0.0,
    "wind_x": 0.0,
    "wind_y": 0.0,
}

VALIDATION_SEEDS = [101, 202, 303, 404, 505, 606]
# ユーザー要件「30〜60秒(ごくたまに数秒)」の"ごくたまに"は少数派であるべきなので、
# 「大半のシードが主要レンジに収まること」を要求する狭いレンジにしている
# (2026-09-09、当初(5,65)秒という緩すぎるレンジで検証した結果、採用された全候補の
# 平均決着秒が20秒未満になってしまったため、実際の要件に合わせて絞り込んだ)。
PRIMARY_DECISION_SECONDS_RANGE = (25.0, 60.0)
PASS_RATE_THRESHOLD = 0.65  # 検証シードの65%以上が主要レンジ+基準判定を通れば採用(残りは"ごくたまに"の即決着として許容)
DAMPEN_FACTORS = [0.5, 0.25, 0.1]  # 弱め再検証で順に試す係数(既定値からの変化量に掛ける倍率)

COMBO_ABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "ability": {"type": "string", "enum": list(ABILITY_PARAM_SCHEMA.keys())},
        "param": {"type": "string"},
        "value": {"type": "number"},
    },
    "required": ["ability", "param", "value"],
}

COMBO_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "この組み合わせの短い英語の名前(2-4語)"},
        "caption": {"type": "string", "description": "画面上部に出す全て大文字の短い英語キャプション(例: LAST ONE STANDING WINS)"},
        "title_template": {
            "type": "string",
            "description": "YouTube Shorts用の英語タイトルテンプレート。参加数を表す{n}を1箇所含めること",
        },
        "flavor_reasoning": {"type": "string", "description": "なぜこの組み合わせが面白い/満足感があると考えたかの短い説明(日本語でよい)"},
        "rule": {"type": "string", "enum": RULES},
        "shape": {"type": "string", "enum": SHAPES},
        "palette": {"type": "string", "enum": PALETTES},
        "gravity": {"type": "number"},
        "elasticity": {"type": "number"},
        "friction": {"type": "number"},
        "damping": {"type": "number"},
        "rotation_speed": {"type": "number"},
        "wind_x": {"type": "number"},
        "wind_y": {"type": "number"},
        "trap": {"type": "boolean"},
        "accel_zone": {"type": "boolean"},
        "ability_overrides": {
            "type": "array",
            "description": "既存の特殊能力の効き目を変えるバリアント(0〜2件。新しい能力の種類は作れない)",
            "items": COMBO_ABILITY_SCHEMA,
        },
    },
    "required": [
        "name", "caption", "title_template", "flavor_reasoning", "rule", "shape", "palette",
        "gravity", "elasticity", "friction", "damping", "rotation_speed", "wind_x", "wind_y",
        "trap", "accel_zone", "ability_overrides",
    ],
}

BATCH_SCHEMA = {
    "type": "object",
    "properties": {"combos": {"type": "array", "items": COMBO_SCHEMA}},
    "required": ["combos"],
}

SYSTEM_PROMPT = f"""\
あなたは幾何学図形の物理演算バトル動画チャンネル(SimuSphere Arena)のゲームデザイナーです。
既存の物理演算パラメータの新しい組み合わせを提案し、動画のバリエーションを増やしてください。

## 前提(既存の実装)
- ルール(rule): hole_fall(最後の1体まで)/goal_reach(ゴール到達)/area_control(安全地帯が縮小)/
  absorb_growth(接触で吸収して成長)の4種類が既に実装済み。**新しい勝敗条件のルール自体は
  提案しないでください**(実際のコードが存在しないため実行できません)。既存4ルールの中から
  選び、物理パラメータの組み合わせで新しい体験を作ってください
- 物理パラメータ: gravity(重力、既定{DEFAULT_GRAVITY})、elasticity(反発係数、既定{DEFAULT_ELASTICITY})、
  friction(摩擦係数、既定{DEFAULT_FRICTION})、damping(空気抵抗、既定{DEFAULT_DAMPING})、
  rotation_speed(枠の回転速度)、wind_x/wind_y(常時吹く風の力、新規実装)
- ステージギミック: trap(内部に触れたら即脱落する穴)、accel_zone(下部の上向き発射台ゾーン。
  元々goal_reach専用だったが今回どのルールでも有効化できるようにした)
- 特殊能力(8種、色ごとに固定): dash/poison_wire/shockwave/vortex/growth_surge/teleport/
  speed_boost/slamが既に実装済み。**新しい種類の能力は提案しないでください**。代わりに
  ability_overridesで既存能力の「効き目」だけを変えたバリアント(例: shockwaveの範囲・威力を
  大きくした"Mega Shockwave")を0〜2件まで提案できます。**相手を強制的に減速させるだけの
  効果は今後も導入しない方針です**(終盤の見栄えを悪くするとのユーザー判断、2026-09-09)。
  vortexは「相手を自分の方へ引き寄せる」効果で、これは減速ではなく衝突を誘発する方向の
  効果なので対象外です

## 絶対厳守の要件(すべての提案がこれを満たす必要がある)
1. **30〜60秒程度で決着すること**(ごくたまに数秒で即決着するのは面白い演出として許容するが、
   基本はこのレンジを狙うこと)
2. **気持ちよい決着であること**(間延びせず、駆け引き・逆転の余地があること)
3. **見ていて退屈しないこと**(衝突やイベントが一定間隔で起き続けること)
これらは提案後に実際のシミュレーションで検証されるため、あなたの提案が外れていてもその提案は
不採用になるだけです。ただし、以下の点は強く意識してください(過去の試行で「派手さを狙いすぎて
決着が5〜10秒程度になってしまう」失敗が多発したため):

- **wind_x/wind_y・rotation_speed・特殊能力の効き目(ability_overrides)を強くするほど決着は
  速くなりやすい**。1つの組み合わせの中でこれらを同時に複数・大きく変更すると、ほぼ確実に
  数秒で終わってしまう
- **1つの組み合わせにつき、大胆に変えるパラメータは1〜2個までに絞り、残りは既定値に近い
  控えめな値のままにすること**。「主役の変化を1つ決めて、それ以外は脇役として既定値付近に
  留める」くらいの意識でちょうどよい
- gravityを既定値から大きく下げる(absorb_growthは既定80が既にその調整済み値)、elasticity/
  dampingを1.0に寄せる(area_controlは既定1.0/1.0が調整済み値)といった「決着を遅らせる」方向の
  調整も積極的に使ってよい

## 命名・キャプションについて
YouTubeでの公開を前提とするため、暴力的・攻撃的・センシティブな表現は避け、
ポップでキャッチーな英語表現にすること。
"""


@dataclass
class ValidationResult:
    pass_rate: float
    mean_overall: float
    mean_decision_seconds: float
    sample_scores: list[dict]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sanitize_combo(raw: dict) -> dict:
    """数値項目を安全なレンジへクランプし、ability_overridesも同様に丸める。"""
    combo = dict(raw)
    for key in ("gravity", "elasticity", "friction", "damping", "rotation_speed"):
        lo, hi = SAFE_RANGES[key]
        combo[key] = _clamp(float(combo.get(key, 0.0)), lo, hi)
    combo["wind_x"] = _clamp(float(combo.get("wind_x", 0.0)), *SAFE_RANGES["wind_x"])
    combo["wind_y"] = _clamp(float(combo.get("wind_y", 0.0)), *SAFE_RANGES["wind_y"])

    sanitized_overrides = []
    for ov in combo.get("ability_overrides") or []:
        ability = ov.get("ability")
        param = ov.get("param")
        schema = ABILITY_PARAM_SCHEMA.get(ability, {})
        if param not in schema:
            continue  # 未知のability/paramの組み合わせは無視(Geminiの幻覚対策)
        lo, hi = schema[param]
        sanitized_overrides.append({"ability": ability, "param": param, "value": _clamp(float(ov.get("value", lo)), lo, hi)})
    combo["ability_overrides"] = sanitized_overrides
    return combo


def _is_safe_text(combo: dict) -> bool:
    for field in ("name", "caption", "title_template", "flavor_reasoning"):
        if check_text(str(combo.get(field, ""))):
            print(f"[警告] NGワード検出のため却下: {field}={combo.get(field)!r}", file=sys.stderr)
            return False
    return True


def _combo_key(combo: dict) -> str:
    """新規性チェック用のキー。数値は丸めて微小な差異を同一視する。"""
    ability_key = sorted((o["ability"], o["param"], round(o["value"], 1)) for o in combo.get("ability_overrides") or [])
    return json.dumps(
        {
            "rule": combo["rule"],
            "shape": combo["shape"],
            "gravity": round(combo["gravity"], -1),
            "elasticity": round(combo["elasticity"], 2),
            "friction": round(combo["friction"], 2),
            "damping": round(combo["damping"], 3),
            "rotation_speed": round(combo["rotation_speed"], 2),
            "wind": (round(combo["wind_x"], -1), round(combo["wind_y"], -1)),
            "trap": combo["trap"],
            "accel_zone": combo["accel_zone"],
            "ability_overrides": ability_key,
        },
        sort_keys=True,
    )


def _dampen_combo(combo: dict, factor: float = 0.5) -> dict:
    """検証に失敗した候補を、各数値項目を「中立値(既定の効き目)」方向へfactor分だけ
    引き戻して弱めたバージョンを作る(factor=0.5なら変化量を半分にする)。
    Geminiの提案が"面白そう"を狙うあまり過激になりがちで、そのままだと決着が
    速すぎる(実測で判明、2026-09-09)ため、弱めて再検証する救済策。
    trap/accel_zoneは数値ではなくON/OFFなので"半分"にできない。実測では、これらを
    ONにしたまま数値だけ弱めても改善しないケースが多かった(trap/accel_zoneそのものが
    不安定化の主因であることが多いため)ため、弱め再検証の際は一律でOFFに戻す
    (accel_zoneはgoal_reachなら元々自動でONになる挙動に合わせる)。"""
    dampened = dict(combo)
    for key, neutral in NEUTRAL_VALUES.items():
        dampened[key] = neutral + (combo[key] - neutral) * factor
    dampened["trap"] = False
    dampened["accel_zone"] = combo["rule"] == "goal_reach"

    dampened_overrides = []
    for ov in combo.get("ability_overrides") or []:
        neutral = ABILITY_PARAM_DEFAULTS.get(ov["ability"], {}).get(ov["param"], ov["value"])
        dampened_overrides.append(
            {"ability": ov["ability"], "param": ov["param"], "value": neutral + (ov["value"] - neutral) * factor}
        )
    dampened["ability_overrides"] = dampened_overrides
    return dampened


def _ability_params_from_overrides(overrides: list[dict]) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for ov in overrides:
        params.setdefault(ov["ability"], {})[ov["param"]] = ov["value"]
    return params


def validate_combo(combo: dict, seeds: list[int] = VALIDATION_SEEDS) -> ValidationResult:
    """複数シードで軽量シミュレーション+基準判定を実行し、実際に要件を満たすか検証する。"""
    ability_params = _ability_params_from_overrides(combo["ability_overrides"])
    scores = []
    for seed in seeds:
        result, _circles = simulate(
            n_circles=5,
            seed=seed,
            gravity=combo["gravity"],
            elasticity=combo["elasticity"],
            rotation_speed=combo["rotation_speed"],
            shape=combo["shape"],
            rule=combo["rule"],
            friction=combo["friction"],
            damping=combo["damping"],
            accel_zone=combo["accel_zone"],
            trap=combo["trap"],
            palette=combo["palette"],
            ability_params=ability_params,
            wind_force=(combo["wind_x"], combo["wind_y"]),
        )
        score = evaluate_candidate(result)
        decision_seconds = (result.decided_frame or 0) / 60.0
        in_range = PRIMARY_DECISION_SECONDS_RANGE[0] <= decision_seconds <= PRIMARY_DECISION_SECONDS_RANGE[1]
        scores.append(
            {
                "seed": seed,
                "overall": score.overall,
                "passed": bool(score.passed and in_range),
                "decision_seconds": round(decision_seconds, 1),
            }
        )

    pass_rate = sum(1 for s in scores if s["passed"]) / len(scores)
    mean_overall = sum(s["overall"] for s in scores) / len(scores)
    mean_decision_seconds = sum(s["decision_seconds"] for s in scores) / len(scores)
    return ValidationResult(pass_rate, mean_overall, mean_decision_seconds, scores)


def _load_pool() -> list[dict]:
    if POOL_PATH.exists():
        try:
            return json.loads(POOL_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_pool(pool: list[dict]) -> None:
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_raw_combos(n: int) -> list[dict]:
    provider = get_text_provider(MODEL_NAME)
    user_prompt = f"新しい組み合わせを{n}件、バリエーション豊かに提案してください(4種のルールに偏りなく)。"
    data = provider.generate_json(SYSTEM_PROMPT, user_prompt, BATCH_SCHEMA)
    return data.get("combos", [])


def generate_and_validate(n: int = 6) -> list[dict]:
    """Geminiで候補を生成し、クランプ→安全性→新規性→シミュレーション検証を経て
    採用されたものをプールに追加する。採用された候補一覧(検証結果込み)を返す。"""
    pool = _load_pool()
    existing_keys = {_combo_key(c) for c in pool}

    raw_combos = generate_raw_combos(n)
    accepted = []
    for raw in raw_combos:
        combo = _sanitize_combo(raw)

        if not _is_safe_text(combo):
            continue
        key = _combo_key(combo)
        if key in existing_keys:
            print(f"[却下:新規性] {combo['name']} は既存プールと重複しています。")
            continue

        validation = validate_combo(combo)
        print(
            f"[検証] {combo['name']} ({combo['rule']}/{combo['shape']}): "
            f"pass_rate={validation.pass_rate:.0%} 平均スコア={validation.mean_overall:.3f} "
            f"平均決着秒={validation.mean_decision_seconds:.1f}s"
        )
        if validation.pass_rate < PASS_RATE_THRESHOLD:
            # Geminiの提案が過激で決着が速すぎるケースが多いため(2026-09-09実測)、
            # 既定値方向に段階的に弱めながら再検証する。DAMPEN_FACTORSの順に試し、
            # 最初に基準を満たした時点で採用する(見つからなければ諦める)。
            for factor in DAMPEN_FACTORS:
                dampened = _dampen_combo(combo, factor=factor)
                retry_validation = validate_combo(dampened)
                print(
                    f"  └[弱め再検証 factor={factor}] pass_rate={retry_validation.pass_rate:.0%} "
                    f"平均スコア={retry_validation.mean_overall:.3f} 平均決着秒={retry_validation.mean_decision_seconds:.1f}s"
                )
                if retry_validation.pass_rate >= PASS_RATE_THRESHOLD:
                    combo, validation = dampened, retry_validation
                    break
            else:
                print(f"[却下:検証不通過(弱め後も)] {combo['name']}")
                continue
            key = _combo_key(combo)
            if key in existing_keys:
                print(f"[却下:新規性(弱め後)] {combo['name']} は既存プールと重複しています。")
                continue

        combo["id"] = key[:12] if len(key) >= 12 else key
        combo["origin"] = "gemini"
        combo["created_at"] = datetime.now(timezone.utc).isoformat()
        combo["validation"] = asdict(validation)
        combo["uses"] = 0
        combo["total_score"] = 0.0
        pool.append(combo)
        existing_keys.add(key)
        accepted.append(combo)

    if accepted:
        _save_pool(pool)
    return accepted


# 2026-09-21、ユーザー指示「バンディット選定の統合」対応: プールの各コンボは元々
# uses/total_score(平均reward=total_score/usesが実績)を持っていたが、これを実際に使って
# 選ぶロジックが無く「生成して眠らせるだけ」だった。イプシロン-グリーディ方式で、
# 探索(EPSILON確率でランダムに選ぶ。0回使用のコンボは常にこちらに含める=楽観的初期化)と
# 活用(残りの確率で実績平均が最も高いコンボを選ぶ)を両立する。
BANDIT_EPSILON = 0.3  # 探索確率(プールがまだ小さいため高めに設定。実績が溜まったら下げる想定)


def select_bandit_combo(rng) -> dict | None:
    """プールからバンディット選定で1件選ぶ。プールが空ならNoneを返す(呼び出し側は
    従来の完全ランダム生成にフォールバックすること)。"""
    pool = _load_pool()
    if not pool:
        return None

    unvisited = [c for c in pool if c.get("uses", 0) == 0]
    if unvisited and rng.random() < BANDIT_EPSILON:
        return rng.choice(unvisited)
    if rng.random() < BANDIT_EPSILON:
        return rng.choice(pool)

    def _avg_reward(c: dict) -> float:
        uses = c.get("uses", 0)
        if uses == 0:
            return float("inf")  # 未使用コンボは常に最優先で試す(楽観的初期化)
        return c.get("total_score", 0.0) / uses

    return max(pool, key=_avg_reward)


def record_combo_result(combo_id: str, score: float) -> None:
    """実際に候補として採用されシミュレーション評価されたコンボの実績(基準判定スコア)を
    プールに反映する。呼び出し側(daily_pipeline.select_candidate)が選ばれた候補について
    毎回呼ぶことを想定(採用されなかった候補は反映しない=「実際に使われた」ものだけを
    reward源にする)。"""
    pool = _load_pool()
    for c in pool:
        if c.get("id") == combo_id:
            c["uses"] = c.get("uses", 0) + 1
            c["total_score"] = c.get("total_score", 0.0) + score
            break
    else:
        return
    _save_pool(pool)


def combo_to_candidate_params(combo: dict, n_circles: int, seed: int) -> dict:
    """コンボプリセットを、daily_pipeline._random_candidate_paramsが返すのと同じ形の
    paramsディクショナリに変換する(simulate()にそのまま渡せる形+タイトル生成等に
    必要なメタ情報を含む)。match_type関連のキーは呼び出し側が個人戦として埋める
    (コンボプールはmatch_type導入(2026-09-21)より前の設計のため、対戦形式は
    プリセットに含めず常に個人戦の候補として提案する)。"""
    return {
        "n_circles": n_circles,
        "seed": seed,
        "shape": combo["shape"],
        "rule": combo["rule"],
        "player_shape": "circle",
        "match_type": "individual",
        "team_count": 2,
        "boss_abilities": None,
        "palette": combo["palette"],
        "gravity": combo["gravity"],
        "elasticity": combo["elasticity"],
        "friction": combo["friction"],
        "damping": combo["damping"],
        "rotation_speed": combo["rotation_speed"],
        "wind_force": (combo["wind_x"], combo["wind_y"]),
        "trap": combo["trap"],
        "accel_zone": combo["accel_zone"],
        "ability_params": _ability_params_from_overrides(combo.get("ability_overrides") or []),
        "terrain": None,
        "camera": "fixed",
        "slow_motion": True,
        "_combo_id": combo.get("id"),
        "_combo_name": combo.get("name"),
        "_combo_caption": combo.get("caption"),
        "_combo_title_template": combo.get("title_template"),
    }


def main() -> None:
    accepted = generate_and_validate(n=6)
    print(f"\n合計{len(accepted)}件の新規コンボをプールに追加しました。")
    for c in accepted:
        print(f"- {c['name']} ({c['rule']}/{c['shape']}): {c['flavor_reasoning']}")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "geometry_battle_idea_generator.py")
