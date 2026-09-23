"""物理演算バトルチャンネル(4ch目)の日次自動投稿パイプライン。

geometry_battle_gen.py(技術検証プロトタイプ)・geometry_battle_scoring.py(基準判定)・
geometry_battle_characters.py(キャラクター性)を、実際の投稿へつなぐ本番エントリーポイント。

流れ:
1. ルール×形状×パラメータ×演出の組み合わせをCANDIDATE_COUNT件サンプリングし、
   軽量シミュレーション(simulate、描画なし)だけを実行する
2. 基準判定(evaluate_candidate)でスコアリングし、合格したものの中から、直近投稿と
   同じruleを避けつつ最高スコアの候補を選ぶ（多様性の弱い優先。1件も合格しなければ
   無投稿を避けるため全候補中の最高スコアで妥協する）
3. 選ばれた候補だけを本番レンダリング(render、音声付き)する
4. キャラクター戦績を更新し、YouTubeへアップロードし、動画ログに記録する

公開設定はpublicで開始する。投稿ごとのDiscord通知は行わない
（週次KPT報告のみ担当。geometry_battle_kpt.py参照）。失敗時は共通の#アラートへ通知する。
"""

import os
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from geometry_battle_characters import CHARACTER_NAMES, record_result
from geometry_battle_gen import ABILITY_BY_COLOR_INDEX, ARENA_SIZE_VARIANTS, COLORS, DEFAULT_GRAVITY, render, simulate
from geometry_battle_idea_generator import combo_to_candidate_params, record_combo_result, select_bandit_combo
from geometry_battle_scoring import evaluate_candidate
from geometry_battle_video_log import load_log, log_video
from geometry_posting_schedule import mark_posted
from youtube_upload import upload_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "geometry_battle"
CLIENT_SECRET_PATH = PROJECT_ROOT / "materials" / "youtube_client_secret_geometry.json"
TOKEN_PATH = PROJECT_ROOT / "materials" / "youtube_token_geometry.json"

CANDIDATE_COUNT = 30
# 2026-09-21、weapon_colosseum追加(剣/槍/ハンマー/弓矢/斧の5武器でHPを削り合う新ルール)。
# shape(square/circle)×arena_size(compact/spacious/tall)×n_circles(4〜7)×trap有無の
# 組み合わせを各40シードずつ実測し、全て未到達0件・エラー0件を確認した上で追加した
# (穴の空いたステージは使わない設計のため、terrain組み合わせの検証は不要)。
RULES = ["hole_fall", "goal_reach", "area_control", "absorb_growth", "gun_duel", "weapon_colosseum"]

# 情報過多によるスワイプ離脱を防ぐため、全ルールで参加人数の上限を4〜5体に固定する
# (従来は4〜7体、hourglassのみ4〜6体)。
N_CIRCLES_MAX = 5
# ルールごとの選定重み。推測ベースの調整ではなく、実績(視聴回数加重平均維持率)に基づいて
# 1ルールずつ検証しながら更新する方針にした。実績でhole_fallが最高・gun_duelが最低と
# 判明したため、以前入れていたgun_duel優遇(3.0)は実際のデータと逆方向だったため撤回し、
# goal_reachの抑制(0.3)も推測ベースの調整だったためデータ駆動で均等(1.0)に戻した。
# RULESに載っていないルールは重み1.0(均等)として扱う(_random_candidate_params参照)。
RULE_WEIGHTS = {
    "hole_fall": 5.0,  # 実績最高のため最大化
    "gun_duel": 0.1,  # 実績最低のため、原因検証用の単一テスト枠程度まで大幅に絞る(0にはしない)
}

# 対戦形式(match_type)は既存のrule(勝敗条件)とは独立した軸で、選ばれたruleが
# それぞれの対戦形式に対応している場合のみ抽選対象にする(absorb_growthは接触・
# 吸収成長がゲーム性の中心でチーム制と相性が悪いため両方とも対象外にしている)。
TEAM_COMPATIBLE_RULES = {"hole_fall", "goal_reach", "area_control", "gun_duel", "weapon_colosseum"}
BOSS_COMPATIBLE_RULES = {"weapon_colosseum", "gun_duel"}
TEAM_MATCH_PROBABILITY = 0.2  # ruleがteam対応の場合、この確率でmatch_type="team"にする
# 視覚的フックが最も強いボス戦の抽選確率を15%→45%に大幅引き上げた。
BOSS_MATCH_PROBABILITY = 0.45  # ruleがboss対応の場合、この確率でmatch_type="boss"にする(teamと排他)
TEAM_COUNTS = [2, 3]  # チーム戦は2〜3チームに分かれる構成のみ対応
# ボスの「2種類のスキル」は既存8種の特殊能力(色↔能力の対応表と同じ実体)から2つを組み合わせる
BOSS_ABILITY_POOL = list(ABILITY_BY_COLOR_INDEX.values())
# チーム戦は最低でも1チームあたり2人以上になるようteam_count*2を、ボス戦はボス以外に最低3人
# 対戦相手がいるよう調整する。n_circles上限を5に引き下げた(下記N_CIRCLES_MAX)のに合わせ、
# team_count=3の下限も6→5に引き下げている(2+2+1構成を許容する)。
TEAM_MIN_N_CIRCLES = {2: 4, 3: 5}
BOSS_MIN_N_CIRCLES = 4  # ボス1体+最低3人

SHAPES = ["square", "circle"]
PLAYER_SHAPES = ["circle", "square", "triangle"]
# 四角形プレイヤーも他の形状と同程度の確率で採用されるようにする対応。
# squareは「重力の影響を受けず常に直進、壁や相手には反射する」専用挙動
# (geometry_battle_gen._no_gravity_velocity_func)を導入したところ、goal_reach(重力に逆らわず
# 一直線でゴールに着くため速すぎて尺フィルターに落ちる)とhole_fall(重力がないため穴へ落ちず
# 未決着になりやすい)で極端に不利になることが判明し、まずこの2ルールでは除外した(採用する
# ルールはこの下のPLAYER_SHAPE_WEIGHTS_BY_RULEのキーで決まる)。
#
# 除外後もなお、静的な重み(合格率の逆数から手計算した値)を何度か試したが、「バッチ内で
# 最高スコア1件だけを選ぶ」という選定ロジックは合格率の平均値だけでは予測できない複雑な
# 挙動(スコア分布の裾/分散に左右される)を示し、机上の重み調整では狙った1/3ずつに収束
# しなかった(実測: circle 47%・triangle 40%・square 13%等、試すたびに結果が変わった)。
# そのため、事前に完璧な重みを推定することは諦め、下のPLAYER_SHAPE_WEIGHTS_BY_RULEは
# 単純な均等重み(フォールバック)にとどめ、実際の投稿実績(video_logのplayer_shape、直近の
# 採用シェア)を見ながら自動的に重みを補正するフィードバック方式に切り替えた
# (_adaptive_player_shape_weights参照)。データが少ない開幕直後はこの均等重みがそのまま使われる。
PLAYER_SHAPE_WEIGHTS_BY_RULE = {
    "goal_reach": {"circle": 1, "triangle": 1},  # squareは使わない(理由は上記コメント参照)
    "hole_fall": {"circle": 1, "triangle": 1},  # squareは使わない(理由は上記コメント参照)
    "area_control": {"circle": 1, "square": 1, "triangle": 1},
    "absorb_growth": {"circle": 1, "square": 1, "triangle": 1},
    "gun_duel": {"circle": 1, "square": 1, "triangle": 1},
    "weapon_colosseum": {"circle": 1, "square": 1, "triangle": 1},
}
# 直近何本分の投稿実績を見て補正するか。少なすぎると1〜2本の偏りに過剰反応し、
# 多すぎると最近の傾向(パラメータ調整の影響等)を反映しにくくなるためのバランス値。
PLAYER_SHAPE_ADAPT_LOOKBACK = 60
# 補正が効き始めるのに必要な最低投稿本数(そのルールでの実績)。これ未満は均等重みのまま。
PLAYER_SHAPE_ADAPT_MIN_SAMPLES = 6
# 補正倍率のクランプ範囲。極端な暴走(1本の偏りで重みが跳ね上がる等)を防ぐ。
PLAYER_SHAPE_ADAPT_MIN_MULT = 0.2
PLAYER_SHAPE_ADAPT_MAX_MULT = 5.0


def _player_shape_recent_share(lookback: int = PLAYER_SHAPE_ADAPT_LOOKBACK) -> dict[str, float]:
    """直近lookback本(全ルール合算)のplayer_shape採用シェアを返す。データ不足時は空dict。"""
    log = load_log()
    recent = [e for e in log if e.get("player_shape") in PLAYER_SHAPES][-lookback:]
    if len(recent) < PLAYER_SHAPE_ADAPT_MIN_SAMPLES:
        return {}
    counts = Counter(e["player_shape"] for e in recent)
    total = len(recent)
    return {s: counts.get(s, 0) / total for s in PLAYER_SHAPES}


def _player_shape_selection_bonus(player_shape: str) -> float:
    """採用シェアが目標(1/3)より低いplayer_shapeほど大きくなる補正係数(1.0が中立)。候補選定時の
    スコア比較にこれを掛け合わせて使う。2026-09-08、生成側の抽選重み(_adaptive_player_shape_weights)
    だけでは「バッチ内の最高スコア1件を選ぶ」という選定ロジックの複雑さ(スコア分布の裾/分散に
    左右され、生成数を増やしても選ばれる確率が線形に増えない)を打ち消せなかったため、選定の
    まさにその瞬間(比較・max())で直接補正する対策を追加した。"""
    shares = _player_shape_recent_share()
    if not shares:
        return 1.0
    target = 1.0 / len(PLAYER_SHAPES)
    actual = shares.get(player_shape, 0.0)
    correction = target / max(actual, 0.02)
    return min(PLAYER_SHAPE_ADAPT_MAX_MULT, max(PLAYER_SHAPE_ADAPT_MIN_MULT, correction))


def _adaptive_player_shape_weights(rule: str) -> dict[str, float]:
    """そのルールで使える各player_shapeの抽選重みを、直近の実際の採用実績(video_log)を見て
    補正する。実績シェアが目標(1/len(shapes))より低い形状は重みを上げ、高い形状は下げる。
    選定ロジックの複雑さを机上で見積もる代わりに、実際の結果を見ながら継続的に収束させる
    フィードバック方式(2026-09-08導入、上のコメント参照)。"""
    base_weights = PLAYER_SHAPE_WEIGHTS_BY_RULE[rule]
    shapes = list(base_weights.keys())
    log = load_log()
    recent = [e for e in log if e.get("rule") == rule and e.get("player_shape") in shapes]
    recent = recent[-PLAYER_SHAPE_ADAPT_LOOKBACK:]
    if len(recent) < PLAYER_SHAPE_ADAPT_MIN_SAMPLES:
        return base_weights
    counts = Counter(e["player_shape"] for e in recent)
    total = len(recent)
    target_share = 1.0 / len(shapes)
    adjusted = {}
    for s in shapes:
        actual_share = counts.get(s, 0) / total
        correction = target_share / max(actual_share, 0.02)
        mult = min(PLAYER_SHAPE_ADAPT_MAX_MULT, max(PLAYER_SHAPE_ADAPT_MIN_MULT, correction))
        adjusted[s] = base_weights[s] * mult
    return adjusted
PALETTES = ["vivid", "pastel", "neon", "sunset"]
# 通常のカメラワークは全体追従(tracking)ではなく固定を基本とする
# (決着時に勝者へズームインする演出は既存のエピローグ側で別途常時行われる、カメラ軸とは独立)。
# trackingの実装自体はgeometry_battle_gen.pyに残しているが、本番の候補生成では選ばない。
CAMERAS = ["fixed"]

# 実運用のチューニング結果に基づく、rule/shapeごとの推奨上書き値。
# 2026-09-09、枠の真円/正方形化(ARENA_HALF_EXTENT基準)に伴い枠のサイズが縮小されたため、
# 全ルールを実際にシミュレーションし直して数値を再調整した。あわせて2つの問題を発見・修正:
# - area_controlのelasticity/damping上書きが漏れていた(前回セッションで発覚・修正済み)
# - area_controlのhole_width上書きも漏れており、既定の穴から普通に落下してしまい
#   ゾーン縮小による決着(狙った尺40秒前後)がほぼ機能していなかった(今回発見)
# SHAPE_PARAM_OVERRIDESより後にRULE_PARAM_OVERRIDESを適用する(下のコード参照)ことで、
# area_controlのhole_width=0が「circle形状だからhole_widthを上書きする」設定より優先される。
RULE_PARAM_OVERRIDES = {
    "absorb_growth": {"gravity": 56.6},  # 尺15〜25秒ターゲットには既にほぼ適正(実測、変更不要と確認済み)
    "area_control": {"elasticity": 1.0, "damping": 1.0, "hole_width": 0.0},
    # gun_duelは「場外に出ない枠の中で戦う」仕様のため密閉必須。
    # 2026-09-09、尺短縮に伴いgravityも底上げ(タイマー短縮と合わせて接触頻度を上げる狙い。
    # それでも未到達率が高いまま残る既知の課題、geometry_battle_gen.py側のコメント参照)
    "gun_duel": {"hole_width": 0.0, "gravity": DEFAULT_GRAVITY * 1.5},
    # 2026-09-09、尺15〜25秒ターゲットへの短縮に伴い新規追加。重力2.0倍で決着を早める
    # (実際にシミュレーションを回して確認: 40シード中median23.9秒、未到達0件)
    "hole_fall": {"gravity": DEFAULT_GRAVITY * 2.0},
}
SHAPE_PARAM_OVERRIDES = {"circle": {"hole_width": 80.0}}

# 新ステージ内部構造(terrain、2026-09-09追加)。全terrain×全ruleの組み合わせを実際に
# シミュレーションで検証したところ、多くの組み合わせで「90秒経っても決着しない」が
# 頻発することが判明した(例: hourglass×gun_duelは20シード中19件が未到達、cross×hole_fallは
# 20シード中20件が未到達)。そのため無条件の総当たりにはせず、実測でundecided=0を確認できた
# 組み合わせだけを許可リスト化する。terrain=None(地形なし)は全ルールで常に選択可能(既存動作)。
TERRAIN_COMPATIBLE_RULES = {
    # (外枠を強制的にcircleへ変更した際に再検証): 円形外枠にすると
    # gun_duelだけ20シード中16件が未到達になる新規の不具合が判明したため除外した
    # (旧square外枠+反発係数1.3の設定では動作していたが、その設定自体が別バグだったため
    # 参考にならない。円形外枠+反発係数1.0の正しい設定での実測に基づく)
    # 「goal_reachの決着のほとんどが場外」という傾向を受けて実測したところ、
    # donut×goal_reachは中心の円形障害物がゴールへの直線的な経路を塞ぎ、プレイヤーが
    # ドーナツ状の狭い通路で衝突を繰り返して場外に弾かれる展開に偏っており、決着した
    # 候補の43%が場外(terrain=Noneなら20%)だった。除外する。
    "donut": ["hole_fall", "area_control", "absorb_growth"],
    "hourglass": ["area_control"],  # area_control以外は未到達率25〜95%で不採用
    "cross": ["area_control"],  # goal_reachも0件だったが尺のばらつきが大きく(5〜77秒)見送り
    "pegboard": ["area_control", "absorb_growth", "hole_fall"],  # hole_fallはTERRAIN_RULE_PARAM_OVERRIDES必須
    # 2026-09-13追加。バンパー/スリングショット/誘導スロープ/自動パルスフリッパーで構成する
    # 本格ピンボール型ステージ(geometry_battle_gen.py の terrain="pinball")。hole_fall/gun_duel/
    # goal_reachも試したが尺のばらつきが大きい・決着率が低いなどで未採用。area_controlは
    # 安全地帯の縮小という独立した決着メカニズムがあるため、pinballの混沌とした跳ね返りが
    # 決着の妨げにならず、20シード中0件未到達・尺17.7〜24.7秒(目標15-25秒にほぼ収まる)を
    # 実測で確認できたため採用した。
    "pinball": ["area_control"],
    # 新ステージとして追加した2種(gun_duel専用、
    # geometry_battle_gen.py参照)。どちらも武器(銃)の受け渡しが決着の唯一の手段のため、
    # gravity/BULLET_SPEED/照準ロジック(_lead_aim_angle)を合わせて実測チューニングした上で
    # 採用。素のgun_duel(terrain=None)自体が15〜25%程度は未到達になる既知の傾向を持つため、
    # それと同等以下(実測ではやや上回る)undecided率に収まることを確認して採用ラインとした。
    # 足場を長方形→山型(三角形)へ変更。隙間の調整も伴い
    # 実測undecided=9/40・尺中央値30.9秒に変化(素のgun_duelよりやや高いが許容範囲として採用)。
    "two_tier": ["gun_duel"],
    "split_horizontal": ["gun_duel"],  # 実測(40シード): undecided=5/40、尺中央値13.9秒
}
# 特定の(rule, terrain)組み合わせでのみ必要な追加パラメータ上書き。RULE_PARAM_OVERRIDESの後に適用する。
TERRAIN_RULE_PARAM_OVERRIDES = {
    # pegboardはピンで落下が遅くなるため、tallアリーナ+重力3.0倍でないと未到達が多発する
    # (実測: compact+gravity2.0倍のままだと20シード中18〜20件が未到達)
    ("hole_fall", "pegboard"): {"gravity": DEFAULT_GRAVITY * 3.0, "arena_size": "tall"},
    # pinballはtallアリーナ前提で設計(バンパー/スリングショット/フリッパーの配置比率が
    # 縦長基準)。重力2.0倍は実測で20シード中0件未到達・尺17.7〜24.7秒を確認した値。
    ("area_control", "pinball"): {"gravity": DEFAULT_GRAVITY * 2.0, "arena_size": "tall"},
    # 2026-09-20追加。gun_duelの通常上書き(gravity*1.5)のままだと2陣地に分かれる/落下する
    # 構造上、重力が強すぎて銃を拾う前に膠着することが判明(実測: gravity*1.5のままだと
    # two_tierは20シード中17件、split_horizontalは20シード中9件が未到達)。gravity*0.7に
    # 弱めることで大幅改善した。
    ("gun_duel", "two_tier"): {"gravity": DEFAULT_GRAVITY * 0.7, "arena_size": "tall"},
    ("gun_duel", "split_horizontal"): {"gravity": DEFAULT_GRAVITY * 0.7, "arena_size": "tall"},
}
TERRAIN_SELECTION_PROBABILITY = 0.4  # 対応ルールの場合でも60%は従来通りterrainなしにする
# pinballは実装済みだが対応terrain5種の均等抽選に埋もれ、
# area_control×terrain選択が発動する場合の1/5(全体では約0.4%)でしか選ばれず、実測でも
# 直近27投稿中0件だった。ビジュアル面で目立たせたい意図もあり、pinballだけ重みを上げる
# (未指定のterrainは重み1のまま)。
TERRAIN_SELECTION_WEIGHTS = {"pinball": 4}

# "public"固定にせず環境変数で切り替え可能にしておく（ユーザーが後日unlisted等に戻したい場合に
# コード変更なしで対応できるように。daily-publish.ymlのPUBLISH_PRIVACY変数と同じ考え方）。
PRIVACY_STATUS = os.environ.get("GEOMETRY_PUBLISH_PRIVACY", "public")

# area_controlのCTRが著しく低い(実測0.88%)ことが判明。CTRが良好なgoal_reach("{n} Shapes
# Race to the Top!")と同じ「目的語+動詞」構文に寄せ、旧来の説明的な文言("The Safe Zone
# is Shrinking...")から目的が瞬時にわかる文言へ変更する。単発の切り替えとして扱い(この
# チャンネルはA/Bテスト機能を持たないため、切り替え前後の期間比較で効果を見る)、
# 他のルールのタイトルは今回変更しない。
TITLE_TEMPLATES = {
    "hole_fall": "{n} Shapes Enter. Only 1 Survives.",
    "goal_reach": "{n} Shapes Race to the Top!",
    "area_control": "{n} Shapes Fight for the Zone!",
    "absorb_growth": "Eat or Be Eaten: {n} Shapes Battle",
    "gun_duel": "{n} Shapes, 1 Gun. Last One Standing.",
    "weapon_colosseum": "{n} Shapes Enter the Weapon Colosseum!",
}

RULE_LABELS = {
    "hole_fall": "Last one standing wins.",
    "goal_reach": "First to reach the goal wins.",
    "area_control": "Stay inside the shrinking safe zone.",
    "absorb_growth": "Absorb rivals to grow. Biggest survivor wins.",
    "gun_duel": "Grab the gun, aim, fire. Last one standing wins.",
    "weapon_colosseum": "Grab a weapon, drain their HP. Last one standing wins.",
}

TAGS_BY_RULE = {
    "hole_fall": ["elimination", "marble race"],
    "goal_reach": ["race", "obstacle course"],
    "area_control": ["battle royale", "shrinking zone"],
    "absorb_growth": ["agar.io", "growth battle"],
    "gun_duel": ["battle royale", "shooter"],
    "weapon_colosseum": ["battle royale", "weapon fight"],
}

DESCRIPTION_TEMPLATE = """\
Geometric shapes battle it out with real physics simulation. No scripts, no fakes \
- every match is decided by physics.

Rule: {rule_label}

#shorts #physicssimulation #satisfying
"""

CATEGORY_ID = "24"  # Entertainment

# 8-5「意図的な期待の裏切り(視覚ミスリード)」: 一定割合の動画で1体だけ、実際の半径・当たり判定は
# 変えずに見た目の大きさだけをずらす(大きく強そうに見えて実力は普通/小さく弱そうに見えて実力は普通)。
# 視聴者の予測を裏切る決着はコメント・シェアの誘発につながるという想定(5章)。
# 2026-09-09、尺15〜25秒への短縮に伴い、短い尺の中でも「えっ、そっちが勝つの?」という
# カタルシスを増やす狙いで0.35→0.45に引き上げた。
VISUAL_MISMATCH_PROBABILITY = 0.45
VISUAL_MISMATCH_SCALES = [1.5, 0.65]


def _random_match_type(rng: random.Random, rule: str) -> tuple[str, int | None, tuple[str, str] | None]:
    """(match_type, team_count, boss_abilities)を返す。ruleがそれぞれの対戦形式に
    対応していない場合は必ず"individual"になる(TEAM_COMPATIBLE_RULES/BOSS_COMPATIBLE_RULES
    参照)。両方に対応するrule(gun_duel/weapon_colosseum)ではteamの抽選を先に行う。"""
    if rule in TEAM_COMPATIBLE_RULES and rng.random() < TEAM_MATCH_PROBABILITY:
        return "team", rng.choice(TEAM_COUNTS), None
    if rule in BOSS_COMPATIBLE_RULES and rng.random() < BOSS_MATCH_PROBABILITY:
        return "boss", None, tuple(rng.sample(BOSS_ABILITY_POOL, 2))
    return "individual", None, None


def _random_candidate_params(rng: random.Random) -> dict:
    rule_weights = [RULE_WEIGHTS.get(r, 1.0) for r in RULES]
    rule = rng.choices(RULES, weights=rule_weights, k=1)[0]
    match_type, team_count, boss_abilities = _random_match_type(rng, rule)
    shape = rng.choice(SHAPES)
    weights = _adaptive_player_shape_weights(rule)
    player_shape = rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
    compatible_terrains = [t for t, rules in TERRAIN_COMPATIBLE_RULES.items() if rule in rules]
    if compatible_terrains and rng.random() < TERRAIN_SELECTION_PROBABILITY:
        terrain_weights = [TERRAIN_SELECTION_WEIGHTS.get(t, 1) for t in compatible_terrains]
        terrain = rng.choices(compatible_terrains, weights=terrain_weights, k=1)[0]
    else:
        terrain = None
    if terrain == "donut":
        # ドーナツ型は外枠も円でないと成立しない
        # (中心の円形障害物+外枠が正方形だと「ドーナツ」に見えない)
        shape = "circle"
    if terrain == "pinball":
        # 2026-09-13追加: スリングショット/誘導スロープの座標は正方形の平らな壁を前提に
        # 計算している(円形外枠だと隅が丸まり、スリングショットの基点が実際の壁の外に
        # はみ出す)。実測もsquareでのみ行っているため、pinball選択時はshapeを固定する。
        shape = "square"
    # 2026-09-09に発覚: hourglassのネック(くびれ)付近で7体が密集して衝突すると、
    # まれに薄い壁を1フレームですり抜けて枠外に出てしまう(トンネリング)。ネックを広げる
    # (CHOKE_WIDTH_RATIO)対応と合わせ、密集の引き金になる最大人数も6に制限して完全に解消した
    # (実測: n_circles<=6+ネック拡大で60万フレーム超のチェックで0件)。
    # 情報過多によるスワイプ離脱を防ぐため、参加人数を全ルール最大4〜5体に固定する。
    # 上記のhourglass専用上限(6)よりさらに絞ってN_CIRCLES_MAXを一律の上限として適用する。
    n_circles_max = min(N_CIRCLES_MAX, 6 if terrain == "hourglass" else 7)
    # 2026-09-21、match_type新軸対応: チーム戦は1チーム最低2人・ボス戦はボス以外最低3人を
    # 確保できるようn_circlesの下限を引き上げる(通常の下限4はそのまま、必要な場合のみ増やす)。
    n_circles_min = 4
    if match_type == "team":
        n_circles_min = max(n_circles_min, TEAM_MIN_N_CIRCLES.get(team_count, 4))
    elif match_type == "boss":
        n_circles_min = max(n_circles_min, BOSS_MIN_N_CIRCLES)
    n_circles_min = min(n_circles_min, n_circles_max)  # 安全策(通常はn_circles_max=6/7を下回らない)
    params = {
        "n_circles": rng.randint(n_circles_min, n_circles_max),
        "seed": rng.randint(1, 10_000_000),
        "shape": shape,
        "rule": rule,
        "player_shape": player_shape,
        "match_type": match_type,
        "team_count": team_count if match_type == "team" else 2,
        "boss_abilities": boss_abilities,
        "palette": rng.choice(PALETTES),
        "camera": rng.choice(CAMERAS),
        # 決着の瞬間のスローモーションは「気持ちの良い決着」演出の一部として毎回入れる
        # (以前の50%抽選だと入らない回があり、決着インパクトが弱くなっていた)
        "slow_motion": True,
        # 2026-09-09: terrain選択時はtrap(別の内部ハザード)との組み合わせを検証していないため、
        # 未検証の複合を避けて無効化する(terrainなしの場合は従来通り20%で有効)
        "trap": (rng.random() < 0.2) if terrain is None else False,
        "rotation_speed": rng.uniform(0.15, 0.4) if rng.random() < 0.2 else 0.0,
        "arena_size": rng.choice(list(ARENA_SIZE_VARIANTS)),
        "terrain": terrain,
    }
    # 2026-09-09: SHAPE→RULE→TERRAINの順で適用する(より具体的な条件を後に適用し、必要な上書きが
    # 潰されないようにする。例: pegboard×hole_fallのgravity/arena_size上書きはRULE_PARAM_OVERRIDESの
    # hole_fall用gravityより優先されるべき)
    params.update(SHAPE_PARAM_OVERRIDES.get(shape, {}))
    params.update(RULE_PARAM_OVERRIDES.get(rule, {}))
    params.update(TERRAIN_RULE_PARAM_OVERRIDES.get((rule, terrain), {}))
    return params


def _last_used_rule() -> str | None:
    log = load_log()
    return log[-1]["rule"] if log else None


JST = timezone(timedelta(hours=9))


def _rules_used_today() -> set[str]:
    """JST基準の「今日」に投稿済みの動画のruleの集合を返す。2026-09-09、1日3本投稿化に
    伴い、同じ日に全く同じruleの動画が並ばないようにするため導入。"""
    log = load_log()
    today_jst = datetime.now(timezone.utc).astimezone(JST).date()
    used = set()
    for e in log:
        published_at = e.get("published_at")
        if not published_at:
            continue
        try:
            dt = datetime.fromisoformat(published_at)
        except ValueError:
            continue
        if dt.astimezone(JST).date() == today_jst:
            used.add(e["rule"])
    return used


# 5秒等の即決着版が公開されてしまう問題への対応。
# evaluate_candidate()が20〜40秒の尺フィルター(geometry_battle_scoring.PRODUCTION_DECISION_SECONDS_RANGE)
# を無条件の足切りとして持つようになったため、1バッチで全滅した場合はすぐに妥協せず
# 最大この回数まで再サンプリングする(軽量シミュレーションのみなのでコストは小さい)。
MAX_CANDIDATE_BATCHES = 3

# 候補の一部を、Geminiが提案し実地シミュレーションで検証済みのコンボプール
# (geometry_battle_idea_generator.py)からイプシロン-グリーディで選ぶ。残りは従来通りの
# 完全ランダム生成のままにし、両者を同じ基準判定(evaluate_candidate)で公平に競わせて
# 最高スコアのものを選ぶ設計にした(=コンボだから優遇する、という特別扱いはしない)。
# プールが空の場合は自動的に従来の完全ランダム生成のみになる(select_bandit_combo参照)。
BANDIT_CANDIDATE_SHARE = 0.3  # 1バッチ(CANDIDATE_COUNT件)のうち、コンボプールから選ぶ割合


def select_candidate(rng: random.Random):
    """CANDIDATE_COUNT件を軽量シミュレーションし、基準判定(20〜40秒の尺フィルター含む)を
    通ったもののうち、今日すでに使われたruleを避けつつ(全て使い切っていれば直近投稿と同じruleを
    避けつつ)最高スコアの候補を選ぶ。
    1バッチも合格しなければMAX_CANDIDATE_BATCHES回まで再サンプリングし、それでも0件の場合のみ
    無投稿を避けるため尺条件を無視して妥協する(その場合もscore.passed/reject_reasonに
    正直に記録されるため、video_log/週次KPTで後から検知できる)。"""
    last_rule = _last_used_rule()
    rules_used_today = _rules_used_today()
    scored = []
    passed: list = []
    for batch in range(MAX_CANDIDATE_BATCHES):
        for _ in range(CANDIDATE_COUNT):
            combo = select_bandit_combo(rng) if rng.random() < BANDIT_CANDIDATE_SHARE else None
            if combo is not None:
                n_circles = rng.randint(4, N_CIRCLES_MAX)
                params = combo_to_candidate_params(combo, n_circles=n_circles, seed=rng.randint(1, 10_000_000))
            else:
                params = _random_candidate_params(rng)
            sim_kwargs = {
                k: v for k, v in params.items()
                if k not in ("camera", "slow_motion") and not k.startswith("_")
            }
            result, circles = simulate(**sim_kwargs)
            score = evaluate_candidate(result)
            scored.append((params, result, circles, score))
        passed = [s for s in scored if s[3].passed]
        if passed:
            break
        print(
            f"[battle daily] {batch + 1}/{MAX_CANDIDATE_BATCHES}バッチ(計{len(scored)}件)で、"
            "基準判定(20〜40秒の尺フィルター含む)に合格した候補がありませんでした。",
            file=sys.stderr,
        )

    if not passed:
        print(
            "[battle daily] 再サンプリングしても合格候補が見つかりませんでした。"
            "無投稿を避けるため尺条件を無視して最高スコアの候補で妥協します(要確認)。",
            file=sys.stderr,
        )
    pool = passed or scored

    # 2026-09-09、1日3本投稿化に伴い、まず「今日すでに使われたrule」を除外する
    # (RULESが5種類あるため、1日3本なら常に回避可能)。今日分の全ruleを使い切った場合
    # (4本目以降)は、従来通り「直近1本と違うrule」を避けるロジックにフォールバックする。
    diverse_pool = [s for s in pool if s[0]["rule"] not in rules_used_today]
    if not diverse_pool:
        diverse_pool = [s for s in pool if s[0]["rule"] != last_rule] or pool
    # 2026-09-08、player_shapeの採用実績が偏っている形状を選ばれやすくする補正
    # (_player_shape_selection_bonus参照)。スコアそのもの(video_logに記録される値)は
    # 変更せず、この選定時の比較にだけ掛け合わせる。
    return max(diverse_pool, key=lambda s: s[3].overall * _player_shape_selection_bonus(s[0]["player_shape"]))


def _random_visual_mismatch(rng: random.Random, n_circles: int) -> dict[int, float] | None:
    """8-5対応。1体だけ描画上の見た目サイズを変える(物理演算には一切影響しない)。"""
    if rng.random() >= VISUAL_MISMATCH_PROBABILITY:
        return None
    entity_id = rng.randrange(n_circles)
    return {entity_id: rng.choice(VISUAL_MISMATCH_SCALES)}


def build_metadata(params: dict) -> tuple[str, str, list[str]]:
    """2026-09-21、match_type新軸対応: rule×match_typeの組み合わせ分だけTITLE_TEMPLATES等の
    テーブルを増やすと管理コストが跳ね上がる(weapon_colosseum追加時の教訓)ため、既存の
    rule別テンプレートに対戦形式の一言を付け足す方式にした(表を増やさない最小限の対応)。"""
    rule = params["rule"]
    n = params["n_circles"]
    match_type = params.get("match_type", "individual")
    base_title = TITLE_TEMPLATES[rule].format(n=n)
    rule_label = RULE_LABELS[rule]
    tags = ["shorts", "physics simulation", "satisfying video", "simulation battle"] + TAGS_BY_RULE[rule]

    if match_type == "team":
        team_count = params.get("team_count") or 2
        title = f"{team_count} Teams Enter! {base_title}"
        rule_label = f"{rule_label} Team battle: no friendly fire, last team standing wins."
        tags = tags + ["team battle"]
    elif match_type == "boss":
        title = f"1 Boss vs {max(n - 1, 1)} Shapes! {base_title}"
        rule_label = f"{rule_label} One boss, bigger and stronger, against the rest."
        tags = tags + ["boss battle"]
    else:
        title = base_title

    description = DESCRIPTION_TEMPLATE.format(rule_label=rule_label)
    return title, description, tags


def main():
    rng = random.Random()
    params, result, circles, score = select_candidate(rng)

    # 実際に採用された候補がコンボプール由来だった場合のみ、そのコンボの実績
    # (uses/total_score)に反映する(採用されなかった候補は「実際に使われた」わけでは
    # ないため反映しない)。
    combo_id = params.get("_combo_id")
    if combo_id:
        record_combo_result(combo_id, score.overall)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{params['rule']}_seed{params['seed']}.mp4"
    visual_mismatch = _random_visual_mismatch(rng, params["n_circles"])
    render(
        result,
        circles,
        video_path,
        camera=params["camera"],
        slow_motion=params["slow_motion"],
        visual_mismatch=visual_mismatch,
    )

    winner_name = CHARACTER_NAMES.get(result.winner_id % len(COLORS)) if result.winner_id is not None else None
    title, description, tags = build_metadata(params)

    # アップロード成功が確定するまでは戦績・動画ログを更新しない
    # （アップロード失敗時に「投稿されていない動画」の戦績だけが記録される不整合を防ぐため）。
    video_id = upload_video(
        video_path,
        title,
        description,
        tags,
        category_id=CATEGORY_ID,
        privacy_status=PRIVACY_STATUS,
        client_secret_path=CLIENT_SECRET_PATH,
        token_path=TOKEN_PATH,
    )

    # 実際に投稿できた枠として消化する(手動実行分も含めてカウントし、自動枠の重複投稿を防ぐ。
    # geometry_posting_schedule.should_publish_now()参照)。
    mark_posted()
    record_result(circles, result.winner_id)
    log_video(
        video_id=video_id,
        published_at=datetime.now(timezone.utc).astimezone().isoformat(),
        privacy_status=PRIVACY_STATUS,
        rule=params["rule"],
        shape=params["shape"],
        player_shape=params["player_shape"],
        terrain=params["terrain"],
        match_type=params["match_type"],
        seed=params["seed"],
        score_breakdown=score.to_dict(),
        winner_character=winner_name,
    )

    try:
        video_path.unlink()
    except OSError:
        pass

    print(f"公開しました: https://youtu.be/{video_id} (rule={params['rule']}, shape={params['shape']}, score={score.overall})")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "geometry_battle_daily_pipeline.py")
