"""
幾何学アニメーション物理演算バトルチャンネル(4ch目) 技術検証プロトタイプ

図形(円、以降「プレイヤー」と呼ぶ)を枠の中に配置し、物理演算で衝突・脱落させて
最後の1体を決める動画を生成する。「ルール」(決着条件)と「ステージ形状」を軸として
切り替えられるようにしてある:

- rule="hole_fall"     : 床の隙間から落ちたら脱落、最後の1体が生存(デフォルト)
- rule="goal_reach"     : 密閉された枠の中でゴール(円形エリア)に最初に触れた者が勝者
- rule="area_control"   : 密閉された枠の中心にある安全地帯が時間経過で縮小し、
                          はみ出たら脱落。最後の1体が生存
- rule="absorb_growth"  : 密閉された枠の中でプレイヤー同士が接触すると大きい方が
                          小さい方を吸収し、吸収するたびに大きくなる(agar.io的マージ)。
                          最後の1体が生存
- shape="square" | "circle" : 枠の形状

アリーナはYouTube Shorts(9:16)の画角そのものに合わせた縦長の矩形/楕円で、
描画は最初からこのアスペクト比で行うため、上下に不要な余白ができない。
画面上部には半透明のHUDバーでルールの英語キャプションと残数を表示する。

一部のプレイヤー(色で決まる)は一定間隔で特殊能力を発動する:
- 黄色: 最も近い相手に向かってダッシュする
- 紫: 有利になりそうな位置に一定時間だけ「毒のワイヤー」(障害物)を設置する

pymunkで1回だけシミュレーションを実行し、毎フレームの位置・半径・枠の角度・
衝突/脱落/特殊能力イベントをログとして記録する(この段階では描画しないため軽量)。
基準判定を通過した場合のみ、同じログを使ってPillow(映像)+numpy合成音(音声)で
レンダリングし、moviepyで動画化する。同一ログから軽量集計と描画の両方を行うため、
シミュレーションを2回走らせて結果がズレる(環境間の浮動小数点差異)リスクを避けている。

投稿パイプラインには接続しない、技術検証用のスクリプト。
衝突音・脱落音・特殊能力音・勝利ジングルはパイプライン疎通確認用のプレースホルダー
合成音であり、本番のブランディング用SE(6章)ではない。

制約事項: 枠の回転ギミック(rotation_speed)は、アリーナが正方形だった旧バージョンでは
回転時の角の張り出し分の描画余白を確保していたが、今回アリーナを画面いっぱいの縦長矩形に
変更したため余白を持たせられなくなった。回転を使うと枠の角が画面端で見切れる可能性がある
(物理演算自体は問題なく動作する、描画上の既知の制約)。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymunk
from moviepy import AudioArrayClip, VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from src.geometry_battle_characters import type_matchup_multiplier
except ImportError:  # `python src/geometry_battle_gen.py`のように直接実行された場合
    from geometry_battle_characters import type_matchup_multiplier

# YouTube Shorts出力仕様(9:16, 1080x1920, 60fps)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
FPS = 60
DT = 1.0 / FPS

# 描画キャンバスはYouTube Shorts画角(9:16)のまま(幅800を基準単位とする)。
ARENA_W = 800.0
ARENA_H = ARENA_W * OUTPUT_HEIGHT / OUTPUT_WIDTH  # = 1422.22
ARENA_CX = ARENA_W / 2
ARENA_CY = ARENA_H / 2
RENDER_SCALE = OUTPUT_WIDTH / ARENA_W  # 最終書き出し時の拡大率

# 2026-09-09、ユーザー要望: 枠(壁)はキャンバスいっぱいに引き伸ばした楕円/矩形である必要はなく、
# 真円/正方形でよい(そのほうが回転にも対応しやすい)。半径/半辺長を両形状で共有し、
# 正方形の対角線がちょうどARENA_Wに収まるサイズにすることで、どちらの形状も全方向の回転で
# キャンバス外にはみ出さない(円は回転しても輪郭が変わらないため無条件に安全)。
# これにより_rotation_zoom_scaleによる縮小補正が不要になった(常に1.0を返すよう変更済み)。
ARENA_HALF_EXTENT = ARENA_W / (2 * math.sqrt(2))  # ≈282.8
ARENA_TOP = ARENA_CY - ARENA_HALF_EXTENT
ARENA_BOTTOM = ARENA_CY + ARENA_HALF_EXTENT
ARENA_LEFT = ARENA_CX - ARENA_HALF_EXTENT
ARENA_RIGHT = ARENA_CX + ARENA_HALF_EXTENT
# 旧枠(ARENA_CX=400を基準に調整されていた各種サイズ)を新しい半径に合わせて比例縮小するための係数
_ARENA_SCALE = ARENA_HALF_EXTENT / ARENA_CX  # = 1/sqrt(2) ≈ 0.707

# 2026-09-08、ユーザー要望「ステージのバリエーションを増やしてほしい、広いステージも歓迎」
# 「画面をもっと使いたい、脱落したら画面外に出るイメージ」への対応。
# 縦横で別の半径/半辺長(half_x, half_y)を持てるようにし、「compact/spacious」(縦横比は
# 正方形/真円のまま一律スケール)に加えて「tall」(縦に伸ばして画面の縦方向をより使う)を追加した。
# 縦横比を変える形状は、正方形の回転安全性(対角線がARENA_Wにちょうど収まる設計)は当然、
# 真円であっても非等方(楕円)だと回転で見切れるため、simulate()側で自動的に回転を止める。
ARENA_SIZE_VARIANTS = {
    "compact": (1.0, 1.0),
    "spacious": (1.24, 1.24),
    # 縦(half_y)だけ大きくする。HUDバー(画面上部)と衝突しない範囲、画面下端に余裕を
    # 持たせた範囲で、実測しながら決めた値(2026-09-08)
    "tall": (1.0, 1.95),
}


def _arena_layout(half_x: float, half_y: float) -> dict:
    """half_x(横方向)・half_y(縦方向)を基準に、枠の境界やゴール・加速ゾーン・
    トラップ・安全地帯サイズなど、枠のサイズに比例して決まるあらゆる位置をまとめて返す。
    ステージサイズをsimulate()の呼び出しごとに変えられるようにするため、以前モジュール定数
    だった計算式(GOAL_POINT等)をここに集約した(2026-09-08、tall対応でhalf_x/half_yの
    2軸に一般化)。compact/spaciousのようにhalf_x==half_yの等方な場合は、以前の
    half_extent基準の値と完全に一致する。"""
    top = ARENA_CY - half_y
    bottom = ARENA_CY + half_y
    left = ARENA_CX - half_x
    right = ARENA_CX + half_x
    return {
        "half_x": half_x,
        "half_y": half_y,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "goal_point": (ARENA_CX, top + half_y * 0.3),
        "accel_zone_rect": (
            ARENA_CX - half_x * 0.4,
            bottom - half_y * 0.55,
            ARENA_CX + half_x * 0.4,
            bottom - half_y * 0.08,
        ),
        "trap_center": (ARENA_CX - half_x * 0.55, ARENA_CY + half_y * 0.15),
        # 安全地帯は円形なので、非等方な場合でも縦横どちらの端も内包できるよう大きい方基準にする
        "zone_start_radius": max(half_x, half_y) * 2.2,
        "camera_min_width": half_x * 2 * 0.42,
    }


_DEFAULT_ARENA_LAYOUT = _arena_layout(ARENA_HALF_EXTENT, ARENA_HALF_EXTENT)  # compact(既定サイズ)の値。既存の定数群と一致する

WALL_THICKNESS = 10  # 物理演算上の壁の厚み(当たり判定)。既存チューニングに影響するため変更しない
# 見た目上の壁の太さ。物理演算とは切り離した描画専用の値にすることで、視認性改善のために
# 太くしてもシミュレーションのチューニング(反発・衝突タイミング等)には一切影響しない
WALL_VISUAL_THICKNESS = 20

# 2026-09-09、ユーザー指摘: 枠(壁)が描画キャンバスの端ギリギリに全面表示されるため、
# YouTube Shortsの下部UI(再生バー・キャプション欄)や右側の操作ボタン列と重なって見えにくい。
# 対策として、最終書き出し時に映像全体をわずかに縮小し、アリーナと同じ背景色の余白で
# 画面全体を囲む(黒帯ではなく背景色の余白なので、フルブリードの見た目はほぼ損なわれない)。
SAFE_ZONE_SCALE = 0.90
MAX_SIM_SECONDS = 90
CIRCLE_RADIUS = round(35 * _ARENA_SCALE)  # 旧35→新しい枠サイズに比例縮小(≈25)
# 「画面外に出た」とみなす余白。穴を完全に通過したことを確認できる程度で十分であり、
# 大きすぎるとプレイヤーが脱落判定される前に描画キャンバス外に出て見切れてしまう
# (壁のすぐ内側で静止している分には壁がブロックするのでこの値を大きくする必要はない)
ESCAPE_MARGIN = 40.0 * _ARENA_SCALE

# 2026-09-12、序盤密度向上(離脱曲線対策): スポーン直後に中心方向への初期インパルスを
# 与え、開始数秒の停滞(単なる落下・漂流)を排除する。値は「無風状態なら中心までこの秒数で
# 到達する」速度になるよう距離から逆算するため、アリーナサイズ(spacious/tall/hourglass等)が
# 変わっても自動的にスケールする。
INITIAL_IMPULSE_TARGET_SECONDS = 0.6

# 2026-09-16、冒頭フック強化(ユーザー指示): 「開始0.1秒で必ず最初の衝突を起こす」ため、
# 中心方向インパルスに加えて各プレイヤーへ最も近い他プレイヤー方向への追加インパルスを
# 与える。値は「無風状態ならこの秒数で最近接プレイヤーに到達する」速度になるよう距離から
# 逆算するが、双方が同時に相手へ向かうため実際の衝突はこの値よりかなり早く起こる。
# 0.1秒ジャストの保証は距離依存のためできないが、現実的に「開始直後、最速で衝突する」
# ことを狙った近似値。
# 2026-09-16追記: 導入直後の値(0.3)はFIRST_IMPACT_MAX_SECONDS(1.5秒)の条件は満たすものの、
# 序盤の衝突が強すぎて全体の決着ペースまで早め、本番の尺フィルター(15〜25秒、
# geometry_battle_scoring.PRODUCTION_DECISION_SECONDS_RANGE)を通過する候補が短尺側に
# 偏る副作用が実測で確認された(ユーザー指摘: 「動画の尺自体が短い」)。daily_pipelineの
# 候補生成ロジックで0.3/0.5/0.6/0.8/1.0/1.2/1.5/2.0を比較したところ、0.8が最も尺フィルター
# 合格率が高く中央値も改善する一方、最初の衝突が1.5秒を超えるケースは0.3〜2.0のどの値でも
# 発生しなかった(中心方向インパルスと合わせて元々十分早いため)ことから0.8を採用する。
OPENING_COLLISION_TARGET_SECONDS = 0.8

# 反発係数・重力・穴幅は「物理パラメータ/ステージギミック」軸として可変にする。
# デフォルトは、縦長アリーナで尺40秒前後・衝突多めを狙って再チューニングした値
# (2026-09-07、縦長化+60fps化に伴い旧バージョンの値から再調整。2026-09-09、真円/正方形化に
# 伴う枠サイズ縮小(_ARENA_SCALE)に合わせてgravity/hole_widthも比例縮小し、再実測で微調整した)。
DEFAULT_GRAVITY = 620.0 * _ARENA_SCALE
DEFAULT_ELASTICITY = 0.992
DEFAULT_DAMPING = 0.9998  # 空気抵抗相当。1.0だと理論上減衰せず終わらないため微量だけ減衰させる
DEFAULT_HOLE_WIDTH = round(130 * _ARENA_SCALE)
# 摩擦係数。低いほど床/壁との接触で水平方向の運動量を失いにくく、跳ね回り続けて
# 「停滞」しにくくなる(2026-09-07、ユーザー指摘によりデフォルトを大幅に下げた)
DEFAULT_FRICTION = 0.05
DEFAULT_ROTATION_SPEED = 0.0
DEFAULT_SHAPE = "square"
DEFAULT_RULE = "hole_fall"
# プレイヤー本体の形状(枠の形状=DEFAULT_SHAPEとは別軸)。"circle"|"square"|"triangle"
# 2026-09-08、ユーザー方針: triangleは通常の物理演算(円と同じ重力・初期スピンのみ)のまま、
# 独自の挙動を提案するまでは実装しない(circleは通常物理、squareのみ「重力なし・常に直進・
# 衝突等の外力では回転しうる」専用挙動を実装済み)。
DEFAULT_PLAYER_SHAPE = "circle"

# goal_reachルール用: 重力に逆らって到達する必要がある上方のゴールエリア
# (2026-09-09、枠の真円/正方形化に伴いARENA_HALF_EXTENT基準の位置に再定義。
# 2026-09-08、ステージサイズのバリエーション対応により_arena_layout()経由の値に変更)
GOAL_POINT = _DEFAULT_ARENA_LAYOUT["goal_point"]
GOAL_RADIUS = round(100 * _ARENA_SCALE)

# ステージギミック: 加速ゾーン。範囲内にいるプレイヤーに毎フレーム一定の力を加える。
# goal_reachは重力に逆らってゴールへ届く必要があり単体では難易度調整が困難だったため、
# 下部に上向きの「発射台」ゾーンを設置して補助する(rule=="goal_reach"では自動的に有効化)
ACCEL_ZONE_RECT = _DEFAULT_ARENA_LAYOUT["accel_zone_rect"]
# 2026-09-09: 単純な比例縮小(-2600*_ARENA_SCALE≈-1839)だと枠が小さくなった分ゴールへの
# 到達が速すぎたため(実測)、実際にシミュレーションを回して30〜60秒近辺に収まる値を探索し直した。
# goal_reachは既存チューニングでも「即決着」と「未到達」の二極化が起きやすい不安定なルールで
# あることが分かっており、この値でも変動は大きいまま(既知の課題、完全解決はしていない)。
# 2026-09-09、尺ターゲット20〜40秒→15〜25秒への短縮に伴い、実際にシミュレーションを回して
# 再チューニング(-600→-450。40シード中15秒〜25秒に収まる割合が最大になる値を探索した。
# 依然として「即決着」と「未到達」の二極化が起きやすい不安定なルールである点は変わらない)。
ACCEL_ZONE_FORCE = (0.0, -450.0)  # y-が上方向

# ステージギミック: 内部トラップ。出口とは別に、枠の中に触れたら即脱落する小さな穴を置く。
# 中心の主要な落下経路を塞がないよう、少し横にずらして「避けられる」配置にする
TRAP_CENTER = _DEFAULT_ARENA_LAYOUT["trap_center"]
TRAP_RADIUS = 38.0 * _ARENA_SCALE

# area_controlルール用: 中心の安全地帯が時間経過で線形に縮小する。
# 開始半径はスポーン範囲を確実に内包する値にする(下のspawn_circlesの範囲を参照)。
# 2026-09-07: 尺40秒前後を狙って再チューニング(elasticity=1.0,damping=1.0前提)。
# 2026-09-09: 枠の真円/正方形化に伴いARENA_HALF_EXTENT基準に再定義・再実測調整。
ZONE_START_RADIUS = _DEFAULT_ARENA_LAYOUT["zone_start_radius"]
ZONE_END_RADIUS = 60.0
# 2026-09-09、尺ターゲット20〜40秒→15〜25秒への短縮に伴い75→30へ再チューニング
# (実際にシミュレーションを回して確認: 40シード中40件が15〜25秒に収まることを確認済み)。
ZONE_SHRINK_SECONDS = 30.0

# 新ステージ内部構造(Terrain、2026-09-09、ユーザー要望)。外枠の形状(真円/正方形)や
# _arena_layoutの基礎ロジックは変更せず、既存の枠の内部にpymunk.Segment/pymunk.Circleによる
# 障害物を追加配置する形で実装する。全て中心座標からの相対位置(ローカル座標)で定義し、
# 壁と同じarena_body(回転するkinematic body)に追加することで、全方向回転のギミックが
# 作動しても外枠と一緒に破綻なく回転する(壁の実装と全く同じ仕組みを流用するだけで済む)。
TERRAIN_TYPES = ["hourglass", "donut", "cross", "pegboard", "pinball", "two_tier", "split_horizontal"]
# 2026-09-09、実測で発見: ネック(くびれ)付近で複数プレイヤーが密集して衝突すると、
# 強い衝撃で1フレーム(1/60秒)のうちに薄い壁(WALL_THICKNESS=10)をすり抜けて枠外に
# 出てしまう(トンネリング)ことがまれにあった(n_circles=7時に確認)。2.6→3.2に広げることで
# 発生率を大幅に低減(実測でほぼ解消、残りはdaily_pipeline.py側でhourglassのn_circlesを
# 6以下に制限することで完全に解消したことを確認済み)。
CHOKE_WIDTH_RATIO = 3.2  # hourglass: 中央通路の幅(プレイヤー半径の倍数)
# 2026-09-09(ユーザー要望で再設計、2回目): 「内部障害物+四角い外枠」ではなく、砂時計の
# シルエットそのものが外枠になる仕様に変更(通常の四角/円の壁を廃止)。それに伴いサイズも
# half_x一杯まで拡大した(以前は外枠と別形状に見せるためHOURGLASS_WIDTH_RATIO=0.5に
# 絞っていたが、外枠自体を廃止したので絞る理由がなくなった。1.0にすると正方形の角と
# 同じ位置になり、既存の回転安全性判定(shapeが非circleの場合half_x基準)をそのまま
# 使い回せる)。hourglassだけ他のterrain(内部障害物として追加)とは扱いが異なるため、
# 専用の_hourglass_boundary_chain/_point_in_hourglassで別管理する
# (build_space/spawn_circles/_iter_rendered_framesそれぞれで分岐)。
HOURGLASS_WIDTH_RATIO = 1.0  # half_xに対する、砂時計の最も幅広い部分(上端・下端)の比率
# 2026-09-09、ユーザー要望「縦方向にもっと大きくしていい」への対応。hourglass専用の実効half_y
# (_hourglass_boundary_chain/_point_in_hourglass/脱落判定のescape_half_yすべてで共通して使う)。
# 1.8倍まで試したが、area_controlの安全地帯(ZONE_SHRINK_SECONDS=30、hourglass以外の全terrainと
# 共有するグローバル定数で地形別の調整はしていない)が広がった分の移動距離に追いつけず、
# 尺ターゲット(15-25秒)から外れがちだった(median13.2秒)。1.4倍だと明確に大きくなりつつ
# 尺ターゲットとの両立も確認できたため(30シード中18件が15-25秒に収まる)、この値を採用した。
HOURGLASS_HEIGHT_RATIO = 1.4
DONUT_INNER_RADIUS_RATIO = 0.38  # donut: 中央障害物の半径(min(half_x,half_y)に対する比率)
CROSS_OPENING_RATIO = 2.8  # cross: 中央開口部の半径(プレイヤー半径の倍数)

# two_tier: 上段に浮かせる3つの足場(左/中央/右)の配置パラメータ。
TWO_TIER_PLATFORM_Y_FRAC = -0.45  # half_yに対する比率(負=上寄り)。山型の「低い端」の高さ
# 2026-09-20、ユーザー指示: 「上部の足場は三つの長方形でなく、間に滑らせる前提で三角形に」。
# 平らな板(2点の水平線分)から山型(への字、3点の折れ線)へ変更。頂点でスポーンし、
# 重力で左右どちらかの斜面を滑り落ちて低い端(枝分かれの先、武器の位置)へ到達する
# 動線を作る狙い。頂点の高さは低い端よりhalf_len*係数だけ高くする(緩すぎず滑り落ちる
# 程度の傾斜になるよう実測で調整)。
TWO_TIER_PLATFORM_PEAK_DROP_FRAC = 0.85
# 隣接する足場の端同士の隙間。旧実装(長方形+GAP_FRAC/SPACING_FRAC方式)では隙間が
# 約10(プレイヤー直径50の1/5)しかなく、山型化で頂点から滑り落ちたプレイヤーが隙間を
# 素通りせず隣の足場の斜面に乗り移って足場群の中で跳ね回り続ける不具合があった。
# プレイヤー直径の4倍を確保することで解消。HALF_LEN_FRACはこのGAPを確保しつつ3つの
# 足場が外壁からはみ出さないよう0.22→0.16に縮小した(3*half_len+gap <= half_xが目安)。
TWO_TIER_PLATFORM_HALF_LEN_FRAC = 0.16  # half_xに対する比率(足場の片側の長さ)
TWO_TIER_PLATFORM_GAP = CIRCLE_RADIUS * 4.0


def _two_tier_platforms(half_x: float, half_y: float) -> list[tuple[float, float, float, float]]:
    """two_tier用、3つの山型足場の(中心x, 低い端のy, 半長, 頂点のy)を返す(ローカル座標)。
    _terrain_obstacles(物理・描画)とspawn_circles/武器配置の全てがこれを共有することで、
    見た目・当たり判定・スポーン位置が常に一致するようにする。

    2026-09-20、三角形化の実測でundecidedが6/40→17/40に悪化した原因が判明したため
    TWO_TIER_PLATFORM_GAPを新設: 旧GAP_FRAC/SPACING_FRACの組み合わせだと隣接する足場の
    端同士の隙間が約10(プレイヤー直径50の1/5)しかなく、頂点から滑り落ちたプレイヤーが
    隙間を素通りせず隣の足場の斜面に乗り移ってしまい、足場群の中で跳ね回り続けていた。
    プレイヤー直径の4倍を隙間として確保(+それに合わせてHALF_LEN_FRACを縮小し外壁との
    はみ出しも修正)することで9/40まで改善した(素のgun_duelの既知の未到達傾向より
    やや高いが、山型化で得られる「枝分かれ」の見た目を優先してこの値を採用)。"""
    edge_y = half_y * TWO_TIER_PLATFORM_Y_FRAC
    half_len = half_x * TWO_TIER_PLATFORM_HALF_LEN_FRAC
    peak_y = edge_y - half_len * TWO_TIER_PLATFORM_PEAK_DROP_FRAC
    spacing = 2 * half_len + TWO_TIER_PLATFORM_GAP
    return [
        (-spacing, edge_y, half_len, peak_y),
        (0.0, edge_y, half_len, peak_y),
        (spacing, edge_y, half_len, peak_y),
    ]


# split_horizontal: 上下の陣地を隔てる仕切りの位置(half_yに対する比率、中心からの距離)。
SPLIT_HORIZONTAL_DIVIDER_FRAC = 0.30
SPLIT_HORIZONTAL_GUN_RESPAWN_DELAY_SECONDS = 0.2  # 通常(0.5秒)より頻度を上げる(ユーザー指示)
PEG_RADIUS = 14.0 * _ARENA_SCALE
# 2026-09-09、実測により再調整: 当初spacing=95だとプレイヤー直径(≈50)とほぼ同じ隙間しか空かず、
# hole_fallで20シード中18〜20件が90秒経っても決着しない(ピンの上で詰まり続ける)不具合が
# あった。spacing=160に広げ、tall(縦長)アリーナ+gravity3.0x(daily_pipeline.py側で上書き)と
# 組み合わせることで20シード中0件まで解消したことを確認済み。
PEG_SPACING_X = 160.0 * _ARENA_SCALE
PEG_SPACING_Y = 160.0 * _ARENA_SCALE
# pegboardは下部(hole_fallの出口付近)を空けておく(完全に塞ぐと出口に到達できなくなるため)
PEG_BOTTOM_MARGIN_RATIO = 0.6

# 2026-09-13、新terrain: pinball。「本格ピンボール型ステージ」指示書に基づき全面改修
# (当初の3連バンパーのみの試作から、バンパー+スリングショット+誘導スロープの構成へ拡張)。
# tallステージ(縦長)をベースに、上部の3連ポップバンパー(逆三角形配置)・下部左右の
# スリングショット(斜面で中央上方へ弾く)・最下部の誘導スロープ(穴へ滑り込ませる)を
# 組み合わせ、「真下に落ちるだけの単調な決着」を防ぐ。
#
# ポップバンパー(逆三角形: 上に2つ・下に1つ)。指示書「Y座標60〜75%付近」は、下から数えた
# 高さの割合と解釈(=上端からは25〜40%の位置)。指示書の「プレイヤー半径の0.8〜1.2倍」に
# 合わせて半径を設定。
PINBALL_BUMPER_RADIUS = CIRCLE_RADIUS * 1.15
PINBALL_BUMPER_TOP_Y_RATIO = -0.5  # 上2つ(逆三角形の上辺)
PINBALL_BUMPER_BOTTOM_Y_RATIO = -0.28  # 下1つ(逆三角形の頂点)
PINBALL_BUMPER_X_RATIO = 0.32
# 指示書指定の反発係数1.3〜1.5の下寄り。2026-09-09に判明した「反発係数>1.0だと衝突の
# 度に運動エネルギーが増え続け壁をすり抜ける」不具合(776行目付近参照)と同じ罠を避けるため、
# pin_wallと同様に毎フレームの速度クランプ(PINBALL_MAX_SPEED)を安全弁として併用する。
PINBALL_BUMPER_RESTITUTION = 1.35
PINBALL_MAX_SPEED = 950.0 * _ARENA_SCALE
PINBALL_COLOR = (255, 210, 60)  # アーケードのバンパーらしい暖色(他terrainの円形障害物と区別)
PINBALL_FLASH_SCALE = 1.2  # ヒット時の一瞬の拡縮倍率(指示書指定)
PINBALL_FLASH_FRAMES = 8  # 拡縮が持続するフレーム数

# 2026-09-13、ユーザー指摘対応「上の領域を活かす工夫」: スポーン帯(最上段)とメインの
# 3連バンパーの間が、ただ落下するだけの空白区間になっていた。同じ仕組み(circles、
# collision_type=8)を使い回せる小型バンパー2個を間に追加し、メインクラスターへ到達する前に
# 一度弾かれる「2段カスケード」にすることで、tallステージの縦方向をより使い切る。
PINBALL_UPPER_BUMPER_RADIUS = PINBALL_BUMPER_RADIUS * 0.7
PINBALL_UPPER_BUMPER_Y_RATIO = -0.72  # メインクラスター(-0.5)よりさらに上
PINBALL_UPPER_BUMPER_X_RATIO = 0.16  # メインクラスターより中央寄り(左右で挟むだけの軽い誘導)
# スポーン安全マージンは実際に最も上にあるバンパー(この上段バンパー)基準で計算する。

# 2026-09-13、ユーザー指摘対応「バンパーヒットが前半(=上部)にしか起こらず、勝負のほとんどが
# 画面下部で行われている」: 上2段のバンパー群(-0.72〜-0.28)より下、スリングショット
# (+0.62)より上の範囲がまるごと空白(ただ自由落下するだけ)になっていたのが原因。
# 中段にもう1組バンパーを追加し、落下の全行程でバンパーとの接触が起こるようにする
# (メインクラスターと左右をずらし、同じ軌道の繰り返しにならないようにする)。
PINBALL_MID_BUMPER_RADIUS = PINBALL_BUMPER_RADIUS * 0.85
PINBALL_MID_BUMPER_Y_RATIO = 0.1  # 中央よりやや下、スリングショット(+0.62)よりはっきり上
PINBALL_MID_BUMPER_X_RATIO = 0.4  # メインクラスター(0.32)より外側にずらす

# 2026-09-13、ユーザー再指摘: 「バンパーの数を上げるのではなく、上部領域での接触期間を
# 意識する方法で」対応してほしいとのこと。フリッパーで押し戻す方向の低段バンパー(前回追加分)
# は削除し(下記参照)、代わりに上部バンパー帯(上段〜メインクラスター)の中では重力を
# 部分的に打ち消し、自然落下より滞空時間を延ばすことでバンパーとの接触機会を増やす
# アプローチに切り替える。space.gravityはspace全体に一様にかかるため、このゾーン内にいる
# プレイヤーへ毎フレーム「重力の一部を打ち消す」補正を加える形で局所的に実現する
# (simulate()メインループのpinball専用ブロック参照)。
PINBALL_FLOAT_ZONE_TOP_Y_RATIO = PINBALL_UPPER_BUMPER_Y_RATIO - 0.08  # 上段バンパーより少し上から
PINBALL_FLOAT_ZONE_BOTTOM_Y_RATIO = PINBALL_MID_BUMPER_Y_RATIO - 0.05  # 中段バンパーの少し上まで
PINBALL_FLOAT_GRAVITY_SCALE = 0.4  # このゾーン内では通常重力の40%分だけ効かせる

# スリングショット(下部左右、斜面で中央上方へ弾く)。指示書のSLINGSHOT_ANGLE(35〜45度)は
# 中央寄りの値を採用。反発係数もバンパーと同じ理由でPINBALL_MAX_SPEEDの安全弁下で運用する。
# 2026-09-13、ユーザー指摘によりさらに大きく(90→170、約1.9倍)。半径282.8のアリーナで
# tip_xが中心線を越えない(左右が交差しない)ことを確認済み。
SLINGSHOT_ANGLE_DEG = 40.0
SLINGSHOT_LENGTH = 170.0 * _ARENA_SCALE
SLINGSHOT_BASE_Y_RATIO = 0.62  # 下部、誘導スロープより少し上
SLINGSHOT_RESTITUTION = 1.3
SLINGSHOT_WALL_MARGIN = 30.0 * _ARENA_SCALE  # 側壁からの内側オフセット

# 誘導スロープ(最下部、左右の壁から中央の穴へ向けて緩やかに傾斜し、横方向の停滞を防ぐ)。
# 実際のhole_widthに関わらず、中央付近の一定幅を狙う簡略化(スポーンはさらに上部に
# 制限しているため、穴の実寸との厳密な整合は必須ではない)。
DRAIN_ANGLE_DEG = 25.0
DRAIN_LENGTH = 140.0 * _ARENA_SCALE
DRAIN_TARGET_HALF_WIDTH = 50.0 * _ARENA_SCALE
DRAIN_WALL_MARGIN = 10.0 * _ARENA_SCALE

# 2026-09-13、指示書④「自動パルスフリッパー」対応。実際のフリッパー形状(回転する2本のバー)は
# 実装せず、指示書が挙げた代替案「周期的なキック力を持たせる」の方を採用する(プレイヤー入力の
# ない自動対戦という前提と相性がよく、当たり判定・描画とも既存terrainの円/セグメントの枠組みを
# 増やさずに済む)。穴の手前に薄い帯状のゾーンを置き、一定間隔でゾーン内の全プレイヤーへ
# まとめて上向きのキックを与えることで、「即落ちを救済して乱戦を長引かせる」狙いを実現する。
FLIPPER_ZONE_HALF_WIDTH = DRAIN_TARGET_HALF_WIDTH * 3.0
FLIPPER_ZONE_HEIGHT = 70.0 * _ARENA_SCALE  # 穴のすぐ手前の帯の高さ
FLIPPER_PULSE_INTERVAL_SECONDS = 0.5  # このリズムでフリッパー群が自動的に跳ね上げる
# バンパー/スリングショットと違い反発係数ではなく直接の速度上書きにしている(こちらは
# PINBALL_MAX_SPEEDの範囲内の値を最初から使うため、安全弁を別途二重に掛ける必要がない)。
# 2026-09-13、ユーザー指摘で低段バンパー(フリッパーで押し戻す方式)を撤回したため、
# フリッパーの狙いも「即落ちの救済」という元の役割に戻し、キック速度も元の値に戻した。
FLIPPER_KICK_SPEED = 700.0 * _ARENA_SCALE
FLIPPER_COLOR = (120, 220, 255)  # バンパー(暖色)と区別する寒色


def _pinball_bumper_positions(half_x: float, half_y: float) -> list:
    r = PINBALL_BUMPER_RADIUS
    ur = PINBALL_UPPER_BUMPER_RADIUS
    mr = PINBALL_MID_BUMPER_RADIUS
    return [
        # 上段の小型バンパー2個(スポーン帯とメインクラスターの間、2026-09-13追加)
        (-half_x * PINBALL_UPPER_BUMPER_X_RATIO, half_y * PINBALL_UPPER_BUMPER_Y_RATIO, ur),
        (half_x * PINBALL_UPPER_BUMPER_X_RATIO, half_y * PINBALL_UPPER_BUMPER_Y_RATIO, ur),
        # メインの3連バンパー(逆三角形)
        (-half_x * PINBALL_BUMPER_X_RATIO, half_y * PINBALL_BUMPER_TOP_Y_RATIO, r),
        (half_x * PINBALL_BUMPER_X_RATIO, half_y * PINBALL_BUMPER_TOP_Y_RATIO, r),
        (0.0, half_y * PINBALL_BUMPER_BOTTOM_Y_RATIO, r),
        # 中段バンパー2個(2026-09-13追加、メインクラスター〜スリングショットの空白を埋める)
        (-half_x * PINBALL_MID_BUMPER_X_RATIO, half_y * PINBALL_MID_BUMPER_Y_RATIO, mr),
        (half_x * PINBALL_MID_BUMPER_X_RATIO, half_y * PINBALL_MID_BUMPER_Y_RATIO, mr),
    ]


def _pinball_slingshot_chains(half_x: float, half_y: float) -> list:
    angle = math.radians(SLINGSHOT_ANGLE_DEG)
    dx = math.cos(angle) * SLINGSHOT_LENGTH
    dy = -math.sin(angle) * SLINGSHOT_LENGTH  # yは下向きが正のため、上向きは負
    base_y = half_y * SLINGSHOT_BASE_Y_RATIO
    left_base = (-half_x + SLINGSHOT_WALL_MARGIN, base_y)
    right_base = (half_x - SLINGSHOT_WALL_MARGIN, base_y)
    return [
        [left_base, (left_base[0] + dx, left_base[1] + dy)],
        [right_base, (right_base[0] - dx, right_base[1] + dy)],
    ]


def _pinball_drain_chains(half_x: float, half_y: float) -> list:
    angle = math.radians(DRAIN_ANGLE_DEG)
    dx = math.cos(angle) * DRAIN_LENGTH
    dy = math.sin(angle) * DRAIN_LENGTH
    return [
        [(-half_x + DRAIN_WALL_MARGIN, half_y - dy), (-DRAIN_TARGET_HALF_WIDTH, half_y)],
        [(half_x - DRAIN_WALL_MARGIN, half_y - dy), (DRAIN_TARGET_HALF_WIDTH, half_y)],
    ]

# 2026-09-13、ユーザー提供の設計指示書に基づく新terrain: pin_wall。ユーザー指定により
# 「内部地形として追加」ではなく「外枠限定」で採用する。hourglassが外枠そのものを砂時計形状に
# 置き換える特殊terrainであるのと同じ位置づけで、外枠(square/circleの壁)そのものを、隣り合う
# 円が重なり合う密度の小さなピンの列に置き換える(内部空間には別途何も敷き詰めない)。
# フラットな壁と違い、当たった位置によって跳ね返る角度が読めないため、外周に当たるだけでも
# 予測不能さが生まれる(指示書1章の「サプライズ指数」の狙いを外枠自体で実現する)。
# 壁の輪郭は_wall_local_segments(shape, hole_width, half_x, half_y)と完全に共有しているため、
# hole_fallの穴・spawn安全判定・脱落判定など、既存のshapeベースの判定はそのまま使い回せる
# (hourglassのように別途専用の境界判定関数を用意する必要がない)。
PIN_WALL_RADIUS = CIRCLE_RADIUS * 0.3  # 指示書「プレイヤー半径の0.25〜0.35倍」の中央値
PIN_WALL_OVERLAP_RATIO = 0.85  # ピンの中心間隔 = 直径 * この比率(1.0未満にして隙間なく重ねる)
PIN_WALL_RESTITUTION = 1.15  # 指示書の「1.1〜1.3程度」の下寄り
# 2026-09-09に判明した「反発係数を1.0超にすると衝突のたびに運動エネルギーが増え続け、
# 加速したプレイヤーが1フレームで壁をすり抜ける」不具合(776行目付近のコメント参照)と
# 同じ罠を踏まないよう、pin_wall選択時は毎フレームこの上限で速度をクランプする安全弁を設ける
# (simulate()のメインループ、space.step直後を参照)。指示書のPEG_RESTITUTIONを素の反発係数の
# まま採用しつつ、このクランプで無限にエネルギーが増え続ける事態だけを防ぐ。
PIN_WALL_MAX_SPEED = 900.0 * _ARENA_SCALE
PIN_WALL_COLOR = (225, 225, 235)  # ピンボール実機の金属ポストを意識した銀白色


def _pin_wall_positions(shape: str, hole_width: float, half_x: float, half_y: float) -> list:
    """pin_wall用: 壁の輪郭(_wall_local_segmentsと同じセグメント)に沿って、隣接する円が
    重なり合う間隔でピンの中心座標を並べる。hole_fallの脱出口はセグメント自体が
    そこだけ間引かれているため、追加の穴処理なしに自然と空く。"""
    spacing = PIN_WALL_RADIUS * 2 * PIN_WALL_OVERLAP_RATIO
    positions = []
    for (ax, ay), (bx, by) in _wall_local_segments(shape, hole_width, half_x, half_y):
        seg_len = math.hypot(bx - ax, by - ay)
        steps = max(1, round(seg_len / spacing))
        for i in range(steps + 1):
            t = i / steps
            positions.append((ax + (bx - ax) * t, ay + (by - ay) * t, PIN_WALL_RADIUS))
    return positions


def _terrain_obstacles(terrain: str | None, half_x: float, half_y: float) -> tuple[list, list]:
    """指定terrainの、中心(0,0)基準ローカル座標での障害物定義を返す。
    戻り値: (chains, circles)。chains=[[p0, p1, ...], ...](連続した折れ線の頂点列。
    描画時にjoint="curve"の1回のdraw.line()で結ぶことで、辺の継ぎ目のくぼみ・はみ出しを
    防ぐ。2点だけのchainは単一セグメントとして扱われる)、circles=[(cx, cy, radius)]
    """
    if terrain in ("hourglass", "pin_wall"):
        # 2026-09-09: hourglassは外枠そのものを置き換える専用扱いになったため、ここでは
        # 何も返さない(内部障害物としては存在しない)。実際の形状は_hourglass_boundary_chain
        # 参照。build_space/spawn_circles/_iter_rendered_framesはterrain=="hourglass"を
        # 個別に分岐して扱う。2026-09-13: pin_wallも同じ理由(外枠そのものの置き換え、
        # 内部空間には何も追加しない)でここに合流させた。実際のピン配置は
        # _pin_wall_positions参照、build_space/_draw_arena_wallsで個別に分岐する。
        return [], []
    if terrain == "donut":
        # 中心に固定の巨大な円形障害物を配置し、視線と直進経路を遮って回り込みの動きを誘発する
        r = min(half_x, half_y) * DONUT_INNER_RADIUS_RATIO
        return [], [(0.0, 0.0, r)]
    if terrain == "cross":
        # 上下左右の各辺の中央から中心に向かって壁を伸ばし、4つの部屋に区切る
        # (中心部は交差させず開けておく: 序盤は各部屋で1対1、終盤は中央で乱戦という起伏を作る)
        opening = CIRCLE_RADIUS * CROSS_OPENING_RATIO
        chains = [
            [(0.0, -half_y), (0.0, -opening)],
            [(0.0, half_y), (0.0, opening)],
            [(-half_x, 0.0), (-opening, 0.0)],
            [(half_x, 0.0), (opening, 0.0)],
        ]
        return chains, []
    if terrain == "two_tier":
        # 2026-09-20、ユーザー指示: 「2段式ステージ」。上段に3つの足場(枝分かれ)を浮かせ、
        # 各足場の先端付近に武器を置く(実際の配置は_two_tier_weapon_position、スポーンは
        # spawn_circles側で分岐)。足場には端があり、そこから落ちると自然に重力で下段(本戦の
        # 広いエリア)へ落下する─という仕組みを、追加の特別な「落下判定」なしで実現できる
        # (単に足場を宙に浮いた短い床として置くだけで、はみ出れば普通に落ちる)。
        # 2026-09-20追記、ユーザー指示: 平らな板(長方形)ではなく「間に滑らせる前提で三角形」に
        # 変更。3点の折れ線(低い端→頂点→低い端)にするだけで、_chains_to_segments/_draw_terrain
        # 側は既存の汎用ロジック(chainを連続セグメント化/1本の折れ線として描画)がそのまま
        # 山型として機能する(専用コード追加は不要)。
        platforms = _two_tier_platforms(half_x, half_y)
        return [
            [(cx - hl, edge_y), (cx, peak_y), (cx + hl, edge_y)] for cx, edge_y, hl, peak_y in platforms
        ], []
    if terrain == "split_horizontal":
        # 2026-09-20、ユーザー指示: 「上下分断ステージ」(gun_duel専用)。既存の外枠(左右・上下の
        # 壁)はそのまま使い、内部に2本の仕切り(上下対称)を追加するだけで、上下に同じ大きさの
        # 陣地+中央の隙間、という構成を実現する(外枠を丸ごと置き換えるhourglass等より安全)。
        # 弾は物理演算に乗せない直進判定(このファイル内コメント参照)のため、この仕切りに
        # 一切影響されず中央の隙間を越えて反対の陣地を攻撃できる(追加の特別対応は不要)。
        divider_y = half_y * SPLIT_HORIZONTAL_DIVIDER_FRAC
        return [
            [(-half_x, -divider_y), (half_x, -divider_y)],
            [(-half_x, divider_y), (half_x, divider_y)],
        ], []
    if terrain == "pegboard":
        # 小さな円形のピンを千鳥配置で多数配置し、軌道の複雑さ・予想外の跳ね返りを演出する
        # (プレイヤーがギリギリ通れる間隔にする)。下部(hole_fallの出口付近)は空けておく。
        circles = []
        y = -half_y + PEG_SPACING_Y
        row = 0
        y_limit = half_y * (1.0 - PEG_BOTTOM_MARGIN_RATIO)
        while y < y_limit:
            x_offset = (PEG_SPACING_X / 2) if row % 2 else 0.0
            x = -half_x + PEG_SPACING_X / 2 + x_offset
            while x < half_x - PEG_SPACING_X / 4:
                circles.append((x, y, PEG_RADIUS))
                x += PEG_SPACING_X
            y += PEG_SPACING_Y
            row += 1
        return [], circles
    if terrain == "pinball":
        # 2026-09-13: スリングショット(build_space側でPINBALL_BUMPER_RESTITUTIONと別枠で
        # 直接追加する。理由はbuild_space内のコメント参照)はここには含めず、通常の反発係数で
        # 済む誘導スロープだけをchainsとして返す(spawn安全判定にも自動的に反映される)。
        return _pinball_drain_chains(half_x, half_y), _pinball_bumper_positions(half_x, half_y)
    return [], []


def _hourglass_boundary_chain(half_x: float, half_y: float) -> list:
    """砂時計そのものが外枠になる場合の、閉じた境界の頂点列(中心(0,0)基準ローカル座標、
    上端の中点から始まり上端の中点に戻って閉じる8点)。

    2026-09-09(ユーザー報告のバグ調査で発見): 以前は「上左角」を始点・終点にしていたため、
    draw.line(..., joint="curve")では始点=終点の"wrap-around"部分(実際には上左角という
    本物の角)がPILの内部joint処理の対象にならず、そこだけ丸め処理が効かずくぼみが残っていた
    (他の5つの角は内部joint扱いになるため正しく丸められていた)。始点・終点を実際には
    「曲がっていない」直線区間である上端の中点に変更することで、全ての本物の角(6箇所)が
    内部jointとして扱われ、くぼみが解消する(始点=終点の場所自体は元々まっすぐな辺の途中
    なので、丸め処理の有無に関わらず見た目に影響しない)。
    縦方向のサイズはHOURGLASS_HEIGHT_RATIOで拡大する(ユーザー要望)。"""
    choke = CIRCLE_RADIUS * CHOKE_WIDTH_RATIO / 2
    wide_half = half_x * HOURGLASS_WIDTH_RATIO
    eff_half_y = half_y * HOURGLASS_HEIGHT_RATIO
    mid_top = (0.0, -eff_half_y)
    return [
        mid_top,
        (wide_half, -eff_half_y),
        (choke, 0.0),
        (wide_half, eff_half_y),
        (-wide_half, eff_half_y),
        (-choke, 0.0),
        (-wide_half, -eff_half_y),
        mid_top,
    ]


def _point_in_hourglass(x: float, y: float, half_x: float, half_y: float, margin: float = 0.0) -> bool:
    """スポーン安全性チェック用: 中心(0,0)基準ローカル座標の点が、砂時計の内側(壁から
    margin以上離れた位置)にあるかを判定する。上下対称なので|y|だけで計算する。
    _hourglass_boundary_chainと同じHOURGLASS_HEIGHT_RATIOを使い、常に実際の壁ジオメトリと
    一致させる(このずれが、後述のネック付近の補間バグと合わさって壁のすり抜けの原因だった)。"""
    eff_half_y = half_y * HOURGLASS_HEIGHT_RATIO
    if eff_half_y <= 0:
        return False
    choke = CIRCLE_RADIUS * CHOKE_WIDTH_RATIO / 2
    wide_half = half_x * HOURGLASS_WIDTH_RATIO
    ay = abs(y)
    if ay > eff_half_y - margin:
        return False
    # y=0(ネック)でchoke、y=half_y(外側)でwide_halfへ線形補間した、その高さでの壁のx位置。
    # 2026-09-09(ユーザー報告のバグ調査で発見): 以前はこの補間が逆向き(ネックで広く、外側で
    # 狭く判定)になっており、_hourglass_boundary_chainの実際の壁ジオメトリと不一致だった。
    # ネック付近で本来より広く「内側」と誤判定するため、スポーンした/移動中のプレイヤーが
    # 実際の壁をすり抜けて枠外に出るケースがあった(実機シミュレーションで確認・特定)。
    edge_x = choke + (wide_half - choke) * (ay / eff_half_y)
    return abs(x) < edge_x - margin


def _chains_to_segments(chains: list) -> list:
    """連続した折れ線の頂点列(chains)を、隣接ペアの(a, b)セグメント一覧に展開する。
    物理演算(pymunk.Segmentは2点しか取れない)とスポーン安全性チェックで使う。"""
    segments = []
    for chain in chains:
        for i in range(len(chain) - 1):
            segments.append((chain[i], chain[i + 1]))
    return segments


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """点(px,py)から線分(a,b)までの最短距離。スポーン位置が壁/terrainの壁セグメントに
    重ならないかを判定するために使う(_sample_position参照)。"""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _in_arena_bounds(x: float, y: float, arena_shape: str, terrain: str | None, half_x: float, half_y: float, margin: float) -> bool:
    """ワールド座標(x,y)が、地形を考慮した実際のプレイ可能領域の内側(margin分の余白込み)か
    どうかを判定する。2026-09-09、ユーザー指摘対応: teleport/poison_wireの配置先選びが
    この判定を一切していなかったため、donutの中心障害物やhourglassの非矩形な境界の外側に
    配置されてしまう不具合があった。spawn_circlesの_sample_positionと共通のロジックを関数化した。"""
    lx, ly = x - ARENA_CX, y - ARENA_CY
    if terrain == "hourglass":
        return _point_in_hourglass(lx, ly, half_x, half_y, margin)
    if arena_shape == "circle":
        mx, my = half_x - margin, half_y - margin
        if mx <= 0 or my <= 0:
            return False
        return (lx / mx) ** 2 + (ly / my) ** 2 <= 1.0
    return -half_x + margin <= lx <= half_x - margin and -half_y + margin <= ly <= half_y - margin


def _overlaps_terrain_obstacles(x: float, y: float, terrain_segments: list, terrain_circles: list, margin: float) -> bool:
    """ワールド座標(x,y)がterrain障害物(donutの中心円・cross/pegboardの壁やピン)にmargin未満まで
    近いかどうかを判定する。teleport/poison_wireの配置先選びとspawn_circlesで共通利用する。"""
    lx, ly = x - ARENA_CX, y - ARENA_CY
    for a, b in terrain_segments:
        if _point_segment_distance(lx, ly, a[0], a[1], b[0], b[1]) < margin:
            return True
    for cx, cy, r in terrain_circles:
        if ((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5 < r + margin:
            return True
    return False


# area_control専用: 保護ゾーン(2026-09-09、ユーザー要望)。「安全地帯が縮むだけで張り合いがない」
# という指摘への対応として、安全地帯の内側に数秒ごとに小さな保護ゾーンが出現し、触れたプレイヤーを
# 一定時間だけ安全地帯の外に出ても脱落しないよう保護する(逆転の可能性を作る駆け引き要素)。
# food/gun等と同じ「位置情報+距離判定のみ、専用pymunkボディは持たない」軽量パターンで実装する。
PROTECTION_ZONE_RADIUS = 45.0 * _ARENA_SCALE  # 未捕獲時のピックアップ判定・表示半径
# 2026-09-09(訂正): 保護ゾーンは「プレイヤーが一定時間無敵で自由に動ける」のではなく、
# 「固定座標のまま実体化し、プレイヤーをその中に閉じ込める(物理的な囲い)」仕様に訂正
# (ユーザー指摘)。実体化後の囲いの半径はピックアップ半径より大きくし、捕獲判定の瞬間に
# プレイヤーが確実に囲いの内側に収まるようにする。
PROTECTION_CAGE_RADIUS = PROTECTION_ZONE_RADIUS * 1.8
PROTECTION_ZONE_INTERVAL_SECONDS = 4.0  # 消化後、次が出現するまでの間隔(食べ物より少し長め)
PROTECTION_ZONE_DURATION_SECONDS = 4.0  # 実体化後、囲いが持続する秒数(ユーザー指定)
PROTECTION_ZONE_INITIAL_DELAY_SECONDS = 2.0  # 開幕直後の出現は忙しないため少し間を置く
# 2026-09-09、ユーザー指摘対応: 以前は誰も取らない限りいつまでも同じゾーンが残り続け、
# 「次のゾーンが出現しない」問題があった(未消化のまま無期限に居座っていた)。数秒間
# 誰にも捕獲されなければ消滅し、通常の間隔を置いて次のゾーンが出現するようにする。
PROTECTION_ZONE_LIFESPAN_SECONDS = 3.5


def _cage_local_segments(shape: str, radius: float) -> list:
    """保護ゾーンが実体化した際の囲いのローカル座標セグメント(中心(0,0)基準)。
    「枠とおなじ形状」(ユーザー指定)にするため、外枠のshapeパラメータをそのまま使う。
    常に隙間なし(hole_width=0相当)の密閉形状にする。"""
    if shape == "square":
        s = radius
        return [((-s, -s), (s, -s)), ((s, -s), (s, s)), ((s, s), (-s, s)), ((-s, s), (-s, -s))]
    n = 28
    angles = [i * (2 * math.pi / n) for i in range(n + 1)]
    pts = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    return [(pts[i], pts[i + 1]) for i in range(n)]

# 2026-09-12、ユーザー提供の設計指示書に基づき全面改修: 旧来の「ズームイン保持→直進移動→
# ズームアウト」3フェーズ(合計2.6秒)を廃止し、波紋(shockwave ring)による瞬時リセット方式
# (0.5秒)に短縮した。しかし実機確認で「全員が瞬間ワープするのはループ演出として不自然」との
# フィードバックを受け、同日中に折衷案へ再修正: 波紋の演出はそのまま活かしつつ、瞬間ワープを
# やめて短時間の滑らかな移動(旧来の「直進移動」を大幅短縮した版)に戻した。
# 2.6秒(長すぎてExit Sign的な間延びがある)と0.5秒(移動が一瞬すぎて不自然)の中間を取る。
EPILOGUE_SECONDS = 1.0
EPILOGUE_FRAMES = round(EPILOGUE_SECONDS * FPS)
# 決着直後、誰も動かさず一瞬だけ余韻を持たせる区間(旧デザインの「ズームイン保持」に相当するが
# 大幅短縮)。この後(HOLD_END〜1.0)が、全員(脱落済み含む)が滑らかに初期位置へ戻る移動フェーズ。
EPILOGUE_HOLD_END = 0.22
EPILOGUE_RING_WIDTH = 6.0  # 波紋の線の太さ(scale倍する)
# 2026-09-21、ユーザー指示「WIN表示のディレイとフェード」対応: 決着後0.5秒(phase_t換算)
# までWIN等のテキストを一切表示せず、そこからほぼ瞬間的に出現して0.3秒で即座にフェードアウト
# させる(=表示自体は決着後0.5〜0.8秒の間だけ)。「表示され続ける=もうすぐ終わる」という
# 合図でスワイプを誘発しないよう、旧来の「決着直後に出現し終盤までずっと表示」仕様
# (2026-09-13実装)から変更した。
EPILOGUE_TEXT_DELAY = 0.5
EPILOGUE_TEXT_FADE_OUT_DURATION = 0.3
EPILOGUE_TEXT_POPIN_FRAMES = 3  # 完全な瞬間表示(1フレーム)だと硬いため、ごく短いポップイン
SAMPLE_RATE = 44100
EFFECT_FRAMES = 24  # 脱落エフェクトの持続フレーム数(60fps基準)

# 8-3「演出のジューシーさ」: 結果には一切影響しない、体感満足度だけを上げる演出。
SHAKE_DURATION_FRAMES = 6  # 衝突1回あたりのスクリーンシェイクの持続フレーム数
SHAKE_MAX_OFFSET = 8.0  # 最大衝撃時のシェイク幅(arena空間の単位。scaleを乗じて実際のpxにする)
DECISION_BURST_FRAMES = 14  # 決着の瞬間の色フラッシュ+パーティクルの持続フレーム数
DECISION_PARTICLE_COUNT = 10

# カメラワーク: 追尾カメラ(camera="tracking")は、生存プレイヤー全員が収まる範囲を
# 動的にズーム/パンして映す。純粋な描画時の選択なので、simulate()ではなくrender()側の
# パラメータにしている(同じシミュレーション結果を異なるカメラ設定で描画し直せる)。
CAMERA_PADDING = 200.0 * _ARENA_SCALE
CAMERA_MIN_WIDTH = _DEFAULT_ARENA_LAYOUT["camera_min_width"]
CAMERA_SMOOTHING = 0.06  # 大きいほど追従が速い(0-1)

# スローモーション: 決着直前の一定時間を、動画・音声とも同じ倍率で引き伸ばす
# (音声は最近傍補間で伸ばすためピッチが下がる。スロー映像との相性がよい効果)
SLOWMO_WINDOW_SECONDS = 0.8
SLOWMO_FACTOR = 3

# 特殊能力(色番号→能力の対応)。色は下のCOLORSのインデックスに対応する
# (0:赤 1:青 2:緑 3:黄 4:紫 5:ティール 6:オレンジ 7:グレー)
ABILITY_BY_COLOR_INDEX = {
    0: "shockwave",      # 赤: 周囲の相手を吹き飛ばす
    1: "vortex",         # 青: 周囲の相手を自分の方へ引き寄せる(2026-09-09、旧freezeから変更。
                         # 減速のみの効果は終盤の見栄えを悪くするとのユーザー指摘のため、
                         # 「相手を強制的に遅くする」のではなく「衝突を誘発する」引き寄せ効果に変更。
                         # 今後も強制減速のみの効果は導入しない方針)
    2: "growth_surge",   # 緑: 自分が一時的に巨大化する
    3: "dash",           # 黄: 最も近い相手へ突進する
    4: "poison_wire",    # 紫: 有利な位置に一時的な障害物を設置する
    5: "teleport",       # ティール: ランダムな安全な位置へ瞬間移動する
    6: "speed_boost",    # オレンジ: 自分の速度を一気に上げる
    7: "slam",           # グレー: 自分を重力方向へ叩きつける
}
ABILITY_BASE_INTERVAL = 4.5
ABILITY_JITTER = 0.7

# 2026-09-21、ユーザー指示「カラーパレットの記号化」対応: 色と物理特性の役割を固定化し、
# 視聴者の認知負荷を下げる(「あの色は速い」「あの色は大きい」を繰り返し見て学習できるように)。
# 指示書で具体例が挙がった3色(赤=攻撃的・初速大、青=巨大・遅い、緑=逃げる)のみ定義し、
# 他の5色(黄/紫/ティール/オレンジ/グレー)は無理に役割をこじつけず既存の能力の個性(dash等)
# のままにする。speed_mult/size_multは開幕時のみ適用(_apply_color_traits参照)、
# fleeは試合中ずっと効く継続的な回避挙動(_apply_flee_behavior参照)。
COLOR_TRAIT_MODIFIERS = {
    0: {"speed_mult": 1.3},                  # 赤: 攻撃的・初速大
    1: {"size_mult": 1.4, "speed_mult": 0.7},  # 青: 巨大・遅い(ボスの色にも使う)
    2: {"flee": True},                       # 緑: 常に最も近い相手から逃げる
}
FLEE_ACCEL = 260.0  # 緑の継続的な回避加速度(逃げる方向への力)
ABILITY_EFFECT_FRAMES = 16

DASH_SPEED_BOOST = 520.0
POISON_WIRE_DURATION = 3.5
POISON_WIRE_LENGTH_FACTOR = 0.8
SHOCKWAVE_RADIUS = 220.0
SHOCKWAVE_IMPULSE = 420.0
VORTEX_RADIUS = 240.0
VORTEX_PULL_FORCE = 480.0  # 中心へ向かう速度成分としてどれだけ加えるか(shockwaveの逆方向版)
GROWTH_SURGE_MULT = 1.35
GROWTH_SURGE_DURATION = 3.0
SPEED_BOOST_MULT = 1.9
SPEED_BOOST_BASE = 320.0
SLAM_BOOST = 520.0

# absorb_growth専用: 「どちらが吸収するのか」を説明なしでわからせるための食べ物パワーアップ。
# ランダムな位置に出現し、触れると一定時間だけ相手を吸収できるようになる
# (2026-09-09、ユーザー要望。以前は常に大きい方が勝つだけで分かりにくかった)。
FOOD_PICKUP_RADIUS = 22.0 * _ARENA_SCALE
EMPOWERED_DURATION_SECONDS = 3.0
# 2026-09-16、冒頭フック強化(ユーザー指示): 以前は「開幕直後に出現すると忙しない」ため
# 1.0秒の間を置いていたが、Frame 0時点で目的(奪い合うもの)が見えている方が離脱防止に
# 優先すると判断し、即時出現(0秒)に変更した。
FOOD_INITIAL_DELAY_SECONDS = 0.0
FOOD_RESPAWN_DELAY_SECONDS = 1.5  # 食べられてから次が出現するまでの間隔

# absorb_growth専用: 早期決着を抑えるための2026-09-08追加ルール(ユーザー指摘対応)。
# 1) 吸収可能な状態(empowered)は、他プレイヤーを1体吸収した時点で即座に解除する
#    (同じ食べ物1個で連鎖的に何体も吸収できてしまうと瞬殺劇場になるため)。
# 2) 自分より大きい相手は吸収しきれない。吸収可能な状態でも自分より大きい相手に触れた場合は
#    全滅(マージ)ではなく、自分を一段階大きく・相手を一段階小さくする「かじり取り」に留める。
ABSORB_PARTIAL_STEP = 0.12  # 「一段階」の半径変化率
ABSORB_MIN_RADIUS = CIRCLE_RADIUS * 0.45  # かじり取られる側が縮みすぎて不安定にならないための下限

# goal_reach専用: ゴールの周りをバリアで囲み、何度か体当たりして破壊してからでないと
# 到達(勝利)できないようにする(2026-09-08、ユーザー要望)。
BARRIER_HITS_TO_BREAK = 3
BARRIER_RADIUS_MULT = 1.6  # GOAL_RADIUSに対する希望倍率。上端の壁とぶつからないよう下で実際にはクランプする

# gun_duel専用: 「銃奪い取り型」新ルール(2026-09-08、ユーザー要望で新規追加)。
# 銃に触れると一番近い相手へ照準し続け、GUN_AIM_SECONDS秒後に自動発射して弾を失う。
# シールドに触れるとSHIELD_DURATION_SECONDS秒だけ弾が効かなくなる(シールドは銃より低頻度)。
# 2026-09-09、尺ターゲット20〜40秒→15〜25秒への短縮に伴い、銃/シールドのサイクルを大幅に
# 短縮(aim3.0→1.0、各種delay/respawnも短縮)。それでも「未到達(90秒経っても決着しない)」
# 率が高いまま(実測40シード中約33%)残っており、gun_duelはこの尺ターゲットに対して
# 依然として最も弱いルールという既知の課題(タイマー調整だけでは解決しきらなかった)。
GUN_AIM_SECONDS = 1.0
# 2026-09-16、冒頭フック強化(ユーザー指示): 銃は「奪い合いの目的」を即座に見せる主役
# アイテムのため0秒即時出現に変更。シールドは防御系の副次アイテムのため、従来通り
# 少し間を置いて段階的に情報を出す(忙しなさ回避)方針を維持する。
GUN_INITIAL_DELAY_SECONDS = 0.0
GUN_RESPAWN_DELAY_SECONDS = 0.5
BULLET_SPEED = 1800.0  # 2026-09-20、命中率改善のため2倍化(元900.0)。_lead_aim_angle導入と合わせて
# 実測: 飛翔時間が短くなるほど「命中前に相手が動いてずれる」問題が緩和され、
# 決着率・尺の適合度が改善した(gun_duel全般に有効、new terrain限定の対症療法ではない)。
BULLET_RADIUS = 9.0
BULLET_MAX_SECONDS = 2.0  # この時間内に何にも当たらなければ消える(枠内なら必ず先に端へ届く)


def _lead_aim_angle(cx: float, cy: float, tx: float, ty: float, tvx: float, tvy: float, bullet_speed: float) -> float:
    """相手の現在位置への直線照準だけだと、着弾までの飛翔時間中に相手が動いてしまい
    命中しないケースが多い(2026-09-20、split_horizontal terrainの実測でundecidedが
    多発する原因として判明。射程が長いほど飛翔時間が伸び、着弾前に相手がずれる)。
    相手の現在速度から単純な線形先読み(intercept)の着弾時刻tを解いて狙う。
    解がない場合(相手が弾速より速く逃げ続ける等の特殊ケース)は従来通り
    現在位置への直線照準にフォールバックする。"""
    rx, ry = tx - cx, ty - cy
    a = tvx * tvx + tvy * tvy - bullet_speed * bullet_speed
    b = 2 * (rx * tvx + ry * tvy)
    c = rx * rx + ry * ry
    t = None
    if abs(a) > 1e-6:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            candidates = [x for x in ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)) if x > 0]
            if candidates:
                t = min(candidates)
    elif abs(b) > 1e-6:
        t_candidate = -c / b
        if t_candidate > 0:
            t = t_candidate
    if t is None:
        return math.atan2(ry, rx)
    return math.atan2(ry + tvy * t, rx + tvx * t)


SHIELD_DURATION_SECONDS = 4.0
SHIELD_INITIAL_DELAY_SECONDS = 1.5  # 銃より遅らせて段階的に情報を出す(2026-09-16、忙しなさ回避のため維持)
SHIELD_RESPAWN_DELAY_SECONDS = 2.5  # 銃より低頻度に出現(ユーザー指定の相対関係は維持)

# ============================================================
# weapon_colosseum: 新ルール(2026-09-20、ユーザー指示)。
# 剣/槍/ハンマー/弓矢/筆/斧の6種の武器を奪い合い、HP(10)を削り合う対戦形式。
# 他ルールの「1回の接触/場外/侵入で即脱落」とは異なり、HPが0になるまで生存する
# (decided_frame/winner_idの判定自体は既存の「生存者1人になったら決着」ロジックを
# そのまま流用でき、weapon_colosseum専用の分岐は不要 = _iter_rendered_frames以降の
# 決着処理コードに手を加える必要がない)。
# ============================================================
WEAPON_KINDS = ["sword", "spear", "hammer", "bow", "axe"]  # 2026-09-20、ユーザー指示で筆(brush)を廃止
WEAPON_STARTING_HP = 10
WEAPON_INITIAL_DELAY_SECONDS = 0.0
# 2026-09-20、ユーザー指示により「1ゲーム内での武器の再出現はなし(無駄に散らかる)。
# 一度所持した武器はずっと持つ」仕様に変更。各種類1回だけ出現させ、拾われても再出現
# させない(simulate()内のweapon_ever_spawned参照)。以前あったWEAPON_RESPAWN_DELAY_SECONDS
# (再出現間隔)は不要になったため削除した。
WEAPON_PICKUP_RADIUS = FOOD_PICKUP_RADIUS

# ============================================================
# 対戦形式(match_type): 個人戦(individual、既定)/チーム戦(team)/ボス戦(boss)
# 2026-09-21、ユーザー指示「新軸」対応。既存のrule(勝敗条件のロジック)とは独立した軸。
# チーム戦はhole_fall/goal_reach/area_control/gun_duel/weapon_colosseumの5ルールに、
# ボス戦はweapon_colosseum/gun_duelの2ルールに対応させる(absorb_growthはそもそも
# 「接触で吸収・成長する」ことがゲーム性の中心でチーム制と相性が悪いため対象外、
# の2点ともユーザー確定済み)。
# ============================================================
BOSS_SIZE_MULT = 1.7  # 「ふたまわり大きい」の目安。物理半径にもそのまま反映する(_resize_entity使用)
BOSS_HP_MULT = 3.0  # weapon_colosseum専用: ボスのHPを通常の3倍にする(有利ステータス)
BOSS_BULLET_HITS_TO_ELIMINATE = 3  # gun_duel専用: ボスは弾を3発受けるまで脱落しない(有利ステータス)

# 2026-09-20、ユーザー指示で筆(brush)を廃止した影響で総ダメージ量が減り、決着の尺が
# 伸びた(中央値21.8秒→27.4秒、目標15〜25秒)ため、全武器のクールダウンを0.7倍に再調整した
# (実測80シード: undecided=0、median=22.1秒、目標レンジ到達60/80)。
# 2026-09-21追記: 近接武器(剣/槍/ハンマー)のクールタイムを「攻撃側個体」単位から
# 「(攻撃側,対象)の組み合わせ」単位に変更(複数の相手を同時期に攻撃できるように)した影響で
# 総ダメージ量が増え、決着が速くなりすぎた(median15.2秒)ため、全武器のクールダウンを
# さらに1.7倍に再調整した(実測100シード: undecided=0、median=19.9秒、目標レンジ到達62/100)。
# 2026-09-21再追記(ユーザー指示「数十回シミュレーションしてバランス調整」): 250シードで
# 勝者が最後に保持していた武器を集計したところ、槍が52.7%を占める一方でハンマーは10%と
# 大きく偏っていた(主な原因は槍の射程4.2rが他の近接武器の2.6rよりずっと長く、常時有効
# だったこと)。槍の射程を短縮し、ハンマーはダメージを1→2に引き上げた上でクールダウンを
# 若干伸ばし、弓/斧はクールダウンを短縮して出番を増やすことで、最終的に5種の勝率を
# 18.8%〜21.2%の範囲(理想20%)まで均した(実測250シード: undecided=0、median=16.5秒)。
SWORD_DAMAGE = 2
SWORD_RANGE = CIRCLE_RADIUS * 2.6
SWORD_COOLDOWN_SECONDS = 2.46
SWORD_SPIN_SPEED = 6.0  # rad/秒。見た目の回転のみで当たり判定には影響しない(指示書「回転する」への対応)

SPEAR_DAMAGE = 2
SPEAR_WALL_BONUS_DAMAGE = 1  # 壁に押し付けた状態で命中すると合計3
SPEAR_RANGE = CIRCLE_RADIUS * 2.4  # 2026-09-21、バランス調整で4.2倍→2.4倍に短縮(勝率52.7%→20%付近)
SPEAR_COOLDOWN_SECONDS = 2.86
SPEAR_WALL_MARGIN = CIRCLE_RADIUS * 1.6  # 相手がこの距離以内の壁にいれば「押し付けた」とみなす
SPEAR_PIN_DURATION_SECONDS = 0.4  # ピン留め中は移動不能にする

HAMMER_DAMAGE = 2  # 2026-09-21、バランス調整で1→2に引き上げ(ノックバックだけでは勝率10%と弱すぎた)
HAMMER_RANGE = CIRCLE_RADIUS * 2.6
HAMMER_COOLDOWN_SECONDS = 2.875
HAMMER_KNOCKBACK = 520.0 * _ARENA_SCALE

BOW_DAMAGE = 2
BOW_COOLDOWN_SECONDS = 2.14  # 2026-09-21、バランス調整で短縮(出番が少なく勝率が低かったため)
ARROW_SPEED = 780.0 * _ARENA_SCALE
ARROW_GRAVITY = 640.0 * _ARENA_SCALE  # 弾道を落とす専用の追加重力(通常の物理重力とは別に矢にのみ加算)
ARROW_RADIUS = 7.0 * _ARENA_SCALE
ARROW_MAX_SECONDS = 2.0
BOW_RECOIL = 260.0 * _ARENA_SCALE  # 発射の反動で射手が逆方向へ押される

# 2026-09-21、ユーザー指示で斧の挙動を全面刷新(3回目、最終版): 「基本は止めて、定期的に
# 当たり判定・ぶっ飛ばし判定つきの高速回転をする」というシンプルな仕様に変更した。
# 静止中は他の近接武器と同じ汎用の構え(狙っている相手の方向)で表示し、一定間隔で
# 剣/ハンマーの周回演出を大幅に速くしたような高速回転攻撃を行う。回転中に届く範囲内の
# 相手は(攻撃側,対象)ごとのクールタイム(melee_pair_cooldown)で独立にヒットするため、
# 1回の回転で複数の相手を巻き込める。
AXE_DAMAGE = 3
AXE_SPIN_DURATION_SECONDS = 0.4  # 高速回転している時間
AXE_SPIN_SPEED = 22.0  # rad/秒(剣のSWORD_SPIN_SPEEDの3倍以上、「高速回転」を表現)
AXE_SPIN_REACH = CIRCLE_RADIUS * 3.4  # 回転中に当たり判定が届く、プレイヤー中心からの距離
AXE_KNOCKBACK = 600.0 * _ARENA_SCALE
AXE_COOLDOWN_SECONDS = 2.06  # 2026-09-21、バランス調整で短縮(出番が少なく勝率が低かったため)
AXE_HELD_OFFSET_MULT = 1.7  # 「より前に出す」ための保持位置オフセット倍率(他の近接武器の1.0倍より大きい)

# 2026-09-21、ユーザー指示: ヒットストップは「速度を1回ゼロにするだけ」だと、命中のたびに
# 運動量を完全に失ってしまい(その後は重力/衝突で一から速度を作り直すしかない)、命中回数の
# 多い試合ほど終盤に動きが鈍くなっていく問題があった。ユーザー提案の「直前の速度を保存して
# おき、短時間ゼロにした後に保存していた速度(+その間に本来受けていたはずの重力分)を
# 復元する」方式に変更し、見た目上は一瞬静止するが運動量は失われないようにする。
HITSTOP_FRAMES = 4

# 各特殊能力パラメータの安全な可変範囲(min, max)。ability_paramsによる新規バリアント生成
# (geometry_battle_idea_generator.py)が、物理演算が破綻しない範囲でのみ値を提案できるようにする。
# 範囲の中心はおおよそ既存の既定値(このファイル内の定数)に合わせている。
ABILITY_PARAM_SCHEMA = {
    "dash": {"speed_boost": (300.0, 900.0)},
    "poison_wire": {"duration": (2.0, 5.0), "length_factor": (0.5, 1.1)},
    "shockwave": {"radius": (120.0, 350.0), "impulse": (200.0, 700.0)},
    "vortex": {"radius": (120.0, 350.0), "pull_force": (200.0, 800.0)},
    "growth_surge": {"mult": (1.15, 1.8), "duration": (1.5, 5.0)},
    "speed_boost": {"mult": (1.3, 2.6), "base": (200.0, 500.0)},
    "slam": {"boost": (300.0, 800.0)},
    # teleportは位置のみで数値パラメータを持たないため対象外
}

RULE_CAPTIONS = {
    "hole_fall": "LAST ONE STANDING WINS",
    "goal_reach": "FIRST TO THE GOAL WINS",
    "area_control": "STAY IN THE ZONE TO WIN",
    "absorb_growth": "ABSORB THEM ALL TO WIN",
    "gun_duel": "GRAB THE GUN, SURVIVE",
    "weapon_colosseum": "GRAB WEAPONS, DRAIN HP",
}

# 視覚テーマ(配色パレット)。並び順・要素数はCOLORSと揃えること
# (色インデックス→特殊能力/名前/タイプの対応は色番号ベースなので、パレットを変えても
# ゲーム性には影響しない、純粋な見た目の軸)
PALETTES = {
    "vivid": [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    ],
    "pastel": [
        (247, 168, 168),
        (168, 216, 234),
        (181, 234, 195),
        (250, 230, 160),
        (214, 189, 232),
        (170, 230, 219),
        (250, 200, 165),
        (210, 210, 215),
    ],
    "neon": [
        (255, 45, 85),
        (0, 200, 255),
        (80, 255, 120),
        (255, 235, 0),
        (200, 0, 255),
        (0, 255, 210),
        (255, 130, 0),
        (230, 230, 240),
    ],
    # 2026-09-09修正: 旧sunsetは暖色に寄せすぎて色相の並び(赤/青/緑/黄/紫/ティール/橙/灰)が
    # 他パレットと揃っておらず(例: 本来「青」枠のindex1が紫になっていた)、同じ能力が
    # パレットによって別の色系統に見えてしまっていた(ユーザー指摘: 色と能力を対応させること)。
    # 「夕焼け」の温かみを保ちつつ、各indexの色相ファミリーは他パレットと揃えて修正した。
    "sunset": [
        (233, 90, 80),   # 赤
        (70, 95, 170),   # 青
        (100, 165, 90),  # 緑
        (255, 210, 70),  # 黄
        (170, 90, 170),  # 紫
        (90, 150, 145),  # ティール
        (240, 140, 60),  # 橙
        (170, 145, 140),  # 灰(暖かみのあるグレー)
    ],
}
DEFAULT_PALETTE = "vivid"
COLORS = PALETTES[DEFAULT_PALETTE]  # 後方互換用のデフォルト参照(能力割り当て等の色インデックス計算はlen(COLORS)基準)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "scripts_templates" / "geometry_test_clips"


@dataclass
class CircleEntity:
    id: int
    body: pymunk.Body
    shape: pymunk.Circle
    color: tuple
    radius: float = CIRCLE_RADIUS
    alive: bool = True
    eliminated_frame: int | None = None
    ability: str | None = None
    ability_timer: float = 0.0
    player_shape: str = "circle"
    empowered_until_frame: int = -1  # absorb_growth専用: この値のフレームまでは相手を吸収できる
    holding_gun: bool = False  # gun_duel専用: 銃を保持中か
    gun_fire_frame: int = -1  # gun_duel専用: この値のフレームで自動発射する(-1は非保持)
    shielded_until_frame: int = -1  # gun_duel専用: この値のフレームまでは弾が効かない
    protected_until_frame: int = -1  # area_control専用: この値のフレームまでは安全地帯の外でも脱落しない
    hp: int = WEAPON_STARTING_HP  # weapon_colosseum専用: 0になった時点で脱落
    weapon: str | None = None  # weapon_colosseum専用: 保持中の武器種(WEAPON_KINDS)
    weapon_cooldown_until_frame: int = -1  # weapon_colosseum専用: この値のフレームまで次の攻撃ができない
    # weapon_colosseum(斧)専用: 2026-09-21、ユーザー指示で全面刷新(3回目)。基本は静止、
    # 一定間隔で高速回転する攻撃(当たり判定+ぶっ飛ばし判定つき)を行う、というシンプルな
    # 仕様に変更した。この値のフレームまで高速回転(攻撃中)する。
    axe_spin_until_frame: int = -1
    pinned_until_frame: int = -1  # weapon_colosseum専用: 槍で壁に押し付けられ、この値のフレームまで動けない
    hitstop_until_frame: int = -1  # weapon_colosseum専用: 命中の瞬間からこの値のフレームまで速度を凍結する
    hitstop_saved_vx: float = 0.0  # weapon_colosseum専用: 凍結前の速度(x)。解除時に復元する
    hitstop_saved_vy: float = 0.0  # weapon_colosseum専用: 凍結前の速度(y)。解除時に復元する
    # 2026-09-21、ユーザー指示「対戦形式(match_type)の新軸」対応。
    # team_id: チーム戦(match_type="team")専用。個人戦/ボス戦ではNoneのまま
    # (Noneは「誰とも同じチームでない」を意味し、_is_teammate()が常にFalseを返す)。
    team_id: int | None = None
    is_boss: bool = False  # ボス戦(match_type="boss")専用: このエンティティがボスかどうか
    # ボス専用の2種類のスキル(既存8種の特殊能力から2つ)。通常のability/ability_timerは
    # そのまま使い、クールダウンが明けるたびにboss_abilities内を交互に切り替えて発動する
    # (dispatch側の8分岐if/elifは変更せず、「今回どの能力を使うか」だけをボスは複数持てる、
    # という最小限の拡張にとどめている)。
    boss_abilities: list = field(default_factory=list)
    boss_ability_index: int = 0
    color_index: int = 0  # 2026-09-21、色の物理特性記号化対応: 能力割り当てと同じ「色スロット」番号(0-7)
    bullet_hits_remaining: int = 1  # gun_duel専用: 被弾から脱落までに耐えられる残り回数(ボスのみ1より大きい)


@dataclass
class SimResult:
    seed: int
    n_circles: int
    gravity: float
    elasticity: float
    shape: str = DEFAULT_SHAPE
    rule: str = DEFAULT_RULE
    frames: list = field(default_factory=list)
    arena_angles: list = field(default_factory=list)
    elimination_order: list = field(default_factory=list)
    collisions: list = field(default_factory=list)  # [{"frame": int, "impulse": float}]
    ability_events: list = field(default_factory=list)  # [{"type","id","frame","x","y"}]
    wires: list = field(default_factory=list)  # [{"id","start_frame","end_frame","a","b"}]
    food_items: list = field(default_factory=list)  # absorb_growth専用: [{"x","y","start_frame","end_frame","eaten_by"}]
    protection_zones: list = field(default_factory=list)  # area_control専用: [{"x","y","radius","cage_radius","start_frame","end_frame","cage_end_frame","claimed_by"}]
    goal_barrier: dict | None = None  # goal_reach専用: {"x","y","radius","hits_to_break","break_frame"}
    guns: list = field(default_factory=list)  # gun_duel専用: [{"x","y","start_frame","end_frame","picked_by"}]
    shields: list = field(default_factory=list)  # gun_duel専用: [{"x","y","start_frame","end_frame","picked_by"}]
    bullets: list = field(default_factory=list)  # gun_duel専用: [{"shooter_id","x","y","angle","start_frame","end_frame","hit_id","blocked"}]
    weapons: list = field(default_factory=list)  # weapon_colosseum専用: [{"kind","x","y","start_frame","end_frame","picked_by"}]
    arrows: list = field(default_factory=list)  # weapon_colosseum(弓矢)専用: [{"shooter_id","x","y","vx","vy","start_frame","end_frame","hit_id"}]
    winner_id: int | None = None
    match_type: str = "individual"  # 2026-09-21追加: "individual"|"team"|"boss"
    winning_team_id: int | None = None  # match_type="team"専用: 勝利したチームのteam_id
    winner_is_boss: bool = False  # match_type="boss"専用: ボス側が勝ったかどうか
    decided_frame: int | None = None
    final_position: tuple | None = None
    final_player_angle: float = 0.0
    final_arena_angle: float = 0.0
    rotation_speed: float = 0.0
    hole_width: float = DEFAULT_HOLE_WIDTH
    accel_zone: bool = False
    trap: bool = False
    player_shape: str = DEFAULT_PLAYER_SHAPE
    arena_half_x: float = ARENA_HALF_EXTENT  # 2026-09-08、ステージサイズのバリエーション対応(tallで2軸化)
    arena_half_y: float = ARENA_HALF_EXTENT
    terrain: str | None = None  # 2026-09-09、新ステージ内部構造(hourglass/donut/cross/pegboard)


_WALL_SEGMENT_CACHE: dict = {}


def _wall_local_segments(shape: str, hole_width: float, half_x: float = ARENA_HALF_EXTENT, half_y: float = ARENA_HALF_EXTENT):
    """枠(壁)のローカル座標セグメント一覧。物理演算・描画の両方でこれを共有する
    (別々に持つと、パラメータを変えたときに片方だけ古い値のまま、というバグの元になる)。

    2026-09-09、ユーザー要望により真円/正方形(縦横ともARENA_HALF_EXTENT)に変更した。
    以前はARENA_CX(横)/ARENA_CY(縦)を別々に使う縦長の楕円/矩形で、回転すると
    キャンバス端で見切れる問題があったが、真円/正方形なら見切れない
    (円は回転で輪郭が変わらない、正方形は対角線がARENA_Wにちょうど収まるサイズにしてある)。

    2026-09-08、ステージサイズのバリエーション(ARENA_SIZE_VARIANTS)対応によりhalf_x/half_yを
    引数化した(tall等、縦横比を変える形状に対応するため2軸に一般化)。half_x!=half_y
    (非等方)、または既定値(ARENA_HALF_EXTENT)より大きい等方の正方形は回転で見切れうるため、
    呼び出し側(simulate())がその場合は自動的に回転を止める。
    """
    key = (shape, hole_width, half_x, half_y)
    if key in _WALL_SEGMENT_CACHE:
        return _WALL_SEGMENT_CACHE[key]

    half_hole = hole_width / 2
    if shape == "square":
        hx, hy = half_x, half_y
        segments = [
            ((-hx, -hy), (-hx, hy)),
            ((hx, -hy), (hx, hy)),
            ((-hx, -hy), (hx, -hy)),
            ((-hx, hy), (-half_hole, hy)),
            ((half_hole, hy), (hx, hy)),
        ]
    elif shape == "circle":
        n = 72
        half_angle = math.asin(min(0.99, half_hole / half_x)) if hole_width > 0 else 0.0
        gap_center = math.pi / 2  # 下方向(局所座標のy+)
        gap_lo, gap_hi = gap_center - half_angle, gap_center + half_angle
        angles = [i * (2 * math.pi / n) for i in range(n + 1)]
        points = [(half_x * math.cos(a), half_y * math.sin(a)) for a in angles]
        segments = []
        for i in range(n):
            a0, a1 = angles[i], angles[i + 1]
            if hole_width > 0 and (gap_lo <= a0 <= gap_hi or gap_lo <= a1 <= gap_hi):
                continue
            segments.append((points[i], points[i + 1]))
    else:
        raise ValueError(f"unknown shape: {shape}")

    _WALL_SEGMENT_CACHE[key] = segments
    return segments


def build_space(
    gravity: float,
    elasticity: float,
    rotation_speed: float,
    hole_width: float,
    damping: float,
    shape: str,
    friction: float,
    half_x: float = ARENA_HALF_EXTENT,
    half_y: float = ARENA_HALF_EXTENT,
    terrain: str | None = None,
) -> tuple[pymunk.Space, pymunk.Body]:
    """枠(壁+脱出口)を1つのkinematic bodyにまとめ、回転できるようにする。
    脱出口も枠と一緒に回転するため、脱出できるタイミングが重力方向との噛み合わせで決まる。
    hole_width=0の場合は隙間なしの密閉された枠になる(goal_reach/area_control/absorb_growth用)。

    terrain(2026-09-09追加): 枠の内部に追加する地形障害物。壁と同じarena_bodyに追加するため、
    回転ギミックが作動した際も外枠と一緒に破綻なく回転する。
    """
    space = pymunk.Space()
    space.gravity = (0, gravity)
    space.damping = damping

    arena_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    arena_body.position = (ARENA_CX, ARENA_CY)
    arena_body.angular_velocity = rotation_speed
    space.add(arena_body)

    if terrain == "hourglass":
        # 2026-09-09、ユーザー要望: 通常の四角/円の壁は作らず、砂時計のシルエットそのものを
        # 外枠にする(_hourglass_boundary_chain、始点に戻って閉じた1本の折れ線)。
        chain = _hourglass_boundary_chain(half_x, half_y)
        for i in range(len(chain) - 1):
            wall_shape = pymunk.Segment(arena_body, chain[i], chain[i + 1], WALL_THICKNESS / 2)
            wall_shape.friction = friction
            wall_shape.elasticity = elasticity
            wall_shape.collision_type = 1
            space.add(wall_shape)
    elif terrain == "pin_wall":
        # 2026-09-13: 外枠そのものを、隙間なく重なり合う小さなピンの列に置き換える
        # (フラットな壁と違い、当たる位置によって跳ね返る角度が読めなくなる)。
        # collision_type=9は他の壁/terrain(1)と区別するための専用値で、simulate()の
        # on_beginでピン衝突をヒットフラッシュ・効果音のトリガーとして検出するのに使う。
        for x, y, r in _pin_wall_positions(shape, hole_width, half_x, half_y):
            pin_shape = pymunk.Circle(arena_body, r, (x, y))
            pin_shape.friction = friction
            pin_shape.elasticity = PIN_WALL_RESTITUTION
            pin_shape.collision_type = 9
            space.add(pin_shape)
    else:
        for a, b in _wall_local_segments(shape, hole_width, half_x, half_y):
            wall_shape = pymunk.Segment(arena_body, a, b, WALL_THICKNESS / 2)
            wall_shape.friction = friction
            wall_shape.elasticity = elasticity
            wall_shape.collision_type = 1
            space.add(wall_shape)

    terrain_chains, terrain_circles = _terrain_obstacles(terrain, half_x, half_y)
    for a, b in _chains_to_segments(terrain_chains):
        obstacle_shape = pymunk.Segment(arena_body, a, b, WALL_THICKNESS / 2)
        obstacle_shape.friction = friction
        obstacle_shape.elasticity = elasticity
        obstacle_shape.collision_type = 1
        space.add(obstacle_shape)
    for cx, cy, r in terrain_circles:
        obstacle_shape = pymunk.Circle(arena_body, r, (cx, cy))
        obstacle_shape.friction = friction
        if terrain == "pinball":
            # 2026-09-13、指示書対応: ポップバンパーはPINBALL_BUMPER_RESTITUTION(1.3〜1.5)で
            # 積極的に弾き飛ばす。2026-09-09に判明した「反発係数>1.0だと運動エネルギーが
            # 際限なく増え続け壁をすり抜ける」不具合(このすぐ下のコメント参照)を踏まないよう、
            # terrain=="pinball"選択時は毎フレームPINBALL_MAX_SPEEDで速度をクランプする安全弁を
            # simulate()のメインループに設けている。collision_type=8は他の地形障害物(1)と
            # 区別するための専用値で、on_beginでヒットフラッシュ・効果音のトリガーに使う。
            obstacle_shape.elasticity = PINBALL_BUMPER_RESTITUTION
            obstacle_shape.collision_type = 8
        else:
            # 2026-09-09(ユーザー指摘で修正): 反発係数を1.0超にすると衝突のたびに運動エネルギーが
            # 増え続け、加速したプレイヤーが1フレームで外壁を通り抜けてしまう(壁の外に出て
            # そのまま決着してしまう)不具合があった。他の壁と同じelasticityを使い、
            # 「速度が上がらずそのまま跳ね返る」仕様に修正。
            obstacle_shape.elasticity = elasticity
            obstacle_shape.collision_type = 1
        space.add(obstacle_shape)

    if terrain == "pinball":
        # 2026-09-13、指示書対応: スリングショットは通常のterrain_chains(誘導スロープ)とは
        # 別枠で追加し、SLINGSHOT_RESTITUTION(高反発)を個別に適用する
        # (誘導スロープは滑らせて中央へ誘導する役割のため、通常のelasticityのまま据え置く)。
        for a, b in _pinball_slingshot_chains(half_x, half_y):
            slingshot_shape = pymunk.Segment(arena_body, a, b, WALL_THICKNESS / 2)
            slingshot_shape.friction = friction
            slingshot_shape.elasticity = SLINGSHOT_RESTITUTION
            slingshot_shape.collision_type = 1
            space.add(slingshot_shape)

    return space, arena_body


def _poly_vertices(player_shape: str, radius: float) -> list | None:
    """プレイヤー形状(square/triangle)のローカル頂点。半径はおおよそ円と footprint が
    近くなるよう、外接円半径として扱う。circleの場合はNone(Circle形状を使うため頂点不要)。
    """
    if player_shape == "square":
        s = radius * 0.82  # 半径基準の半辺(円とほぼ同等の見た目サイズになるよう経験的に調整)
        return [(-s, -s), (s, -s), (s, s), (-s, s)]
    if player_shape == "triangle":
        pts = []
        for i in range(3):
            a = -math.pi / 2 + i * (2 * math.pi / 3)
            pts.append((radius * math.cos(a), radius * math.sin(a)))
        return pts
    return None


def _make_player_shape(body: pymunk.Body, radius: float, player_shape: str) -> pymunk.Shape:
    if player_shape == "circle":
        return pymunk.Circle(body, radius)
    verts = _poly_vertices(player_shape, radius)
    return pymunk.Poly(body, verts)


def _player_moment(mass: float, radius: float, player_shape: str) -> float:
    if player_shape == "circle":
        return pymunk.moment_for_circle(mass, 0, radius)
    # 2026-09-08、ユーザー指摘により修正: 四角形は重力の影響こそ受けないが、回転は固定せず
    # 通常の慣性モーメントにする(衝突等の外力に応じて回転できるようにする)。一時無回転にして
    # いたが、ユーザーの意図は「無回転」ではなく「重力に引っ張られない・直進して反射する」
    # ことだったため、square専用の分岐(float("inf"))は撤廃し、他形状と同じ計算式を使う。
    verts = _poly_vertices(player_shape, radius)
    return pymunk.moment_for_poly(mass, verts)


def _no_gravity_velocity_func(body: pymunk.Body, gravity: tuple, damping: float, dt: float) -> None:
    """player_shape="square"専用(2026-09-08、ユーザー要望): 重力の影響を受けず常に直進させる。
    pymunkの既定の速度更新(Body.update_velocity)からgravity項だけを除いたもの。
    壁や相手プレイヤーとの衝突による反射(弾性衝突)は通常のpymunk物理演算がそのまま処理するため、
    ここでは減衰(damping)だけ適用すればよい。"""
    body.velocity = body.velocity * damping


def spawn_circles(
    space: pymunk.Space,
    n: int,
    seed: int,
    elasticity: float,
    friction: float,
    y_range: tuple | None = None,
    player_shape: str = "circle",
    palette: str = DEFAULT_PALETTE,
    arena_shape: str = "square",
    half_x: float = ARENA_HALF_EXTENT,
    half_y: float = ARENA_HALF_EXTENT,
    rule: str | None = None,
    terrain: str | None = None,
    match_type: str = "individual",
    team_count: int = 2,
    boss_abilities: tuple[str, str] | None = None,
    boss_size_mult: float = BOSS_SIZE_MULT,
) -> list[CircleEntity]:
    rng = random.Random(seed)
    terrain_chains, terrain_circles = _terrain_obstacles(terrain, half_x, half_y)
    terrain_segments = _chains_to_segments(terrain_chains)
    terrain_margin = CIRCLE_RADIUS + WALL_THICKNESS / 2 + 15 * _ARENA_SCALE

    def _overlaps_terrain(x: float, y: float) -> bool:
        for a, b in terrain_segments:
            if _point_segment_distance(x - ARENA_CX, y - ARENA_CY, a[0], a[1], b[0], b[1]) < terrain_margin:
                return True
        for cx, cy, r in terrain_circles:
            if ((x - ARENA_CX - cx) ** 2 + (y - ARENA_CY - cy) ** 2) ** 0.5 < r + terrain_margin:
                return True
        return False
    circles = []
    palette_colors = PALETTES.get(palette, PALETTES[DEFAULT_PALETTE])
    margin = CIRCLE_RADIUS + 20
    layout = _arena_layout(half_x, half_y)
    # 2026-09-09: 枠が真円/正方形化(ARENA_HALF_EXTENT基準)されたため、既定のスポーン範囲も
    # キャンバス全体ではなく枠の内側(ARENA_TOP〜中心よりやや上)に収める
    # (2026-09-08: ステージサイズのバリエーション対応によりhalf_x/half_y基準に変更)
    y_lo, y_hi = y_range if y_range is not None else (layout["top"] + margin, ARENA_CY - half_y * 0.16)
    x_lo, x_hi = layout["left"] + margin, layout["right"] - margin
    # 2026-09-09バグ修正: arena_shape="circle"の場合、x/yを単純に矩形範囲でサンプリングすると
    # 「四隅」が真円の外側になってしまい(円に内接する矩形の外側)、初期位置が枠外になって
    # 即脱落するプレイヤーが発生していた(実測で約1割)。circle形状の時だけ、中心からの距離が
    # 枠の半径に収まるまで再抽選する(30回試しても見つからなければ中心方向へクランプする)。
    # 2026-09-08: tall等の非等方ステージに対応するため楕円の内外判定に一般化。
    mx, my = half_x - margin, half_y - margin

    # 2026-09-09、ユーザー指示: goal_reachで、開始直後にバリア(GOAL_RADIUS*BARRIER_RADIUS_MULT)へ
    # 密着スポーンして即座に体当たりで破壊→ゴール、という「つまらない決着」を物理的に禁止する。
    # ゴール・バリアは回転しない(space.static_body、arena_bodyの回転に追従しない)ため、
    # スポーン時点の固定距離チェックで恒久的に安全(rotation_speedの影響を受けない)。
    goal_point = layout["goal_point"]
    min_goal_distance = (GOAL_RADIUS * BARRIER_RADIUS_MULT + CIRCLE_RADIUS + 60 * _ARENA_SCALE) if rule == "goal_reach" else 0.0
    # 2026-09-09、ユーザー要望でhourglassが外枠そのものになったため、arena_shape(square/circle)
    # 基準の内外判定は使えない。_point_in_hourglassを唯一の内外判定として使う。
    hourglass_margin = CIRCLE_RADIUS + 15 * _ARENA_SCALE
    _band_cursor = [0]  # two_tier/split_horizontalの交互配置に使う(閉じ込め済みの変数)

    def _sample_position() -> tuple[float, float]:
        # 2026-09-20、ユーザー指示: two_tier/split_horizontalは、ランダム抽選+当たり判定
        # 回避ではなく、意図した位置(足場の上/上下どちらかの陣地)に直接決定論的に配置する
        # (ランダム抽選だと大半のプレイヤーが足場を外れて即落下したり、陣地の偏りが
        # 起きたりするため)。
        if terrain == "two_tier":
            platforms = _two_tier_platforms(half_x, half_y)
            pcx, edge_y, phl, peak_y = platforms[_band_cursor[0] % len(platforms)]
            _band_cursor[0] += 1
            # 2026-09-20、三角形化(山型)に伴い、スポーン地点を足場全体への一様ジッターから
            # 頂点付近に変更(ユーザー指示「上段でスポーン」→頂点から左右どちらかの斜面へ
            # 滑り落ちる「枝分かれ」の起点にする)。同じ足場に複数人乗ることもあるため、
            # 重なり回避の小さなジッターだけ与える。
            jitter = rng.uniform(-phl * 0.15, phl * 0.15)
            return ARENA_CX + pcx + jitter, ARENA_CY + peak_y - CIRCLE_RADIUS - 12 * _ARENA_SCALE
        if terrain == "split_horizontal":
            divider_y = half_y * SPLIT_HORIZONTAL_DIVIDER_FRAC
            band_margin = CIRCLE_RADIUS + 20
            top_band = _band_cursor[0] % 2 == 0
            _band_cursor[0] += 1
            bx = rng.uniform(x_lo, x_hi)
            if top_band:
                by = rng.uniform(ARENA_CY - half_y + band_margin, ARENA_CY - divider_y - band_margin)
            else:
                by = rng.uniform(ARENA_CY + divider_y + band_margin, ARENA_CY + half_y - band_margin)
            return bx, by
        x = y = 0.0
        for _ in range(30):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if terrain == "hourglass":
                if _point_in_hourglass(x - ARENA_CX, y - ARENA_CY, half_x, half_y, hourglass_margin):
                    return x, y
                continue
            in_bounds = arena_shape != "circle" or (((x - ARENA_CX) / mx) ** 2 + ((y - ARENA_CY) / my) ** 2) <= 1.0
            if not in_bounds:
                continue
            if min_goal_distance and ((x - goal_point[0]) ** 2 + (y - goal_point[1]) ** 2) ** 0.5 < min_goal_distance:
                continue
            if _overlaps_terrain(x, y):
                continue
            return x, y
        if terrain == "hourglass":
            return ARENA_CX, ARENA_CY - half_y * 0.3  # 上チャンバー内の安全な既定位置(フォールバック)
        # 2026-09-09、terrain対応: 30回試しても空きスペースが見つからない場合(密なpegboard等)は、
        # terrain判定だけ無視して再抽選する(即自滅にはならない範囲内蔵/壁マージンのみのフォールバック。
        # 完全に埋まった配置になることは現状のterrain密度では想定していない)。
        for _ in range(20):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            in_bounds = arena_shape != "circle" or (((x - ARENA_CX) / mx) ** 2 + ((y - ARENA_CY) / my) ** 2) <= 1.0
            if in_bounds and not (min_goal_distance and ((x - goal_point[0]) ** 2 + (y - goal_point[1]) ** 2) ** 0.5 < min_goal_distance):
                return x, y
        angle = math.atan2((y - ARENA_CY) / my, (x - ARENA_CX) / mx)
        return ARENA_CX + math.cos(angle) * mx * 0.9, ARENA_CY + math.sin(angle) * my * 0.9

    for i in range(n):
        mass = 1.0
        moment = _player_moment(mass, CIRCLE_RADIUS, player_shape)
        body = pymunk.Body(mass, moment)
        body.position = _sample_position()
        body.velocity = (rng.uniform(-250, 250), rng.uniform(-150, 150))
        # 2026-09-12: 上記のランダム初速だけだと開始直後に停滞して見えるシードが一定数あった
        # (序盤の情報密度が低いと離脱の主要因になる)ため、中心方向への初期インパルスを追加する。
        # 中心までの実際の距離から逆算した速度にすることで、アリーナサイズが変わっても
        # 「無風ならINITIAL_IMPULSE_TARGET_SECONDS秒程度で中心に到達する」挙動を保つ。
        px, py = body.position
        dx, dy = ARENA_CX - px, ARENA_CY - py
        dist = math.hypot(dx, dy)
        if dist > 1.0:
            impulse_speed = dist / INITIAL_IMPULSE_TARGET_SECONDS
            body.velocity += (dx / dist * impulse_speed, dy / dist * impulse_speed)
        # 2026-09-08: 初期スピンはtriangleだけ(circleは見た目上スピンが分からないため元々0、
        # squareは重力なし直進の入りをまっすぐにするため0からスタートするが、慣性モーメントは
        # 通常通りなので衝突等の外力を受ければそこから回転できる)
        body.angular_velocity = rng.uniform(-2.0, 2.0) if player_shape == "triangle" else 0.0
        if player_shape == "square":
            body.velocity_func = _no_gravity_velocity_func
        shape = _make_player_shape(body, CIRCLE_RADIUS, player_shape)
        shape.friction = friction
        shape.elasticity = elasticity
        shape.collision_type = 2
        space.add(body, shape)
        # 2026-09-21、match_type="team"専用: 「同じ色で2〜3チーム」というユーザー指示通り、
        # チーム戦では色(=能力も連動)をエンティティ個別ではなくチーム単位で割り当てる
        # (team_id自体は色とは独立したフィールドなので、フレンドリーファイア判定
        # (_is_teammate)は色に依存しない。色を揃えるのはあくまで視覚的にチームを
        # 判別しやすくするための演出)。
        if match_type == "team":
            team_id = i % team_count
            color_index = team_id
        else:
            team_id = None
            color_index = i
        entity = CircleEntity(
            id=i, body=body, shape=shape, color=palette_colors[color_index % len(palette_colors)],
            radius=CIRCLE_RADIUS, player_shape=player_shape, team_id=team_id, color_index=color_index % len(COLORS),
        )
        entity.ability = ABILITY_BY_COLOR_INDEX.get(color_index % len(COLORS))
        if entity.ability == "teleport" and rule == "goal_reach":
            # 2026-09-08、ユーザー指示: ゴール到達型のルールにワープ系スキルは禁止
            # (ゴール直前でのワープが到達判定と相性が悪い/ずるく見えるため)。
            # 色↔能力の対応は他ルールと常に一致させる方針(5章)のため、他の能力に差し替えず
            # 「このルールでは無効」として単純に発動しないようにする。
            entity.ability = None
        if entity.ability:
            entity.ability_timer = rng.uniform(1.5, ABILITY_BASE_INTERVAL)

        # 2026-09-21、match_type="boss"専用: 先頭(id=0)を「ふたまわり大きい」「有利ステータス」
        # 「2種類のスキル持ち」のボスにする。他のプレイヤーには一切手を加えない。
        if match_type == "boss" and i == 0:
            entity.is_boss = True
            # 2026-09-21、ユーザー指示「カラーパレットの記号化」対応: 「青=巨大・遅い」の
            # 役割をボスと一致させ、色の意味を統一する(ボスは常に青系の見た目になる)。
            entity.color_index = 1
            entity.color = palette_colors[1 % len(palette_colors)]
            if boss_abilities:
                entity.boss_abilities = list(boss_abilities)
                entity.ability = entity.boss_abilities[0]
                if entity.ability_timer <= 0:
                    entity.ability_timer = rng.uniform(1.5, ABILITY_BASE_INTERVAL)
            if rule == "gun_duel":
                entity.bullet_hits_remaining = BOSS_BULLET_HITS_TO_ELIMINATE
            if rule == "weapon_colosseum":
                entity.hp = int(WEAPON_STARTING_HP * BOSS_HP_MULT)
            _resize_entity(space, entity, CIRCLE_RADIUS * boss_size_mult, elasticity, friction)

        # 2026-09-21、ユーザー指示「カラーパレットの記号化」対応: 色ごとの物理特性(初速)を
        # 開幕時に適用する。sizeはボス以外にのみ適用する(ボスは上のboss_size_multで
        # 既に「ふたまわり大きい」を実現済みのため、二重に拡大しない)。
        traits = COLOR_TRAIT_MODIFIERS.get(entity.color_index, {})
        speed_mult = traits.get("speed_mult")
        if speed_mult and speed_mult != 1.0:
            vx, vy = entity.body.velocity
            entity.body.velocity = (vx * speed_mult, vy * speed_mult)
        if not entity.is_boss:
            size_mult = traits.get("size_mult")
            if size_mult and size_mult != 1.0:
                _resize_entity(space, entity, entity.radius * size_mult, elasticity, friction)

        circles.append(entity)

    # 2026-09-16、冒頭フック強化(ユーザー指示): 全員の位置が出揃った後、各プレイヤーへ
    # 「最も近い他プレイヤー」方向への追加インパルスを与える。両者が同時に相手へ向かうため
    # 実際の最初の衝突はOPENING_COLLISION_TARGET_SECONDSより早く起こる。1人だけの場合は対象なし。
    positions = [(e.body.position.x, e.body.position.y) for e in circles]
    for i, entity in enumerate(circles):
        px, py = positions[i]
        nearest_dist = None
        nearest_dx = nearest_dy = 0.0
        for j, (ox, oy) in enumerate(positions):
            if j == i:
                continue
            ddx, ddy = ox - px, oy - py
            d = math.hypot(ddx, ddy)
            if nearest_dist is None or d < nearest_dist:
                nearest_dist, nearest_dx, nearest_dy = d, ddx, ddy
        if nearest_dist and nearest_dist > 1.0:
            impulse_speed = nearest_dist / OPENING_COLLISION_TARGET_SECONDS
            entity.body.velocity += (nearest_dx / nearest_dist * impulse_speed, nearest_dy / nearest_dist * impulse_speed)
    return circles


def _zone_radius(t: float, zone_start_radius: float = ZONE_START_RADIUS) -> float:
    if t >= ZONE_SHRINK_SECONDS:
        return ZONE_END_RADIUS
    frac = t / ZONE_SHRINK_SECONDS
    return zone_start_radius - (zone_start_radius - ZONE_END_RADIUS) * frac


def _is_teammate(a: "CircleEntity", b: "CircleEntity") -> bool:
    """2026-09-21、チーム戦(match_type="team")のフレンドリーファイア防止用。
    team_idがNone(個人戦・ボス戦)の場合は誰とチームを組んでいないため常にFalseを返す
    (=誰もが攻撃対象になりうる、既存の個人戦の挙動をそのまま維持する)。"""
    return a.team_id is not None and a.team_id == b.team_id


def _resize_entity(space: pymunk.Space, entity: "CircleEntity", new_radius: float, elasticity: float, friction: float) -> None:
    """エンティティの半径を差し替える(shapeの作り直し+質量/慣性モーメントの再計算)。
    absorb_growthの吸収成長・growth_surgeの一時巨大化の両方で使う共通ロジック。
    player_shapeがsquare/triangleの場合はPoly形状を頂点から作り直す。"""
    space.remove(entity.shape)
    new_shape = _make_player_shape(entity.body, new_radius, entity.player_shape)
    new_shape.friction = friction
    new_shape.elasticity = elasticity
    new_shape.collision_type = 2
    space.add(new_shape)
    entity.shape = new_shape
    new_mass = entity.body.mass * (new_radius / entity.radius) ** 2
    entity.body.mass = new_mass
    entity.body.moment = _player_moment(new_mass, new_radius, entity.player_shape)
    entity.radius = new_radius


def _find_teleport_spot(
    circles: list,
    self_entity: "CircleEntity",
    rng: random.Random,
    half_x: float = ARENA_HALF_EXTENT,
    half_y: float = ARENA_HALF_EXTENT,
    arena_shape: str = "square",
    terrain: str | None = None,
) -> tuple | None:
    """2026-09-09、ユーザー指摘対応: 以前は矩形範囲でしかサンプリングしておらず、
    donutの中心障害物に重なる/hourglassの非矩形な境界の外側(到達不可能な場所)に
    テレポートし、そのまま「枠の外に出て脱落する」もったいない結果になることがあった。
    _in_arena_bounds/_overlaps_terrain_obstaclesで実際に到達可能な位置だけに絞り込む。"""
    margin = self_entity.radius + 20
    terrain_chains, terrain_circles = _terrain_obstacles(terrain, half_x, half_y)
    terrain_segments = _chains_to_segments(terrain_chains)
    terrain_margin = self_entity.radius + WALL_THICKNESS / 2 + 15 * _ARENA_SCALE
    for _ in range(24):
        x = rng.uniform(ARENA_CX - half_x + margin, ARENA_CX + half_x - margin)
        y = rng.uniform(ARENA_CY - half_y + margin, ARENA_CY + half_y - margin)
        if not _in_arena_bounds(x, y, arena_shape, terrain, half_x, half_y, margin):
            continue
        if _overlaps_terrain_obstacles(x, y, terrain_segments, terrain_circles, terrain_margin):
            continue
        ok = True
        for o in circles:
            if not o.alive or o.id == self_entity.id:
                continue
            ox, oy = o.body.position
            if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < (self_entity.radius + o.radius + 15):
                ok = False
                break
        if ok:
            return (x, y)
    return None


def _distance_to_nearest_wall(x: float, y: float, arena_shape: str, half_x: float, half_y: float) -> float:
    """weapon_colosseum(槍)専用: 対象から最も近い外壁までの距離。「壁に押し付けている」判定
    (SPEAR_WALL_MARGIN以内かどうか)に使う。circle枠は中心からの半径ベース、square枠は
    上下左右4辺の最短距離で近似する(枠の回転・内部terrainの壁は無視した簡易判定。
    弾/矢の場外判定と同程度の精度感で十分なため)。"""
    if arena_shape == "circle":
        dist_from_center = math.hypot(x - ARENA_CX, y - ARENA_CY)
        return max(0.0, max(half_x, half_y) - dist_from_center)
    dx = half_x - abs(x - ARENA_CX)
    dy = half_y - abs(y - ARENA_CY)
    return max(0.0, min(dx, dy))


def _weapon_blade_position(c: "CircleEntity", kind: str, frame_idx: int) -> tuple[float, float]:
    """weapon_colosseum専用: 剣/ハンマー(常時回転)・斧(回転中)の「武器が実際に描画されている
    位置」を物理空間で計算する(_draw_held_weaponの周回計算と同じ式)。
    2026-09-21、ユーザー指摘対応: 「当たり判定がプレイヤー本体についているのでは」という
    指摘の通り、これらの武器の当たり判定はプレイヤー中心からの距離で行っていたため、
    見た目上は武器が反対側にある時でも近くにいるだけで命中してしまっていた。武器の実位置を
    起点に判定し直すことで、見た目と当たり判定を一致させる。"""
    cx, cy = c.body.position
    if kind == "sword":
        orbit_r = c.radius * WEAPON_SPIN_ORBIT_MULT["sword"]
        spin_angle = frame_idx * DT * SWORD_SPIN_SPEED
    elif kind == "hammer":
        orbit_r = c.radius * WEAPON_SPIN_ORBIT_MULT["hammer"]
        spin_angle = frame_idx * DT * SWORD_SPIN_SPEED
    elif kind == "axe":
        orbit_r = c.radius * AXE_HELD_OFFSET_MULT
        spin_angle = frame_idx * DT * AXE_SPIN_SPEED
    else:
        return cx, cy
    return cx + math.cos(spin_angle) * orbit_r, cy + math.sin(spin_angle) * orbit_r


def _random_food_position(
    rng: random.Random, arena_shape: str, half_x: float = ARENA_HALF_EXTENT, half_y: float = ARENA_HALF_EXTENT
) -> tuple[float, float]:
    """absorb_growth専用: 食べ物のランダムなスポーン位置を、枠の内側(真円/正方形)に収まるように選ぶ。
    ロジックはspawn_circlesの_sample_positionと同じ(circle形状での境界外はみ出し対策の再抽選)。"""
    margin = FOOD_PICKUP_RADIUS + 30
    layout = _arena_layout(half_x, half_y)
    x_lo, x_hi = layout["left"] + margin, layout["right"] - margin
    y_lo, y_hi = layout["top"] + margin, layout["bottom"] - margin
    mx, my = half_x - margin, half_y - margin
    x = y = 0.0
    for _ in range(30):
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        if arena_shape != "circle" or (((x - ARENA_CX) / mx) ** 2 + ((y - ARENA_CY) / my) ** 2) <= 1.0:
            return x, y
    angle = math.atan2((y - ARENA_CY) / my, (x - ARENA_CX) / mx)
    return ARENA_CX + math.cos(angle) * mx * 0.9, ARENA_CY + math.sin(angle) * my * 0.9


def _two_tier_weapon_position(rng: random.Random, half_x: float, half_y: float) -> tuple[float, float]:
    """two_tier専用: 銃を山型足場の斜面上、低い端(枝分かれの先)寄りに配置する
    (2026-09-20、ユーザー指示「枝分かれの先で武器を取れるように」。三角形化に伴い、
    x座標だけでなくy座標も斜面(頂点→低い端の線形補間)に沿わせる。
    ※weapon_colosseum(新ルール、未実装)導入時は「隙間(GAP_FRAC分の空間)からも武器を
    入手できる・全プレイヤーが武器を持てる」仕様が別途必要になる(ユーザー指示、2026-09-20)。
    現状はgun_duel専用(銃1丁の奪い合い)のためこの関数はそのままで良いが、weapon_colosseum
    実装時はルール分岐で複数武器スポーン版を追加すること。"""
    cx, edge_y, hl, peak_y = rng.choice(_two_tier_platforms(half_x, half_y))
    edge_sign = rng.choice((-1, 1))
    frac = 0.85  # 頂点から端までの85%地点(枝分かれの先寄り)
    x = ARENA_CX + cx + edge_sign * hl * frac
    slope_y = peak_y + frac * (edge_y - peak_y)
    return x, ARENA_CY + slope_y - CIRCLE_RADIUS * 0.6


def _split_horizontal_weapon_position(
    rng: random.Random, half_x: float, half_y: float, circles: list | None = None
) -> tuple[float, float]:
    """split_horizontal専用: 銃を上下どちらかの陣地の中に配置する(仕切りを挟んだ
    中央の隙間には置かない。隙間は物理的にプレイヤーが到達できない領域のため)。
    2026-09-20、実測でundecided=6/20(30%)が判明したため2段階で修正:
    (1) プレイヤーは仕切りを越えられないため、生存者が0人の陣地に銃が出ると誰も
        拾えず永久にスタックしていた→生存者がいる陣地のみを抽選対象にする
        (片方しか生存者がいなければそちらに固定。spawn後の入れ替わりで無人化する
        ケースは、呼び出し側(simulate内)の毎フレーム再判定で別途対応している)。
    (2) それでも約10%stallが残存: 重力で各陣地内の「床」側(上段陣地なら仕切り際、
        下段陣地なら外壁際)に滞在時間が偏るため、y座標を陣地内で一様分布にすると
        「天井」寄りの位置に銃が出た場合に長時間(90秒上限内でも)誰も通りかからない
        ケースがあった→y座標を「床」側に寄せてサンプリングする。"""
    divider_y = half_y * SPLIT_HORIZONTAL_DIVIDER_FRAC
    margin = FOOD_PICKUP_RADIUS + 30
    x = rng.uniform(ARENA_CX - half_x + margin, ARENA_CX + half_x - margin)
    top_occupied = bottom_occupied = True
    if circles is not None:
        alive_ys = [c.body.position[1] for c in circles if c.alive]
        if alive_ys:
            top_occupied = any(y < ARENA_CY for y in alive_ys)
            bottom_occupied = any(y >= ARENA_CY for y in alive_ys)
    use_top = (rng.random() < 0.5) if (top_occupied and bottom_occupied) else top_occupied
    # どちらの陣地も「床」はy値が大きい側(上段陣地は仕切り際、下段陣地は外壁際)。
    # frac**0.4で床寄りに偏らせる(0.4は実測でstall率が十分下がる値として選定)。
    frac = rng.random() ** 0.4
    if use_top:
        y_lo, y_hi = ARENA_CY - half_y + margin, ARENA_CY - divider_y - margin
    else:
        y_lo, y_hi = ARENA_CY + divider_y + margin, ARENA_CY + half_y - margin
    y = y_lo + frac * (y_hi - y_lo)
    return x, y


def _poison_wire_position(
    rule: str,
    hole_width: float,
    half_x: float,
    half_y: float,
    arena_shape: str,
    terrain: str | None,
    rng: random.Random,
) -> tuple:
    """「有利になりそうな位置」の簡易ヒューリスティック: 出口があればその近く、
    密閉ステージなら中心付近に置く。

    2026-09-09、ユーザー指摘対応: donutのように中心が障害物で埋まっているterrainだと、
    中心付近に置いたワイヤーがそのまま到達不可能で「発動の意味がない」結果になっていた。
    中心が塞がっている場合は、実際に到達可能な位置をランダムに探す。ワイヤーは中心から
    左右に±WIRE_HALF_LENほど伸びるため、中心点だけでなく想定される全長ぶんの余白を見て
    判定する(左右の端が障害物に刺さるのを防ぐ)。"""
    if rule == "hole_fall" and hole_width > 0:
        return (ARENA_CX, ARENA_CY + half_y - 90 * _ARENA_SCALE)
    wire_half_len = 70.0  # 密閉ステージでの既定の半長(呼び出し側のhalf_lenと同じ値)
    terrain_chains, terrain_circles = _terrain_obstacles(terrain, half_x, half_y)
    if not terrain_circles and terrain != "hourglass":
        return (ARENA_CX, ARENA_CY)
    terrain_segments = _chains_to_segments(terrain_chains)
    margin = wire_half_len + 20 * _ARENA_SCALE
    for _ in range(20):
        x = rng.uniform(ARENA_CX - half_x * 0.7, ARENA_CX + half_x * 0.7)
        y = rng.uniform(ARENA_CY - half_y * 0.7, ARENA_CY + half_y * 0.7)
        if not _in_arena_bounds(x, y, arena_shape, terrain, half_x, half_y, margin):
            continue
        if _overlaps_terrain_obstacles(x, y, terrain_segments, terrain_circles, margin):
            continue
        return (x, y)
    return (ARENA_CX, ARENA_CY)  # 見つからなければ従来通り中心(最悪でも無干渉なだけで実害はない)


def _ability_param(ability_params: dict | None, ability: str, key: str, default: float) -> float:
    """ability_paramsで上書きされていればその値を、無ければ既定の定数を返す。
    新規アイデア生成(geometry_battle_idea_generator.py)が「既存の特殊能力の効き目だけを変えた
    新しいバリアント」を安全に作れるようにするための差し込み口(物理演算コード自体は変えない)。"""
    if not ability_params:
        return default
    return ability_params.get(ability, {}).get(key, default)


def simulate(
    n_circles: int,
    seed: int,
    gravity: float = DEFAULT_GRAVITY,
    elasticity: float = DEFAULT_ELASTICITY,
    rotation_speed: float = DEFAULT_ROTATION_SPEED,
    hole_width: float = DEFAULT_HOLE_WIDTH,
    damping: float = DEFAULT_DAMPING,
    shape: str = DEFAULT_SHAPE,
    rule: str = DEFAULT_RULE,
    friction: float = DEFAULT_FRICTION,
    accel_zone: bool | None = None,
    trap: bool = False,
    player_shape: str = DEFAULT_PLAYER_SHAPE,
    palette: str = DEFAULT_PALETTE,
    ability_params: dict[str, dict[str, float]] | None = None,
    wind_force: tuple[float, float] = (0.0, 0.0),
    arena_size: str = "compact",
    terrain: str | None = None,
    match_type: str = "individual",
    team_count: int = 2,
    boss_abilities: tuple[str, str] | None = None,
):
    """軽量シミュレーション: 描画せず、毎フレームの位置・半径・角度・枠の角度・
    衝突/脱落/特殊能力イベントだけをログに記録する。

    match_type: "individual"(既定・従来通り)|"team"|"boss"。2026-09-21追加、既存のrule
    (勝敗条件のロジック)とは独立した対戦形式の軸。"team"はhole_fall/goal_reach/
    area_control/gun_duel/weapon_colosseumの5ルール、"boss"はweapon_colosseum/gun_duelの
    2ルールにのみ対応させる方針(absorb_growthはゲーム性上チーム制と相性が悪いため対象外)。
    呼び出し側(daily_pipeline.py)がこの組み合わせを守る前提で、simulate()自体は
    どのrule×match_typeの組み合わせで呼ばれても物理的にクラッシュしないようにしてある。
    accel_zone=Noneの場合、goal_reachでは自動的に有効化する(重力に逆らってゴールへ
    届くための「発射台」として補助しないと難易度が高すぎるため)。
    player_shape: "circle"|"square"|"triangle" — プレイヤー本体の形状(枠の形状=shapeとは別軸)
    ability_params: {ability名: {パラメータ名: 値}}で各特殊能力の効き目を上書きする
    (「新しい能力」ではなく既存能力の強さ違いバリアントを安全に作るための差し込み口)。
    wind_force: 全プレイヤーに毎フレーム一定に加わる水平方向中心の外力(重力とは別軸)。
    3章「物理パラメータ」の「風等の外力」に対応(2026-09-09実装)。
    arena_size: "compact"(既定)|"spacious"|"tall"。ARENA_SIZE_VARIANTS参照。ステージのバリエーションを
    増やす/画面をもっと使いたいというユーザー要望への対応(2026-09-08)。等方(縦横比1:1)かつ
    compactの半径以下でのみ回転が安全なため、それ以外(非等方のtall、またはcompactより大きい
    等方サイズ)では下で自動的に回転を止める。
    """
    if accel_zone is None:
        accel_zone = rule == "goal_reach"

    if rule == "weapon_colosseum":
        # 2026-09-20、ユーザー指示: 「脱落のほとんどが落下による場外なので、Weapon Colosseumの
        # 場合は穴の空いてないステージでやってください」。呼び出し元が何を渡しても関係なく、
        # このルールでは常に密閉する(gun_duelの「場外に出ない枠の中で戦う」と同じ思想)。
        hole_width = 0.0

    size_x_mult, size_y_mult = ARENA_SIZE_VARIANTS.get(arena_size, (1.0, 1.0))
    half_x = ARENA_HALF_EXTENT * size_x_mult
    half_y = ARENA_HALF_EXTENT * size_y_mult
    is_isotropic = abs(half_x - half_y) < 1e-6
    rotation_safe = is_isotropic and (shape == "circle" or half_x <= ARENA_HALF_EXTENT + 1e-6)
    if not rotation_safe:
        rotation_speed = 0.0
    if terrain == "hourglass":
        # hourglassは縦方向にHOURGLASS_HEIGHT_RATIO倍拡大した独自形状(_hourglass_boundary_chain)
        # のため、half_x/half_y基準の既存の回転安全性判定はそのままでは適用できない。安全側に倒し
        # 常に回転なしにする(見た目的にも「砂時計は縦に立っている」ほうが自然)。
        rotation_speed = 0.0
    layout = _arena_layout(half_x, half_y)
    # hourglass専用: 「画面外に出た」の判定(ESCAPE_MARGIN)に使う実効half_y。壁自体が
    # half_yのHOURGLASS_HEIGHT_RATIO倍まで拡張されているため、判定もそれに合わせないと
    # 実際にはまだ壁の内側にいるプレイヤーを誤って脱落させてしまう。
    escape_half_x = half_x
    escape_half_y = half_y * HOURGLASS_HEIGHT_RATIO if terrain == "hourglass" else half_y
    # area_control専用: 保護ゾーンのスポーン位置選びで使う(terrain対応、下記参照)
    pz_terrain_chains, pz_terrain_circles = _terrain_obstacles(terrain, half_x, half_y)
    pz_terrain_segments = _chains_to_segments(pz_terrain_chains)

    space, arena_body = build_space(
        gravity, elasticity, rotation_speed, hole_width, damping, shape, friction, half_x, half_y, terrain
    )
    # goal_reachはゴールが上方にあるため、初期スポーンがゴール圏内に重ならないよう下寄りにする
    # (2026-09-09: 枠の真円/正方形化に伴いARENA_HALF_EXTENT基準に変更。
    # 2026-09-08: ステージサイズのバリエーション対応によりhalf_x/half_y基準に変更)
    if rule == "goal_reach":
        spawn_y_range = (ARENA_CY + half_y * 0.15, layout["bottom"] - CIRCLE_RADIUS - 20)
    elif rule == "absorb_growth":
        # 密集スポーンだと開始直後に吸収が連鎖して尺が極端に短くなるため、
        # 全高近くまで散らして初期の接触機会を減らす
        spawn_y_range = (layout["top"] + CIRCLE_RADIUS + 20, layout["bottom"] - CIRCLE_RADIUS - 20)
    elif terrain == "pinball":
        # 2026-09-13、指示書対応: 開始直後に必ずポップバンパー群へ突入する「強制ファースト・
        # インパクト」を作るため、スポーンを最も上にあるバンパー(上段の小型バンパー、
        # PINBALL_UPPER_BUMPER_Y_RATIO)よりさらに上のステージ最上段だけに限定する。
        bumper_top_world_y = ARENA_CY + half_y * PINBALL_UPPER_BUMPER_Y_RATIO
        spawn_y_range = (
            layout["top"] + CIRCLE_RADIUS + 20,
            bumper_top_world_y - PINBALL_UPPER_BUMPER_RADIUS - CIRCLE_RADIUS - 10 * _ARENA_SCALE,
        )
    else:
        spawn_y_range = None
    circles = spawn_circles(
        space,
        n_circles,
        seed,
        elasticity,
        friction,
        spawn_y_range,
        player_shape,
        palette,
        arena_shape=shape,
        half_x=half_x,
        half_y=half_y,
        rule=rule,
        terrain=terrain,
        match_type=match_type,
        team_count=team_count,
        boss_abilities=boss_abilities,
    )
    body_to_entity = {id(c.body): c for c in circles}
    entity_by_id = {c.id: c for c in circles}  # weapon_colosseum専用: 矢の命中時に射手エンティティを引くために使う

    def _weapon_eliminate(target: CircleEntity, at_frame: int) -> None:
        """weapon_colosseum専用: HPが尽きた相手を脱落させる(gun_duelの弾ヒット処理と同じ
        パターン。collisionコールバックの外、通常のフレーム処理中なのでspaceを直接操作できる)。"""
        target.alive = False
        target.eliminated_frame = at_frame
        tx, ty = target.body.position
        result.elimination_order.append({"id": target.id, "frame": at_frame, "x": tx, "y": ty, "radius": target.radius, "cause": "weapon"})
        space.remove(target.body, target.shape)
        body_to_entity.pop(id(target.body), None)

    def _trigger_hitstop(entity: CircleEntity, at_frame: int) -> None:
        """weapon_colosseum専用: 現在の速度を保存してから速度をゼロにする(=ヒットストップ開始)。
        保存した速度はHITSTOP_FRAMES後、メインループの専用ブロックで重力分を足し戻した上で
        復元する(_apply_weapon_damageのコメント参照。運動量を失わないための肝)。"""
        vx, vy = entity.body.velocity
        entity.hitstop_saved_vx = vx
        entity.hitstop_saved_vy = vy
        entity.body.velocity = (0.0, 0.0)
        entity.hitstop_until_frame = at_frame + HITSTOP_FRAMES

    def _apply_weapon_damage(
        attacker_id: int, target: CircleEntity, dmg: int, at_frame: int, kind: str, target_hitstop: bool = True
    ) -> None:
        """weapon_colosseum専用: 武器のダメージを与え、HPが0以下になったら脱落させる。
        2026-09-21、ユーザー指示による「ヒットストップ」の最終実装: 単に速度を1回ゼロに
        するだけだと、命中のたびに運動量を完全に失い(以後は重力/衝突で速度を一から
        作り直すしかない)、命中回数の多い試合ほど終盤に動きが鈍くなる問題があった。
        命中直前の速度を保存しておき、HITSTOP_FRAMESの間だけゼロで静止させた後、保存して
        いた速度(+その間に本来受けていたはずの重力分)を復元することで、見た目は一瞬
        静止しつつ運動量は失われないようにする(_trigger_hitstop/メインループの復元処理参照)。
        target_hitstop=False(ハンマー/斧などノックバックを伴う武器)の場合は、呼び出し側が
        既にtargetの速度をノックバック方向へ設定済みのため、ここでは上書きしない
        (「ノックバックが無い限り速度を0にする」というユーザー指示に対応)。"""
        target.hp -= dmg
        tx, ty = target.body.position
        result.ability_events.append(
            {"type": "weapon_hit", "id": attacker_id, "target_id": target.id, "frame": at_frame, "x": tx, "y": ty, "phase": kind}
        )
        attacker = entity_by_id.get(attacker_id)
        if attacker is not None and attacker.alive:
            _trigger_hitstop(attacker, at_frame)
        if target.hp <= 0:
            _weapon_eliminate(target, at_frame)
        elif target_hitstop:
            _trigger_hitstop(target, at_frame)
    result = SimResult(
        seed=seed,
        n_circles=n_circles,
        gravity=gravity,
        elasticity=elasticity,
        shape=shape,
        rule=rule,
        rotation_speed=rotation_speed,
        hole_width=hole_width,
        accel_zone=accel_zone,
        trap=trap,
        player_shape=player_shape,
        arena_half_x=half_x,
        arena_half_y=half_y,
        terrain=terrain,
        match_type=match_type,
    )

    # goal_reach専用: ゴールをバリアで囲み、何度か体当たりして破壊してからでないと到達できない
    # ようにする(2026-09-08、ユーザー要望)。ゴールが上端の壁に近い(コンパクトなステージだと
    # GOAL_RADIUS*1.6の全周は上の壁に食い込む)ため、半径は縮めずに上端の壁より上に出る
    # セグメントだけを間引く(D字型のリングになる)。プレイヤーは常に下/横から接近するため、
    # 上側が開いていても実質的な抜け道にはならない。
    barrier_state = {"active": False, "hits": 0, "shapes": [], "x": 0.0, "y": 0.0, "radius": 0.0, "break_pending": False}
    if rule == "goal_reach":
        bx, by = layout["goal_point"]
        b_radius = GOAL_RADIUS * BARRIER_RADIUS_MULT
        n_seg = 40
        barrier_shapes = []
        for i in range(n_seg):
            a0 = i * (2 * math.pi / n_seg)
            a1 = (i + 1) * (2 * math.pi / n_seg)
            p0 = (bx + b_radius * math.cos(a0), by + b_radius * math.sin(a0))
            p1 = (bx + b_radius * math.cos(a1), by + b_radius * math.sin(a1))
            if p0[1] < layout["top"] and p1[1] < layout["top"]:
                continue  # 上端の壁より上に出る区間は間引く
            seg = pymunk.Segment(space.static_body, p0, p1, 5)
            seg.friction = friction
            seg.elasticity = elasticity
            seg.collision_type = 7
            space.add(seg)
            barrier_shapes.append(seg)
        barrier_state = {
            "active": True,
            "hits": 0,
            "shapes": barrier_shapes,
            "x": bx,
            "y": by,
            "radius": b_radius,
            "break_pending": False,
        }
        result.goal_barrier = {"x": bx, "y": by, "radius": b_radius, "hits_to_break": BARRIER_HITS_TO_BREAK, "break_frame": None}

    ability_rng = random.Random(seed + 54321)
    food_rng = random.Random(seed + 13579)
    gun_rng = random.Random(seed + 24680)
    shield_rng = random.Random(seed + 97531)
    active_wires: list = []
    pending_growth_reverts: list = []
    active_food: dict | None = None  # absorb_growth専用: {"record": result.food_itemsの要素への参照}
    food_respawn_at_frame = int(FOOD_INITIAL_DELAY_SECONDS * FPS)
    protection_zone_rng = random.Random(seed + 11223)
    active_protection_zone: dict | None = None  # area_control専用: result.protection_zonesの要素への参照
    protection_zone_respawn_at_frame = int(PROTECTION_ZONE_INITIAL_DELAY_SECONDS * FPS)
    active_cages: list = []  # area_control専用: 実体化した保護ゾーンの囲い(pymunk shapesと期限)
    active_gun: dict | None = None  # gun_duel専用
    gun_respawn_at_frame = int(GUN_INITIAL_DELAY_SECONDS * FPS)
    active_shield: dict | None = None  # gun_duel専用
    shield_respawn_at_frame = int(SHIELD_INITIAL_DELAY_SECONDS * FPS)
    active_bullets: list = []  # gun_duel専用: result.bulletsの要素への参照のリスト(飛行中のみ)
    flipper_next_pulse_frame = 0  # pinball専用: 自動パルスフリッパーの次回発動フレーム
    weapon_rng = random.Random(seed + 86420)
    active_weapons: dict = {k: None for k in WEAPON_KINDS}  # weapon_colosseum専用: 種類ごとに1つまで同時出現
    weapon_respawn_at_frame: dict = {k: int(WEAPON_INITIAL_DELAY_SECONDS * FPS) for k in WEAPON_KINDS}
    weapon_ever_spawned: set = set()  # weapon_colosseum専用: 種類ごとに一度出現したら二度と出現させない
    # 2026-09-21、ユーザー指示: 近接武器(剣/槍/ハンマー)のクールタイムは「武器そのもの」では
    # なく「武器と当てた相手の組み合わせ」ごとに管理する(同時に複数の相手を攻撃できるように
    # するため)。キーは(攻撃側id, 対象id)。
    melee_pair_cooldown: dict[tuple[int, int], int] = {}
    active_arrows: list = []  # weapon_colosseum(弓矢)専用: result.arrowsの要素への参照のリスト(飛行中のみ)

    frame_box = {"idx": 0}
    active_pairs: set = set()
    pending_merges: list = []
    pending_partial_trades: list = []  # absorb_growth専用: [(grower, shrinker)]
    queued_loser_ids: set = set()

    def on_begin(arbiter, space, data):
        a, b = arbiter.shapes
        pair = frozenset((id(a), id(b)))
        active_pairs.add(pair)
        if rule == "absorb_growth" and a.collision_type == 2 and b.collision_type == 2:
            ea = body_to_entity.get(id(a.body))
            eb = body_to_entity.get(id(b.body))
            if ea and eb and ea.alive and eb.alive and ea.id not in queued_loser_ids and eb.id not in queued_loser_ids:
                # 2026-09-09、ユーザー指摘対応: 「どちらが吸収するのか説明なしでわかる」ように、
                # ランダムに出現する食べ物(food_items)に触れて一定時間だけ得られる「吸収可能」
                # 状態(empowered_until_frame)を吸収の必須条件にした。どちらも吸収可能でない
                # 組み合わせはただ跳ね返るだけ(合体しない、通常の衝突として扱う)。
                cur_frame = frame_box["idx"]
                ea_can_eat = ea.empowered_until_frame > cur_frame
                eb_can_eat = eb.empowered_until_frame > cur_frame
                if ea_can_eat or eb_can_eat:
                    if ea_can_eat and eb_can_eat:
                        # 両者とも吸収可能な場合のみ、従来のタイプ相性(5章)+実効半径で決める
                        ea_color, eb_color = ea.id % len(COLORS), eb.id % len(COLORS)
                        ea_effective = ea.radius * type_matchup_multiplier(ea_color, eb_color)
                        eb_effective = eb.radius * type_matchup_multiplier(eb_color, ea_color)
                        attacker, defender = (
                            (ea, eb) if ea_effective > eb_effective or (ea_effective == eb_effective and ea.id < eb.id) else (eb, ea)
                        )
                    else:
                        attacker, defender = (ea, eb) if ea_can_eat else (eb, ea)

                    # 2026-09-08、ユーザー指摘対応(早期決着が多いための難易度調整):
                    # 自分より大きい相手は吸収しきれない。その場合は全滅させず、自分を一段階
                    # 大きく・相手を一段階小さくする「かじり取り」に留める(pending_partial_trades)。
                    if attacker.radius >= defender.radius:
                        pending_merges.append((attacker, defender))
                        queued_loser_ids.add(defender.id)
                        # 吸収可能な状態は1体吸収した時点で解除する(連続吸収による瞬殺を防ぐ)
                        attacker.empowered_until_frame = cur_frame
                    else:
                        pending_partial_trades.append((attacker, defender))

        if terrain == "pin_wall" and {a.collision_type, b.collision_type} == {2, 9}:
            # 2026-09-13、pin_wall新terrain: ピン(collision_type=9)に触れた瞬間、そのピンの
            # 位置でability_events経由の汎用リングフラッシュ+専用ポップ音(ABILITY_EFFECT_COLOR/
            # _synth_for_ability参照)を鳴らす。反発自体はpin_shape.elasticity(>1.0)に任せており、
            # ここでは速度を書き換えない(ライブ感の演出はpinballのような能動キックではなく、
            # 指示書通りピン自体の反発係数の高さで表現する。安全弁は速度クランプ側で担保する)。
            player_shape, pin_shape = (a, b) if a.collision_type == 2 else (b, a)
            entity = body_to_entity.get(id(player_shape.body))
            pin_center = pin_shape.body.local_to_world(pin_shape.offset)
            result.ability_events.append(
                {
                    "type": "pin_hit",
                    "id": entity.id if entity is not None else -1,
                    "frame": frame_box["idx"],
                    "x": pin_center.x,
                    "y": pin_center.y,
                }
            )

        if terrain == "pinball" and {a.collision_type, b.collision_type} == {2, 8}:
            # 2026-09-13、「本格ピンボール型ステージ」指示書対応: 弾き飛ばし自体はバンパーの
            # elasticity(PINBALL_BUMPER_RESTITUTION、build_space参照)に任せ、ここでは速度を
            # 書き換えない。ability_events経由で汎用の_draw_ability_effectsがヒット位置に
            # リング状のフラッシュ+専用ポップ音(ABILITY_EFFECT_COLOR/_synth_for_ability
            # 参照)を鳴らし、加えてbumper_x/bumper_yでどのバンパーが光ったかを_draw_terrainに
            # 伝え、指示書指定の「一瞬1.2倍に拡縮」を該当バンパーだけに適用する。
            player_shape, bumper_shape = (a, b) if a.collision_type == 2 else (b, a)
            body = player_shape.body
            bumper_center = bumper_shape.body.local_to_world(bumper_shape.offset)
            entity = body_to_entity.get(id(body))
            result.ability_events.append(
                {
                    "type": "pinball_kick",
                    "id": entity.id if entity is not None else -1,
                    "frame": frame_box["idx"],
                    "x": bumper_center.x,
                    "y": bumper_center.y,
                    "bumper_x": bumper_shape.offset.x,
                    "bumper_y": bumper_shape.offset.y,
                }
            )

        if barrier_state["active"] and {a.collision_type, b.collision_type} == {2, 7}:
            # 2026-09-08、ユーザー要望: ゴールをバリアで囲み、既定回数ぶつかるまで到達できない
            # ようにする。space変更(除去)はここではできないため、破壊判定だけ立てて
            # 実際の除去はstep完了後(下のbarrier_state["break_pending"]処理)で行う。
            cur_frame = frame_box["idx"]
            barrier_state["hits"] += 1
            result.ability_events.append(
                {"type": "barrier_hit", "id": -1, "frame": cur_frame, "x": barrier_state["x"], "y": barrier_state["y"]}
            )
            if barrier_state["hits"] >= BARRIER_HITS_TO_BREAK:
                barrier_state["break_pending"] = True
        return True

    def on_post_solve(arbiter, space, data):
        a, b = arbiter.shapes
        pair = frozenset((id(a), id(b)))
        if pair in active_pairs:
            impulse = arbiter.total_impulse.length
            if impulse > 1.0:
                result.collisions.append({"frame": frame_box["idx"], "impulse": impulse})
            active_pairs.discard(pair)

    def on_separate(arbiter, space, data):
        a, b = arbiter.shapes
        active_pairs.discard(frozenset((id(a), id(b))))

    space.on_collision(begin=on_begin, post_solve=on_post_solve, separate=on_separate)

    max_frames = int(MAX_SIM_SECONDS * FPS)
    for frame_idx in range(max_frames):
        frame_box["idx"] = frame_idx
        space.step(DT)
        result.arena_angles.append(arena_body.angle)

        if terrain == "pin_wall":
            # PIN_WALL_RESTITUTION(>1.0)により衝突の度に運動エネルギーが増え得るため、
            # PIN_WALL_MAX_SPEED定義部のコメントの通り毎フレーム上限でクランプする安全弁。
            for c in circles:
                if not c.alive:
                    continue
                speed = c.body.velocity.length
                if speed > PIN_WALL_MAX_SPEED:
                    c.body.velocity = c.body.velocity * (PIN_WALL_MAX_SPEED / speed)

        if terrain == "pinball":
            # 2026-09-13、ユーザー指摘対応: バンパーを増やすのではなく、上部バンパー帯
            # (PINBALL_FLOAT_ZONE_TOP_Y_RATIO〜BOTTOM_Y_RATIO)にいる間だけ重力の一部を
            # 打ち消し、自然落下より滞空時間を延ばすことで接触機会を増やす。space.gravityは
            # space全体に一様にかかるため、このフレームで既に加算された分の一部
            # (gravity*(1-scale)*DT)を差し引くことで、ゾーン内だけ実効重力を弱める。
            float_zone_top = ARENA_CY + half_y * PINBALL_FLOAT_ZONE_TOP_Y_RATIO
            float_zone_bottom = ARENA_CY + half_y * PINBALL_FLOAT_ZONE_BOTTOM_Y_RATIO
            for c in circles:
                if not c.alive:
                    continue
                cx, cy = c.body.position
                if float_zone_top <= cy <= float_zone_bottom:
                    vx, vy = c.body.velocity
                    c.body.velocity = (vx, vy - gravity * (1.0 - PINBALL_FLOAT_GRAVITY_SCALE) * DT)

            # バンパー(PINBALL_BUMPER_RESTITUTION)・スリングショット(SLINGSHOT_RESTITUTION)
            # とも反発係数>1.0で運用しているため、pin_wallと同じ理由・同じ仕組みで
            # PINBALL_MAX_SPEEDによる毎フレームの速度クランプを安全弁として設ける。
            for c in circles:
                if not c.alive:
                    continue
                speed = c.body.velocity.length
                if speed > PINBALL_MAX_SPEED:
                    c.body.velocity = c.body.velocity * (PINBALL_MAX_SPEED / speed)

            # 指示書④「自動パルスフリッパー」対応(2026-09-13): 穴の手前の帯状ゾーンに
            # いる全プレイヤーへ、FLIPPER_PULSE_INTERVAL_SECONDSごとに一斉に上向きのキックを
            # 与える(実際のフリッパー形状は実装せず、指示書の代替案「周期的なキック力」を採用。
            # 理由はFLIPPER_ZONE_HALF_WIDTH等の定義部のコメント参照)。
            if frame_idx >= flipper_next_pulse_frame:
                zone_top = layout["bottom"] - FLIPPER_ZONE_HEIGHT
                for c in circles:
                    if not c.alive:
                        continue
                    cx, cy = c.body.position
                    if abs(cx - ARENA_CX) <= FLIPPER_ZONE_HALF_WIDTH and zone_top <= cy <= layout["bottom"]:
                        vx, _vy = c.body.velocity
                        c.body.velocity = (vx, -FLIPPER_KICK_SPEED)
                        result.ability_events.append(
                            {"type": "flipper_kick", "id": c.id, "frame": frame_idx, "x": cx, "y": cy}
                        )
                flipper_next_pulse_frame = frame_idx + int(FLIPPER_PULSE_INTERVAL_SECONDS * FPS)

        # 加速ゾーン: 範囲内にいるプレイヤーへ毎フレーム力を加える(連続的な力を
        # 速度への増分として近似)。goal_reachでは重力に逆らってゴールへ届くための
        # 「発射台」として使う
        if accel_zone:
            x0, y0, x1, y1 = layout["accel_zone_rect"]
            fx, fy = ACCEL_ZONE_FORCE
            for c in circles:
                if not c.alive:
                    continue
                cx, cy = c.body.position
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    c.body.velocity = (c.body.velocity[0] + fx * DT, c.body.velocity[1] + fy * DT)

        # 風(外力): 全プレイヤーへ毎フレーム一定に加わる力。加速ゾーンと違い範囲を問わない
        wfx, wfy = wind_force
        if wfx or wfy:
            for c in circles:
                if c.alive:
                    c.body.velocity = (c.body.velocity[0] + wfx * DT, c.body.velocity[1] + wfy * DT)

        # absorb_growthのマージ処理。collisionコールバック中はspaceを変更できないため、
        # step完了後のこのタイミングでshape差し替え・body除去を行う。
        if pending_merges:
            for winner, loser in pending_merges:
                if not loser.alive or not winner.alive:
                    continue
                new_radius = (winner.radius ** 2 + loser.radius ** 2) ** 0.5
                _resize_entity(space, winner, new_radius, elasticity, friction)

                loser.alive = False
                loser.eliminated_frame = frame_idx
                lx, ly = loser.body.position
                result.elimination_order.append({"id": loser.id, "frame": frame_idx, "x": lx, "y": ly, "radius": loser.radius, "cause": "absorbed"})
                space.remove(loser.body, loser.shape)
                body_to_entity.pop(id(loser.body), None)
            pending_merges.clear()
            queued_loser_ids.clear()

        # absorb_growthの「かじり取り」処理(2026-09-08追加): 自分より大きい相手を吸収しきれない
        # 場合の部分的なサイズ交換。全滅は発生しない(_resize_entityのみ、除去はしない)。
        if pending_partial_trades:
            for grower, shrinker in pending_partial_trades:
                if not grower.alive or not shrinker.alive:
                    continue
                gx, gy = grower.body.position
                sx, sy = shrinker.body.position
                new_grower_radius = grower.radius * (1 + ABSORB_PARTIAL_STEP)
                new_shrinker_radius = max(ABSORB_MIN_RADIUS, shrinker.radius * (1 - ABSORB_PARTIAL_STEP))
                _resize_entity(space, grower, new_grower_radius, elasticity, friction)
                _resize_entity(space, shrinker, new_shrinker_radius, elasticity, friction)
                result.ability_events.append({"type": "partial_absorb_grow", "id": grower.id, "frame": frame_idx, "x": gx, "y": gy})
                result.ability_events.append(
                    {"type": "partial_absorb_shrink", "id": shrinker.id, "frame": frame_idx, "x": sx, "y": sy}
                )
            pending_partial_trades.clear()

        # goal_reach専用: ゴールバリアの破壊処理。collisionコールバック中はspaceを変更できないため、
        # step完了後のこのタイミングでshape除去を行う(pending_merges等と同じパターン)。
        if barrier_state["break_pending"]:
            for seg in barrier_state["shapes"]:
                space.remove(seg)
            barrier_state["active"] = False
            barrier_state["shapes"] = []
            barrier_state["break_pending"] = False
            if result.goal_barrier is not None:
                result.goal_barrier["break_frame"] = frame_idx
            result.ability_events.append(
                {"type": "barrier_break", "id": -1, "frame": frame_idx, "x": barrier_state["x"], "y": barrier_state["y"]}
            )

        # gun_duel専用: 銃/シールドの出現・拾得・自動照準・発射・弾の飛行判定。
        if rule == "gun_duel":
            # 銃の出現・拾得
            if active_gun is None and frame_idx >= gun_respawn_at_frame:
                if terrain == "two_tier":
                    gx, gy = _two_tier_weapon_position(gun_rng, half_x, half_y)
                elif terrain == "split_horizontal":
                    gx, gy = _split_horizontal_weapon_position(gun_rng, half_x, half_y, circles)
                elif frame_idx == 0:
                    # 2026-09-21、ユーザー指示「スワイプ離脱(広告誤認・無風)防止」対応:
                    # 開幕直後の初回スポーンだけ中央寄り(半径40%圏内)に絞り、全プレイヤーの
                    # 中央方向への強い初期インパルスと噛み合わせて早期の争奪戦を作る。
                    # 銃を失った後の再出現(2回目以降)は従来通り全域ランダムのままにし、
                    # 中盤以降の展開の多様性は変えない。
                    gx, gy = _random_food_position(gun_rng, shape, half_x * 0.4, half_y * 0.4)
                else:
                    gx, gy = _random_food_position(gun_rng, shape, half_x, half_y)
                gun_record = {"x": gx, "y": gy, "start_frame": frame_idx, "end_frame": None, "picked_by": None}
                result.guns.append(gun_record)
                active_gun = gun_record
            # 2026-09-20、実測でsplit_horizontalが依然約30%stallすることが判明: 銃の出現時点では
            # 到達可能な陣地に置いていても、その後その陣地の生存者が全員脱落すると、銃だけが
            # 誰にも拾えない陣地に取り残されてしまう(プレイヤーは仕切りを越えられないため)。
            # 生存者の入れ替わりに追従して、その都度その場で再配置する。
            if terrain == "split_horizontal" and active_gun is not None:
                alive_ys = [c.body.position[1] for c in circles if c.alive]
                if alive_ys:
                    gun_is_top = active_gun["y"] < ARENA_CY
                    gun_band_occupied = any((y < ARENA_CY) == gun_is_top for y in alive_ys)
                    if not gun_band_occupied:
                        gx, gy = _split_horizontal_weapon_position(gun_rng, half_x, half_y, circles)
                        active_gun["x"], active_gun["y"] = gx, gy
            if active_gun is not None:
                gx, gy = active_gun["x"], active_gun["y"]
                for c in circles:
                    if not c.alive or c.holding_gun:
                        continue
                    cx, cy = c.body.position
                    if ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 <= FOOD_PICKUP_RADIUS + c.radius:
                        c.holding_gun = True
                        c.gun_fire_frame = frame_idx + int(GUN_AIM_SECONDS * FPS)
                        active_gun["end_frame"] = frame_idx
                        active_gun["picked_by"] = c.id
                        active_gun = None
                        # 2026-09-20、ユーザー指示: split_horizontalは通常より銃のスポーン頻度を上げる。
                        respawn_delay = (
                            SPLIT_HORIZONTAL_GUN_RESPAWN_DELAY_SECONDS
                            if terrain == "split_horizontal"
                            else GUN_RESPAWN_DELAY_SECONDS
                        )
                        gun_respawn_at_frame = frame_idx + int(respawn_delay * FPS)
                        break

            # シールドの出現・拾得(銃より低頻度)
            if active_shield is None and frame_idx >= shield_respawn_at_frame:
                sx, sy = _random_food_position(shield_rng, shape, half_x, half_y)
                shield_record = {"x": sx, "y": sy, "start_frame": frame_idx, "end_frame": None, "picked_by": None}
                result.shields.append(shield_record)
                active_shield = shield_record
            if active_shield is not None:
                sx, sy = active_shield["x"], active_shield["y"]
                for c in circles:
                    if not c.alive:
                        continue
                    cx, cy = c.body.position
                    if ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 <= FOOD_PICKUP_RADIUS + c.radius:
                        c.shielded_until_frame = frame_idx + int(SHIELD_DURATION_SECONDS * FPS)
                        active_shield["end_frame"] = frame_idx
                        active_shield["picked_by"] = c.id
                        active_shield = None
                        shield_respawn_at_frame = frame_idx + int(SHIELD_RESPAWN_DELAY_SECONDS * FPS)
                        break

            # 銃を保持中のプレイヤーの自動発射(GUN_AIM_SECONDS経過)。保持者が発射前に脱落した
            # 場合はc.aliveがFalseになるため下のループで自然にスキップされ、発射されない。
            for c in circles:
                if not c.alive or not c.holding_gun or frame_idx < c.gun_fire_frame:
                    continue
                cx, cy = c.body.position
                others = [o for o in circles if o.alive and o.id != c.id and not _is_teammate(c, o)]
                if others:
                    target = min(others, key=lambda o: (o.body.position - c.body.position).get_length_sqrd())
                    tx, ty = target.body.position
                    tvx, tvy = target.body.velocity
                    angle = _lead_aim_angle(cx, cy, tx, ty, tvx, tvy, BULLET_SPEED)
                else:
                    angle = 0.0
                bullet_record = {
                    "shooter_id": c.id,
                    "x": cx,
                    "y": cy,
                    "angle": angle,
                    "start_frame": frame_idx,
                    "end_frame": None,
                    "hit_id": None,
                    "blocked": False,
                }
                result.bullets.append(bullet_record)
                active_bullets.append(bullet_record)
                c.holding_gun = False
                c.gun_fire_frame = -1

            # 弾の飛行判定。物理演算(pymunk)には乗せず、直進する幾何学的な位置を毎フレーム
            # 計算し、生存プレイヤーとの距離だけで当たり判定する(単純な直進のみのため十分)。
            if active_bullets:
                still_flying = []
                for b in active_bullets:
                    t = (frame_idx - b["start_frame"]) * DT
                    bx = b["x"] + math.cos(b["angle"]) * BULLET_SPEED * t
                    by = b["y"] + math.sin(b["angle"]) * BULLET_SPEED * t
                    done = False
                    if (
                        t > BULLET_MAX_SECONDS
                        or bx < layout["left"] - BULLET_RADIUS
                        or bx > layout["right"] + BULLET_RADIUS
                        or by < layout["top"] - BULLET_RADIUS
                        or by > layout["bottom"] + BULLET_RADIUS
                    ):
                        b["end_frame"] = frame_idx
                        done = True
                    else:
                        shooter = entity_by_id.get(b["shooter_id"])
                        for o in circles:
                            if not o.alive or o.id == b["shooter_id"]:
                                continue
                            if shooter is not None and _is_teammate(shooter, o):
                                continue
                            ox, oy = o.body.position
                            if ((bx - ox) ** 2 + (by - oy) ** 2) ** 0.5 <= BULLET_RADIUS + o.radius:
                                if o.shielded_until_frame > frame_idx:
                                    b["blocked"] = True
                                elif o.bullet_hits_remaining > 1:
                                    # 2026-09-21、match_type="boss"専用: ボスは複数発被弾するまで
                                    # 脱落しない(有利ステータス)。命中自体は起きているので弾は
                                    # ここで消すが、脱落・elimination_orderへの記録はしない。
                                    o.bullet_hits_remaining -= 1
                                else:
                                    o.alive = False
                                    o.eliminated_frame = frame_idx
                                    result.elimination_order.append(
                                        {"id": o.id, "frame": frame_idx, "x": ox, "y": oy, "radius": o.radius, "cause": "bullet"}
                                    )
                                    space.remove(o.body, o.shape)
                                    body_to_entity.pop(id(o.body), None)
                                    b["hit_id"] = o.id
                                b["end_frame"] = frame_idx
                                done = True
                                break
                    if not done:
                        still_flying.append(b)
                active_bullets = still_flying

        # weapon_colosseum専用: 5種の武器(剣/槍/ハンマー/弓矢/斧、筆は2026-09-20にユーザー指示で
        # 廃止)の出現・拾得・各武器固有の攻撃処理・矢の飛行判定をまとめて行う。他ルールは
        # 「1回の接触/場外で即脱落」だが、このルールはHP(10)が0になるまで生存する。脱落自体は_apply_weapon_damage/
        # _weapon_eliminate(gun_duelの弾ヒット処理と同じ場所・同じパターン)で直接行い、
        # 後段の汎用eliminated判定(area_control/goal_reach/場外用)には一切触れない。
        if rule == "weapon_colosseum":
            # 武器の出現・拾得。2026-09-20、ユーザー指示: 「1ゲーム内での武器の再出現はなし
            # (無駄に散らかる)。一度所持した武器はずっと持つ」に対応し、各種類1回だけ出現させ、
            # 拾われても再出現させない(weapon_ever_spawnedで一度きりに制御)。拾った武器を
            # 手放す処理もどこにも実装していないため、保持したプレイヤーは脱落するまでずっと
            # 同じ武器を持ち続ける。
            for kind in WEAPON_KINDS:
                if (
                    kind not in weapon_ever_spawned
                    and active_weapons[kind] is None
                    and frame_idx >= weapon_respawn_at_frame[kind]
                ):
                    # 2026-09-21、ユーザー指示「スワイプ離脱(広告誤認・無風)防止」対応:
                    # weapon_colosseumの武器は一度きりの出現(再出現なし)のため、常に
                    # 「開幕直後の初回スポーン」に相当する。中央寄り(半径40%圏内)に絞り、
                    # 全プレイヤーの中央方向への強い初期インパルスと噛み合わせて早期の
                    # 武器の奪い合いを作る。
                    wx, wy = _random_food_position(weapon_rng, shape, half_x * 0.4, half_y * 0.4)
                    weapon_record = {"kind": kind, "x": wx, "y": wy, "start_frame": frame_idx, "end_frame": None, "picked_by": None}
                    result.weapons.append(weapon_record)
                    active_weapons[kind] = weapon_record
                    weapon_ever_spawned.add(kind)
                record = active_weapons[kind]
                if record is not None:
                    wx, wy = record["x"], record["y"]
                    for c in circles:
                        if not c.alive or c.weapon is not None:
                            continue
                        cx, cy = c.body.position
                        if ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5 <= WEAPON_PICKUP_RADIUS + c.radius:
                            c.weapon = kind
                            c.weapon_cooldown_until_frame = frame_idx  # 拾った直後から攻撃可能
                            record["end_frame"] = frame_idx
                            record["picked_by"] = c.id
                            active_weapons[kind] = None  # 再出現はしない(weapon_ever_spawned済みのため)
                            break

            # 槍でピン留めされている間は移動不能にする
            for c in circles:
                if c.alive and c.pinned_until_frame > frame_idx:
                    c.body.velocity = (0.0, 0.0)

            # ヒットストップ中は速度ゼロで静止させ、解除フレームで保存していた速度
            # (+静止していた間に本来受けていたはずの重力分)を復元する。
            # (_trigger_hitstop/_apply_weapon_damageのコメント参照。運動量を失わないための実装)。
            for c in circles:
                if not c.alive:
                    continue
                if c.hitstop_until_frame > frame_idx:
                    c.body.velocity = (0.0, 0.0)
                elif c.hitstop_until_frame == frame_idx:
                    gravity_gain = gravity * (HITSTOP_FRAMES * DT)
                    c.body.velocity = (c.hitstop_saved_vx, c.hitstop_saved_vy + gravity_gain)
                    c.hitstop_until_frame = -1

            # 各武器保持者の攻撃処理(近接系は最寄りの相手との距離、弓矢は先読み照準)
            for c in circles:
                if not c.alive or c.weapon is None:
                    continue
                cx, cy = c.body.position
                others = [o for o in circles if o.alive and o.id != c.id and not _is_teammate(c, o)]
                nearest = None
                nearest_dist = None
                if others:
                    nearest = min(others, key=lambda o: (o.body.position - c.body.position).get_length_sqrd())
                    nox, noy = nearest.body.position
                    nearest_dist = ((nox - cx) ** 2 + (noy - cy) ** 2) ** 0.5

                if c.weapon == "sword":
                    # 2026-09-21、ユーザー指摘対応: 当たり判定はプレイヤー中心でなく、実際に
                    # 描画されている(周回中の)剣の位置を起点に行う(_weapon_blade_position参照)。
                    # クールタイムは武器自体でなく(攻撃側,対象)の組み合わせごとに管理し、
                    # range内の相手全員を毎フレーム判定するため、複数の相手を同時期に
                    # (それぞれ独立したクールタイムで)攻撃できる。
                    wx, wy = _weapon_blade_position(c, "sword", frame_idx)
                    for o in others:
                        ox, oy = o.body.position
                        if ((ox - wx) ** 2 + (oy - wy) ** 2) ** 0.5 > SWORD_RANGE:
                            continue
                        key = (c.id, o.id)
                        if frame_idx >= melee_pair_cooldown.get(key, 0):
                            _apply_weapon_damage(c.id, o, SWORD_DAMAGE, frame_idx, "sword")
                            melee_pair_cooldown[key] = frame_idx + int(SWORD_COOLDOWN_SECONDS * FPS)

                elif c.weapon == "spear":
                    for o in others:
                        ox, oy = o.body.position
                        if ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5 > SPEAR_RANGE:
                            continue
                        key = (c.id, o.id)
                        if frame_idx >= melee_pair_cooldown.get(key, 0):
                            dmg = SPEAR_DAMAGE
                            if _distance_to_nearest_wall(ox, oy, shape, half_x, half_y) <= SPEAR_WALL_MARGIN:
                                dmg += SPEAR_WALL_BONUS_DAMAGE
                                o.pinned_until_frame = frame_idx + int(SPEAR_PIN_DURATION_SECONDS * FPS)
                            _apply_weapon_damage(c.id, o, dmg, frame_idx, "spear")
                            melee_pair_cooldown[key] = frame_idx + int(SPEAR_COOLDOWN_SECONDS * FPS)

                elif c.weapon == "hammer":
                    # 2026-09-21、ユーザー指摘対応: 剣と同様、当たり判定・ノックバック方向とも
                    # プレイヤー中心でなく実際に描画されている槌の位置を起点にする。
                    wx, wy = _weapon_blade_position(c, "hammer", frame_idx)
                    for o in others:
                        ox, oy = o.body.position
                        dx, dy = ox - wx, oy - wy
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > HAMMER_RANGE:
                            continue
                        key = (c.id, o.id)
                        if frame_idx >= melee_pair_cooldown.get(key, 0):
                            safe_dist = dist or 1.0
                            o.body.velocity = (
                                o.body.velocity[0] + dx / safe_dist * HAMMER_KNOCKBACK,
                                o.body.velocity[1] + dy / safe_dist * HAMMER_KNOCKBACK,
                            )
                            _apply_weapon_damage(c.id, o, HAMMER_DAMAGE, frame_idx, "hammer", target_hitstop=False)
                            melee_pair_cooldown[key] = frame_idx + int(HAMMER_COOLDOWN_SECONDS * FPS)

                elif c.weapon == "bow":
                    if nearest is not None and frame_idx >= c.weapon_cooldown_until_frame:
                        nox, noy = nearest.body.position
                        nvx, nvy = nearest.body.velocity
                        angle = _lead_aim_angle(cx, cy, nox, noy, nvx, nvy, ARROW_SPEED)
                        arrow_record = {
                            "shooter_id": c.id,
                            "x": cx,
                            "y": cy,
                            "vx": math.cos(angle) * ARROW_SPEED,
                            "vy": math.sin(angle) * ARROW_SPEED,
                            # vx0/vy0(発射時点の速度)は描画側が放物線を再計算するために保持する。
                            # vx/vyはシミュレーション中にARROW_GRAVITYで毎フレーム更新される
                            # 「現在速度」なので、そのままでは軌道の再現に使えない。
                            "vx0": math.cos(angle) * ARROW_SPEED,
                            "vy0": math.sin(angle) * ARROW_SPEED,
                            "start_frame": frame_idx,
                            "end_frame": None,
                            "hit_id": None,
                        }
                        result.arrows.append(arrow_record)
                        active_arrows.append(arrow_record)
                        # 反動: 発射方向と逆向きにshooterを押す(指示書「反動」対応)
                        c.body.velocity = (
                            c.body.velocity[0] - math.cos(angle) * BOW_RECOIL,
                            c.body.velocity[1] - math.sin(angle) * BOW_RECOIL,
                        )
                        c.weapon_cooldown_until_frame = frame_idx + int(BOW_COOLDOWN_SECONDS * FPS)

                elif c.weapon == "axe":
                    # 2026-09-21、ユーザー指示で全面刷新(3回目、最終版): 「基本は止めて、
                    # 定期的に当たり判定・ぶっ飛ばし判定つきの高速回転をする」。静止中は
                    # 何もしない(描画側で他の近接武器と同じ汎用の構えを表示する)。一定間隔
                    # (クールダウン)で高速回転(axe_spin_until_frame)を開始し、回転中は
                    # 届く範囲(AXE_SPIN_REACH)にいる相手を(攻撃側,対象)ごとのクールタイムで
                    # 独立にヒットさせる(melee_pair_cooldownの次回解禁がAXE_COOLDOWN_SECONDS後に
                    # なるため、1回の回転で同じ相手を連続ヒットすることはない)。
                    if others and frame_idx >= c.weapon_cooldown_until_frame:
                        c.axe_spin_until_frame = frame_idx + int(AXE_SPIN_DURATION_SECONDS * FPS)
                        c.weapon_cooldown_until_frame = frame_idx + int(AXE_COOLDOWN_SECONDS * FPS)
                    if frame_idx < c.axe_spin_until_frame:
                        # 2026-09-21、ユーザー指摘対応: 剣/ハンマーと同様、当たり判定・
                        # ノックバック方向ともプレイヤー中心でなく実際に回転している斧の位置を起点にする。
                        wx, wy = _weapon_blade_position(c, "axe", frame_idx)
                        for o in others:
                            ox, oy = o.body.position
                            dx, dy = ox - wx, oy - wy
                            dist = (dx * dx + dy * dy) ** 0.5
                            if dist > AXE_SPIN_REACH:
                                continue
                            key = (c.id, o.id)
                            if frame_idx >= melee_pair_cooldown.get(key, 0):
                                safe_dist = dist or 1.0
                                o.body.velocity = (
                                    o.body.velocity[0] + dx / safe_dist * AXE_KNOCKBACK,
                                    o.body.velocity[1] + dy / safe_dist * AXE_KNOCKBACK,
                                )
                                _apply_weapon_damage(c.id, o, AXE_DAMAGE, frame_idx, "axe", target_hitstop=False)
                                melee_pair_cooldown[key] = frame_idx + int(AXE_COOLDOWN_SECONDS * FPS)

            # 矢(弓矢)の飛行判定。gun_duelの弾と同じく物理演算(pymunk)には乗せず毎フレーム
            # 位置を直接積分するが、矢は専用の追加重力(ARROW_GRAVITY)を受けて弾道が落ちる点が
            # 弾と異なる(指示書「重力の影響を受ける」対応)。
            if active_arrows:
                still_flying_arrows = []
                for a in active_arrows:
                    t = (frame_idx - a["start_frame"]) * DT
                    a["vy"] += ARROW_GRAVITY * DT
                    a["x"] += a["vx"] * DT
                    a["y"] += a["vy"] * DT
                    ax, ay = a["x"], a["y"]
                    done = False
                    if (
                        t > ARROW_MAX_SECONDS
                        or ax < layout["left"] - ARROW_RADIUS
                        or ax > layout["right"] + ARROW_RADIUS
                        or ay < layout["top"] - ARROW_RADIUS
                        or ay > layout["bottom"] + ARROW_RADIUS
                    ):
                        a["end_frame"] = frame_idx
                        done = True
                    else:
                        arrow_shooter = entity_by_id.get(a["shooter_id"])
                        for o in circles:
                            if not o.alive or o.id == a["shooter_id"]:
                                continue
                            if arrow_shooter is not None and _is_teammate(arrow_shooter, o):
                                continue
                            ox, oy = o.body.position
                            if ((ax - ox) ** 2 + (ay - oy) ** 2) ** 0.5 <= ARROW_RADIUS + o.radius:
                                shooter = entity_by_id.get(a["shooter_id"])
                                if shooter is not None:
                                    _apply_weapon_damage(shooter.id, o, BOW_DAMAGE, frame_idx, "bow")
                                a["hit_id"] = o.id
                                a["end_frame"] = frame_idx
                                done = True
                                break
                    if not done:
                        still_flying_arrows.append(a)
                active_arrows = still_flying_arrows

        # 期限切れの毒ワイヤーを除去
        if active_wires:
            still_active = []
            for w in active_wires:
                if frame_idx >= w["expire_frame"]:
                    space.remove(w["shape"])
                else:
                    still_active.append(w)
            active_wires = still_active

        # growth_surgeの巨大化を時間経過で元に戻す(倍率で割り戻すため、
        # 巨大化中にabsorb_growthでさらに吸収していても自然に整合する)
        if pending_growth_reverts:
            still_growing = []
            for g in pending_growth_reverts:
                e = g["entity"]
                if not e.alive:
                    continue
                if frame_idx >= g["expire_frame"]:
                    _resize_entity(space, e, e.radius / g["mult"], elasticity, friction)
                else:
                    still_growing.append(g)
            pending_growth_reverts = still_growing

        # absorb_growth専用: ランダムに出現する食べ物。触れると一定時間だけ相手を吸収できるようになる
        # (「どちらが吸収するのか」を見た目で分かるようにするための仕組み、2026-09-09導入。
        # on_begin内のマージ判定はempowered_until_frameを見ているため、ここで設定するだけで反映される)
        if rule == "absorb_growth":
            if active_food is None and frame_idx >= food_respawn_at_frame:
                fx, fy = _random_food_position(food_rng, shape, half_x, half_y)
                food_record = {"x": fx, "y": fy, "start_frame": frame_idx, "end_frame": None, "eaten_by": None}
                result.food_items.append(food_record)
                active_food = food_record
            if active_food is not None:
                fx, fy = active_food["x"], active_food["y"]
                for c in circles:
                    if not c.alive:
                        continue
                    cx, cy = c.body.position
                    if ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5 <= FOOD_PICKUP_RADIUS + c.radius:
                        c.empowered_until_frame = frame_idx + int(EMPOWERED_DURATION_SECONDS * FPS)
                        active_food["end_frame"] = frame_idx
                        active_food["eaten_by"] = c.id
                        active_food = None
                        food_respawn_at_frame = frame_idx + int(FOOD_RESPAWN_DELAY_SECONDS * FPS)
                        break

        # area_control専用: 保護ゾーン。安全地帯の内側に数秒ごとに小さなゾーンが出現する。
        # 固定座標のまま存在し、プレイヤーが入ると実体化(物理的な囲いが生成され)、
        # プレイヤーはその固定座標の中で数秒間跳ね返り続ける(閉じ込められる)。囲いの中にいる間は
        # 安全地帯の外に出ていても脱落しない(2026-09-09、ユーザー要望「shrinking safe zoneだけでは
        # 張り合いがない」への対応。2026-09-09に「無敵で自由に動ける」から「物理的に閉じ込める」に
        # ユーザー訂正あり)。
        if rule == "area_control":
            if active_protection_zone is None and frame_idx >= protection_zone_respawn_at_frame:
                current_zone_r = _zone_radius(frame_idx / FPS, layout["zone_start_radius"])
                # 安全地帯の開始半径は壁自体より大きい(ZONE_START_RADIUS=extent*2.2)ため、
                # 試合序盤はcurrent_zone_rだけで抽選すると物理的な壁の外側(到達不可能な場所)に
                # 保護ゾーンが出現してしまうバグがあった(2026-09-09、レンダリング確認で発見)。
                # さらに、min(half_x, half_y)の円形範囲でしか判定していなかったため、hourglassの
                # ような非矩形の境界や、donut/cross/pegboardの内部障害物を一切考慮できておらず、
                # 依然として到達不可能な場所に出現することがあった(2026-09-09、ユーザー再指摘で
                # 修正)。_in_arena_bounds/_overlaps_terrain_obstacles(teleportと同じ判定)で
                # 実際に到達可能な位置だけに絞り込む。実体化後の囲い(cage、ピックアップ判定より
                # 大きい)が外壁やterrain障害物からはみ出さないよう、余白計算にはcageの半径を使う。
                pz_margin = PROTECTION_CAGE_RADIUS + 10 * _ARENA_SCALE
                usable_r = min(current_zone_r, half_x, half_y)
                available_r = max(0.0, usable_r - pz_margin)
                found_pos = None
                if available_r > 5:
                    for _ in range(24):
                        angle = protection_zone_rng.uniform(0, 2 * math.pi)
                        dist = protection_zone_rng.uniform(0, available_r)
                        pzx = ARENA_CX + math.cos(angle) * dist
                        pzy = ARENA_CY + math.sin(angle) * dist
                        if not _in_arena_bounds(pzx, pzy, shape, terrain, half_x, half_y, pz_margin):
                            continue
                        if _overlaps_terrain_obstacles(pzx, pzy, pz_terrain_segments, pz_terrain_circles, pz_margin):
                            continue
                        found_pos = (pzx, pzy)
                        break
                if found_pos is not None:
                    pzx, pzy = found_pos
                    zone_record = {
                        "x": pzx,
                        "y": pzy,
                        "radius": PROTECTION_ZONE_RADIUS,
                        "cage_radius": PROTECTION_CAGE_RADIUS,
                        "start_frame": frame_idx,
                        "end_frame": None,
                        "cage_end_frame": None,
                        "claimed_by": None,
                    }
                    result.protection_zones.append(zone_record)
                    active_protection_zone = zone_record
                else:
                    # 有効な位置が見つからなかった場合(安全地帯がまだ狭い終盤など)、少し待って
                    # 再試行する(即座に無限リトライしないよう短い間隔を置く)
                    protection_zone_respawn_at_frame = frame_idx + int(1.0 * FPS)
            if active_protection_zone is not None:
                pzx, pzy = active_protection_zone["x"], active_protection_zone["y"]
                pzr = active_protection_zone["radius"]
                # 2026-09-09、ユーザー指摘対応: 誰にも捕獲されないままPROTECTION_ZONE_LIFESPAN_SECONDS
                # 経過したら消滅させ、通常の間隔を置いて次のゾーンが出現できるようにする
                # (以前は無期限に居座り、次のゾーンが二度と出現しないことがあった)。
                if frame_idx - active_protection_zone["start_frame"] >= int(PROTECTION_ZONE_LIFESPAN_SECONDS * FPS):
                    active_protection_zone["end_frame"] = frame_idx
                    active_protection_zone = None
                    protection_zone_respawn_at_frame = frame_idx + int(PROTECTION_ZONE_INTERVAL_SECONDS * FPS)
            if active_protection_zone is not None:
                for c in circles:
                    if not c.alive:
                        continue
                    cx, cy = c.body.position
                    if ((cx - pzx) ** 2 + (cy - pzy) ** 2) ** 0.5 <= pzr + c.radius:
                        # 実体化: 固定座標(pzx, pzy)に物理的な囲い(枠と同じshape)を生成する。
                        # space.static_body(回転しない、world座標に固定)に追加することで、
                        # 「固定された座標のまま」という指定を満たす。
                        cage_shapes = []
                        for a, b in _cage_local_segments(shape, PROTECTION_CAGE_RADIUS):
                            seg = pymunk.Segment(space.static_body, (pzx + a[0], pzy + a[1]), (pzx + b[0], pzy + b[1]), WALL_THICKNESS / 2)
                            seg.friction = friction
                            seg.elasticity = elasticity
                            seg.collision_type = 1
                            space.add(seg)
                            cage_shapes.append(seg)
                        expire_frame = frame_idx + int(PROTECTION_ZONE_DURATION_SECONDS * FPS)
                        active_cages.append({"shapes": cage_shapes, "expire_frame": expire_frame})
                        c.protected_until_frame = expire_frame
                        active_protection_zone["end_frame"] = frame_idx
                        active_protection_zone["cage_end_frame"] = expire_frame
                        active_protection_zone["claimed_by"] = c.id
                        active_protection_zone = None
                        protection_zone_respawn_at_frame = frame_idx + int(PROTECTION_ZONE_INTERVAL_SECONDS * FPS)
                        result.ability_events.append(
                            {"type": "protection_claim", "id": c.id, "frame": frame_idx, "x": cx, "y": cy}
                        )
                        break
            # 期限切れの囲いを撤去する(space.remove)
            if active_cages:
                still_active = []
                for cage in active_cages:
                    if frame_idx >= cage["expire_frame"]:
                        for s in cage["shapes"]:
                            space.remove(s)
                    else:
                        still_active.append(cage)
                active_cages = still_active

        # 2026-09-21、ユーザー指示「カラーパレットの記号化」対応: 緑(color_index=2)は
        # 特殊能力とは別に、常に最も近い相手(チーム戦なら味方以外)から逃げる方向へ
        # 継続的な小さな加速度を受ける(クールダウン制の特殊能力と違い毎フレーム効く)。
        for c in circles:
            if not c.alive or not COLOR_TRAIT_MODIFIERS.get(c.color_index, {}).get("flee"):
                continue
            nearest = None
            nearest_dist_sqrd = None
            for o in circles:
                if not o.alive or o.id == c.id or _is_teammate(c, o):
                    continue
                d = (o.body.position - c.body.position).get_length_sqrd()
                if nearest_dist_sqrd is None or d < nearest_dist_sqrd:
                    nearest, nearest_dist_sqrd = o, d
            if nearest is None:
                continue
            away = c.body.position - nearest.body.position
            if away.length > 1e-3:
                direction = away.normalized()
                c.body.velocity = (
                    c.body.velocity[0] + direction[0] * FLEE_ACCEL * DT,
                    c.body.velocity[1] + direction[1] * FLEE_ACCEL * DT,
                )

        # 特殊能力の発動判定
        for c in circles:
            if not c.alive or not c.ability:
                continue
            c.ability_timer -= DT
            if c.ability_timer > 0:
                continue
            c.ability_timer = ABILITY_BASE_INTERVAL + ability_rng.uniform(-ABILITY_JITTER, ABILITY_JITTER)

            # 2026-09-21、match_type="boss"専用: ボスはboss_abilitiesに2種類のスキルを
            # 持っており、クールダウンが明けるたびに交互に切り替える(下のif/elif分岐自体は
            # 一切変更せず、「今回どの能力を使うか」だけを毎回1つ選び直す最小限の拡張)。
            if c.boss_abilities:
                c.boss_ability_index = (c.boss_ability_index + 1) % len(c.boss_abilities)
                c.ability = c.boss_abilities[c.boss_ability_index]

            if c.ability == "dash":
                speed_boost = _ability_param(ability_params, "dash", "speed_boost", DASH_SPEED_BOOST)
                # 2026-09-08バグ修正: 「脱落判定(c.alive=False)」は穴を完全に通過してESCAPE_MARGIN
                # を超えるまで確定しないため、既に壁の穴を抜けて枠の外に出ている(だが脱落判定は
                # まだ)プレイヤーが一瞬だけ「最も近い相手」として選ばれてしまうことがあった。
                # そちらへ突進すると自分も同じ穴を追いかけて自滅するため、alive判定に加えて
                # 枠の内側にまだいることも条件にする(ユーザー指摘対応)。
                others = [
                    o
                    for o in circles
                    if o.alive
                    and o.id != c.id
                    and not _is_teammate(c, o)
                    and layout["left"] <= o.body.position[0] <= layout["right"]
                    and layout["top"] <= o.body.position[1] <= layout["bottom"]
                ]
                if others:
                    cx, cy = c.body.position
                    target = min(others, key=lambda o: (o.body.position - c.body.position).get_length_sqrd())
                    direction = target.body.position - c.body.position
                    if direction.length > 1e-3:
                        direction = direction.normalized()
                        c.body.velocity = c.body.velocity + direction * speed_boost
                    result.ability_events.append(
                        {"type": "dash", "id": c.id, "frame": frame_idx, "x": cx, "y": cy, "target_id": target.id}
                    )

            elif c.ability == "poison_wire":
                wire_duration = _ability_param(ability_params, "poison_wire", "duration", POISON_WIRE_DURATION)
                length_factor = _ability_param(ability_params, "poison_wire", "length_factor", POISON_WIRE_LENGTH_FACTOR)
                wx, wy = _poison_wire_position(rule, hole_width, half_x, half_y, shape, terrain, ability_rng)
                half_len = (hole_width * length_factor / 2) if hole_width > 0 else 70
                a_pt = (wx - half_len, wy)
                b_pt = (wx + half_len, wy)
                wire_shape = pymunk.Segment(space.static_body, a_pt, b_pt, 6)
                wire_shape.friction = 0.2
                wire_shape.elasticity = 0.9
                wire_shape.collision_type = 4
                space.add(wire_shape)
                expire_frame = frame_idx + int(wire_duration * FPS)
                active_wires.append({"shape": wire_shape, "expire_frame": expire_frame, "a": a_pt, "b": b_pt})
                result.wires.append({"id": c.id, "start_frame": frame_idx, "end_frame": expire_frame, "a": a_pt, "b": b_pt})
                result.ability_events.append({"type": "poison_wire", "id": c.id, "frame": frame_idx, "x": wx, "y": wy})

            elif c.ability == "shockwave":
                radius = _ability_param(ability_params, "shockwave", "radius", SHOCKWAVE_RADIUS)
                impulse_mag = _ability_param(ability_params, "shockwave", "impulse", SHOCKWAVE_IMPULSE)
                cx, cy = c.body.position
                for o in circles:
                    if not o.alive or o.id == c.id or _is_teammate(c, o):
                        continue
                    ox, oy = o.body.position
                    dx, dy = ox - cx, oy - cy
                    dist = (dx * dx + dy * dy) ** 0.5
                    if 1e-3 < dist < radius:
                        falloff = 1 - dist / radius
                        push = impulse_mag * falloff
                        o.body.velocity = (o.body.velocity[0] + dx / dist * push, o.body.velocity[1] + dy / dist * push)
                result.ability_events.append({"type": "shockwave", "id": c.id, "frame": frame_idx, "x": cx, "y": cy})

            elif c.ability == "vortex":
                # 2026-09-09: 旧freeze(周囲を強制減速)から変更。「相手を遅くする」演出は
                # 終盤の見栄えを悪くするというユーザー指摘のため、shockwaveの逆(周囲を
                # 自分の方へ引き寄せる)にした。減速させないため衝突・接近が増え、むしろ
                # 見応えが上がる方向の効果になる。
                radius = _ability_param(ability_params, "vortex", "radius", VORTEX_RADIUS)
                pull_force = _ability_param(ability_params, "vortex", "pull_force", VORTEX_PULL_FORCE)
                cx, cy = c.body.position
                for o in circles:
                    if not o.alive or o.id == c.id or _is_teammate(c, o):
                        continue
                    ox, oy = o.body.position
                    dx, dy = cx - ox, cy - oy  # 相手→自分方向(引き寄せる向き)
                    dist = (dx * dx + dy * dy) ** 0.5
                    if 1e-3 < dist < radius:
                        falloff = 1 - dist / radius
                        pull = pull_force * falloff
                        o.body.velocity = (o.body.velocity[0] + dx / dist * pull, o.body.velocity[1] + dy / dist * pull)
                result.ability_events.append({"type": "vortex", "id": c.id, "frame": frame_idx, "x": cx, "y": cy})

            elif c.ability == "growth_surge":
                mult = _ability_param(ability_params, "growth_surge", "mult", GROWTH_SURGE_MULT)
                duration = _ability_param(ability_params, "growth_surge", "duration", GROWTH_SURGE_DURATION)
                cx, cy = c.body.position
                _resize_entity(space, c, c.radius * mult, elasticity, friction)
                pending_growth_reverts.append({"entity": c, "expire_frame": frame_idx + int(duration * FPS), "mult": mult})
                result.ability_events.append({"type": "growth_surge", "id": c.id, "frame": frame_idx, "x": cx, "y": cy})

            elif c.ability == "teleport":
                old_x, old_y = c.body.position
                result.ability_events.append(
                    {"type": "teleport", "id": c.id, "frame": frame_idx, "x": old_x, "y": old_y, "phase": "out"}
                )
                new_pos = _find_teleport_spot(circles, c, ability_rng, half_x, half_y, shape, terrain)
                if new_pos:
                    c.body.position = new_pos
                    result.ability_events.append(
                        {"type": "teleport", "id": c.id, "frame": frame_idx, "x": new_pos[0], "y": new_pos[1], "phase": "in"}
                    )

            elif c.ability == "speed_boost":
                mult = _ability_param(ability_params, "speed_boost", "mult", SPEED_BOOST_MULT)
                base = _ability_param(ability_params, "speed_boost", "base", SPEED_BOOST_BASE)
                vx, vy = c.body.velocity
                speed = (vx * vx + vy * vy) ** 0.5
                if speed < 30:
                    angle = ability_rng.uniform(0, 2 * math.pi)
                    c.body.velocity = (math.cos(angle) * base, math.sin(angle) * base)
                else:
                    c.body.velocity = (vx * mult, vy * mult)
                cx, cy = c.body.position
                result.ability_events.append({"type": "speed_boost", "id": c.id, "frame": frame_idx, "x": cx, "y": cy})

            elif c.ability == "slam":
                boost = _ability_param(ability_params, "slam", "boost", SLAM_BOOST)
                vx, vy = c.body.velocity
                c.body.velocity = (vx, vy + boost)
                cx, cy = c.body.position
                result.ability_events.append({"type": "slam", "id": c.id, "frame": frame_idx, "x": cx, "y": cy})

        frame_state = []
        winner_this_frame = None
        for c in circles:
            if not c.alive:
                continue
            x, y = c.body.position
            # 2026-09-09: 枠が真円/正方形化(ARENA_HALF_EXTENT基準)されたため、「画面外に出た」の
            # 基準もキャンバス端(旧ARENA_W/ARENA_H)ではなく枠自体の境界を基準にする
            # 2026-09-08バグ修正: 枠が回転している間、壁もプレイヤーと一緒に回転しているのに
            # この判定はワールド座標の固定された軸並行境界のまま比較していたため、回転した枠の
            # 角に触れているだけ(枠の外に出ていない)プレイヤーが誤って脱落判定されることがあった
            # (回転角45°付近で正方形の角がARENA_CXから最大約half_x*sqrt(2)まで達するため)。
            # 枠のローカル座標系(=枠の回転を打ち消した座標系)に変換してから判定することで解消する
            # (rotation_speed=0の場合は従来と完全に同じ結果になる)。
            cur_angle = arena_body.angle
            dxw, dyw = x - ARENA_CX, y - ARENA_CY
            cos_a, sin_a = math.cos(-cur_angle), math.sin(-cur_angle)
            local_x = dxw * cos_a - dyw * sin_a
            local_y = dxw * sin_a + dyw * cos_a
            escaped_bounds = (
                local_x < -escape_half_x - ESCAPE_MARGIN or local_x > escape_half_x + ESCAPE_MARGIN
                or local_y < -escape_half_y - ESCAPE_MARGIN or local_y > escape_half_y + ESCAPE_MARGIN
            )
            # 2026-09-20、ユーザー指摘対応: 「そのルール独自の脱落原因でなく、穴/端から
            # 場外に落ちた脱落の方が多い試合は選ばないように」フィルター(scoring側)のため、
            # 脱落原因をcauseとして記録する。escaped_bounds(枠の外に完全に出た=穴や端からの
            # 場外)を最優先の原因とし、area_control/trapは「まだ枠内だが独自条件に該当」した
            # 場合のみそちらの原因を記録する(枠外に出ていれば当然ゾーンの外でもあるため、
            # 二重にカウントしないようelifで排他にする)。
            eliminated = escaped_bounds
            elimination_cause = "escaped" if escaped_bounds else None

            if rule == "area_control":
                dist_from_center = ((x - ARENA_CX) ** 2 + (y - ARENA_CY) ** 2) ** 0.5
                outside_zone = dist_from_center > _zone_radius(frame_idx / FPS, layout["zone_start_radius"])
                protected = c.protected_until_frame > frame_idx
                if not eliminated and outside_zone and not protected:
                    eliminated = True
                    elimination_cause = "zone_out"
            elif rule == "goal_reach":
                gx, gy = layout["goal_point"]
                dist_to_goal = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
                # 2026-09-08、ユーザー要望のバリア追加に伴い、バリアが壊れるまでは到達判定自体を
                # 無効にする(物理的にも壁で塞がれているはずだが、念のため二重に防ぐ)。
                if dist_to_goal <= GOAL_RADIUS and not barrier_state["active"]:
                    winner_this_frame = c

            if result.trap:
                tx, ty = layout["trap_center"]
                if ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5 <= TRAP_RADIUS:
                    if not eliminated:
                        elimination_cause = "trap"
                    eliminated = True

            if eliminated:
                c.alive = False
                c.eliminated_frame = frame_idx
                result.elimination_order.append(
                    {"id": c.id, "frame": frame_idx, "x": x, "y": y, "radius": c.radius, "cause": elimination_cause or "escaped"}
                )
                space.remove(c.body, c.shape)
                body_to_entity.pop(id(c.body), None)
                continue
            vx, vy = c.body.velocity
            frame_state.append(
                {
                    "id": c.id,
                    "x": x,
                    "y": y,
                    "radius": c.radius,
                    "angle": c.body.angle,
                    "vx": vx,
                    "vy": vy,
                    "empowered": c.empowered_until_frame > frame_idx,
                    "holding_gun": c.holding_gun,
                    "gun_fire_frame": c.gun_fire_frame,
                    "shielded": c.shielded_until_frame > frame_idx,
                    "protected": c.protected_until_frame > frame_idx,
                    "hp": c.hp if rule == "weapon_colosseum" else None,
                    "max_hp": (WEAPON_STARTING_HP * BOSS_HP_MULT if c.is_boss else WEAPON_STARTING_HP) if rule == "weapon_colosseum" else None,
                    "weapon": c.weapon if rule == "weapon_colosseum" else None,
                    "axe_spinning": frame_idx < c.axe_spin_until_frame,
                    "is_boss": c.is_boss,
                }
            )
        result.frames.append(frame_state)

        alive = [c for c in circles if c.alive]

        if rule == "goal_reach" and winner_this_frame is not None:
            result.decided_frame = frame_idx
            result.final_arena_angle = arena_body.angle
            result.winner_id = winner_this_frame.id
            result.winning_team_id = winner_this_frame.team_id  # match_type="team"の場合のみ意味を持つ
            result.final_position = tuple(winner_this_frame.body.position)
            result.final_player_angle = winner_this_frame.body.angle
            break

        # 2026-09-21、match_type="team"専用: 「最後の1人」ではなく「最後の1チーム」で決着する
        # (goal_reach以外の4ルール: hole_fall/area_control/gun_duel/weapon_colosseum)。
        if match_type == "team":
            remaining_teams = {c.team_id for c in alive}
            if len(remaining_teams) <= 1:
                result.decided_frame = frame_idx
                result.final_arena_angle = arena_body.angle
                if alive:
                    result.winner_id = alive[0].id
                    result.winning_team_id = alive[0].team_id
                    result.final_position = tuple(alive[0].body.position)
                    result.final_player_angle = alive[0].body.angle
                break
        # 2026-09-21、match_type="boss"専用: 「ボスの脱落」または「ボス以外の全脱落」で決着する
        # (weapon_colosseum/gun_duelの2ルールのみ対象。他ルールでboss_abilitiesを渡さなければ
        # is_boss=Trueのエンティティが存在しないため、この分岐は実質発火しない)。
        elif match_type == "boss":
            boss_alive = [c for c in alive if c.is_boss]
            non_boss_alive = [c for c in alive if not c.is_boss]
            if not boss_alive or not non_boss_alive:
                result.decided_frame = frame_idx
                result.final_arena_angle = arena_body.angle
                winner_pool = boss_alive or non_boss_alive
                if winner_pool:
                    winner_entity = winner_pool[0]
                    result.winner_id = winner_entity.id
                    result.winner_is_boss = winner_entity.is_boss
                    result.final_position = tuple(winner_entity.body.position)
                    result.final_player_angle = winner_entity.body.angle
                break
        elif len(alive) <= 1:
            result.decided_frame = frame_idx
            result.final_arena_angle = arena_body.angle
            if alive:
                result.winner_id = alive[0].id
                result.final_position = tuple(alive[0].body.position)
                result.final_player_angle = alive[0].body.angle
            break

    return result, circles


def summarize(result: SimResult) -> dict:
    """基準判定(4章)に使う指標の元になる集計値。プロトタイプ段階では主要な値のみ。"""
    decided_frame = result.decided_frame if result.decided_frame is not None else len(result.frames)
    decision_seconds = decided_frame / FPS
    return {
        "seed": result.seed,
        "n_circles": result.n_circles,
        "shape": result.shape,
        "rule": result.rule,
        "gravity": result.gravity,
        "elasticity": result.elasticity,
        "rotation_speed": result.rotation_speed,
        "hole_width": result.hole_width,
        "winner_id": result.winner_id,
        "decision_seconds": round(decision_seconds, 2),
        "collision_count": len(result.collisions),
        "elimination_count": len(result.elimination_order),
        "ability_event_count": len(result.ability_events),
        "elimination_order": result.elimination_order,
        "reached_max_duration_without_decision": result.winner_id is None,
    }


def _rotation_zoom_scale(rotation_speed: float) -> float:
    """回転ギミック用の描画ズーム倍率(現在は常に1.0=無補正)。

    2026-09-09、枠を真円/正方形(ARENA_HALF_EXTENT基準)に変更したことで、この補正自体が
    不要になった: 円は回転しても輪郭が変わらないため無条件に安全、正方形は対角線が
    ちょうどARENA_Wに収まるサイズにしてあるため全方向の回転でキャンバス外にはみ出さない。
    呼び出し側のscale引数を全て消す大きな変更を避けるため、関数自体は残して常に1.0を返す。
    """
    return 1.0


def _zoom_point(pt: tuple, scale: float) -> tuple:
    if scale == 1.0:
        return pt
    x, y = pt
    return (ARENA_CX + (x - ARENA_CX) * scale, ARENA_CY + (y - ARENA_CY) * scale)


def _rotate_point(local: tuple, angle: float, scale: float = 1.0) -> tuple:
    lx, ly = local
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    wx = ARENA_CX + lx * cos_a - ly * sin_a
    wy = ARENA_CY + lx * sin_a + ly * cos_a
    return _zoom_point((wx, wy), scale)


def _draw_arena(
    draw: ImageDraw.ImageDraw,
    angle: float,
    shape: str,
    hole_width: float,
    half_x: float = ARENA_HALF_EXTENT,
    half_y: float = ARENA_HALF_EXTENT,
    scale: float = 1.0,
) -> None:
    width = max(1, int(WALL_VISUAL_THICKNESS * scale))

    if shape == "square":
        # 2026-09-08、ユーザー指摘対応: 各壁セグメントを個別のdraw.line()で描画すると、直角に
        # 交わる外側の角に線幅ぶんの小さなくぼみ(継ぎ目)ができる。角に正方形パッチを重ねる
        # 対策を試したが、今度は角の先端がわずかに尖って飛び出す(はみ出し)副作用が出た。
        # 根本対応として、壁全体(穴の両端を起点・終点とする一筆書き)を1回のdraw.line()呼び出しで
        # 連続した折れ線として描画し、joint="curve"でPIL自身に角の継ぎ目を処理させる
        # (くぼみ・はみ出しのどちらも発生しない)。物理演算側(build_space)は従来通り
        # _wall_local_segmentsの個別セグメントを使うため、当たり判定の形状に変更はない。
        half_hole = hole_width / 2
        chain = [
            (half_hole, half_y),
            (half_x, half_y),
            (half_x, -half_y),
            (-half_x, -half_y),
            (-half_x, half_y),
            (-half_hole, half_y),
        ]
        points = [_rotate_point(p, angle, scale) for p in chain]
        draw.line(points, fill=(220, 220, 220), width=width, joint="curve")
        return

    for a, b in _wall_local_segments(shape, hole_width, half_x, half_y):
        draw.line(
            [_rotate_point(a, angle, scale), _rotate_point(b, angle, scale)],
            fill=(220, 220, 220),
            width=width,
        )


def _draw_hourglass_arena(draw: ImageDraw.ImageDraw, angle: float, half_x: float, half_y: float, scale: float = 1.0) -> None:
    """砂時計そのものが外枠になる場合の描画(2026-09-09、ユーザー要望で通常の四角い外枠を廃止)。
    _draw_arenaのsquare分岐と同じく、始点に戻る1本の連続した折れ線をjoint="curve"で描画し、
    辺の継ぎ目のくぼみ・はみ出しを防ぐ(_hourglass_boundary_chainは物理演算側とも共有)。"""
    width = max(1, int(WALL_VISUAL_THICKNESS * scale))
    chain = _hourglass_boundary_chain(half_x, half_y)
    points = [_rotate_point(p, angle, scale) for p in chain]
    draw.line(points, fill=(220, 220, 220), width=width, joint="curve")


def _draw_pin_wall_arena(draw: ImageDraw.ImageDraw, angle: float, shape: str, hole_width: float, half_x: float, half_y: float, scale: float = 1.0) -> None:
    """外枠をピンの列に置き換える場合の描画(2026-09-13)。_pin_wall_positionsは物理演算
    (build_space)と同じ座標を返すため、当たり判定と見た目が常に一致する。"""
    for x, y, r in _pin_wall_positions(shape, hole_width, half_x, half_y):
        zx, zy = _rotate_point((x, y), angle, scale)
        zr = r * scale
        draw.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=PIN_WALL_COLOR, outline=(120, 120, 130), width=max(1, int(2 * scale)))


def _draw_arena_walls(
    draw: ImageDraw.ImageDraw,
    angle: float,
    shape: str,
    hole_width: float,
    terrain: str | None,
    half_x: float = ARENA_HALF_EXTENT,
    half_y: float = ARENA_HALF_EXTENT,
    scale: float = 1.0,
) -> None:
    """外枠描画の唯一の入口。terrain=="hourglass"は外枠そのものを砂時計形状に置き換える
    特殊扱いのため、呼び出し側ごとに`if terrain == "hourglass"`を書かせるとエピローグ等の
    分岐追加漏れで四角になって描画される(2026-09-12、ループ演出のズームアウトで実際に発生した
    バグ)。今後外枠を描画する箇所が増えてもここを呼ぶだけで済むよう一本化した。
    2026-09-13、pin_wallも同じ理由でここに合流させた。"""
    if terrain == "hourglass":
        _draw_hourglass_arena(draw, angle, half_x, half_y, scale)
    elif terrain == "pin_wall":
        _draw_pin_wall_arena(draw, angle, shape, hole_width, half_x, half_y, scale)
    else:
        _draw_arena(draw, angle, shape, hole_width, half_x, half_y, scale)


TERRAIN_COLOR = (220, 220, 220)  # 2026-09-13、ユーザー指摘で外枠の壁と同じ白に統一(旧: 灰色)


def _draw_terrain(
    draw: ImageDraw.ImageDraw,
    terrain: str | None,
    angle: float,
    half_x: float,
    half_y: float,
    scale: float = 1.0,
    flash_local_positions: frozenset | None = None,
) -> None:
    """内部地形障害物の描画。壁と同じ_rotate_point変換を使うことで、回転ギミックが
    作動していても壁と完全に同期して回転する(物理演算側もarena_bodyに追加しているため一致する)。
    2026-09-09、ユーザー指摘対応: 各chainを1本の連続したdraw.line()(joint="curve")で描画する。
    セグメントごとに個別のdraw.line()を呼んでいた旧実装は、外壁の角と同じ「継ぎ目のくぼみ・
    はみ出し」バグを内部地形でも再発させていた(_draw_arenaで既に解決済みのパターンを流用)。

    flash_local_positions(2026-09-13追加): pinball専用。直近でヒットしたバンパーのローカル
    座標(round(cx,1), round(cy,1))の集合。該当バンパーだけ指示書指定の一瞬の拡縮(1.2倍)を
    適用する(_iter_rendered_framesがresult.ability_eventsのbumper_x/bumper_yから構築する)。"""
    chains, circles = _terrain_obstacles(terrain, half_x, half_y)
    width = max(1, int(WALL_VISUAL_THICKNESS * scale))
    for chain in chains:
        points = [_rotate_point(p, angle, scale) for p in chain]
        draw.line(points, fill=TERRAIN_COLOR, width=width, joint="curve")
    if terrain == "pinball":
        # スリングショットはbuild_space側で別枠追加のため_terrain_obstaclesのchainsには
        # 含まれない(誘導スロープと反発係数を分けるため、build_space内のコメント参照)。
        # 描画も同じ理由で個別に行うが、座標はbuild_spaceと同じ_pinball_slingshot_chainsを
        # 共有しているため物理判定と見た目は一致する。
        for chain in _pinball_slingshot_chains(half_x, half_y):
            points = [_rotate_point(p, angle, scale) for p in chain]
            draw.line(points, fill=PINBALL_COLOR, width=width, joint="curve")
    bumper_fill = PINBALL_COLOR if terrain == "pinball" else (40, 40, 46)
    flash_local_positions = flash_local_positions or frozenset()
    for cx, cy, r in circles:
        zx, zy = _rotate_point((cx, cy), angle, scale)
        zr = r * scale
        if terrain == "pinball" and (round(cx, 1), round(cy, 1)) in flash_local_positions:
            zr *= PINBALL_FLASH_SCALE
            draw.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=(255, 255, 255), outline=TERRAIN_COLOR, width=max(1, int(4 * scale)))
            continue
        draw.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=bumper_fill, outline=TERRAIN_COLOR, width=max(1, int(4 * scale)))


def _draw_rule_overlay(draw: ImageDraw.ImageDraw, rule: str, t: float, layout: dict, scale: float = 1.0) -> None:
    if rule == "goal_reach":
        gx, gy = _zoom_point(layout["goal_point"], scale)
        r = GOAL_RADIUS * scale
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(255, 205, 60), width=5)
    elif rule == "area_control":
        cx, cy = _zoom_point((ARENA_CX, ARENA_CY), scale)
        r = _zone_radius(t, layout["zone_start_radius"]) * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 70, 50), width=5)


_DIST_FIELD_CACHE: dict[tuple, np.ndarray] = {}


def _distance_field(w: int, h: int, cx: float, cy: float) -> np.ndarray:
    """画面中心(cx,cy)からの距離マップ。w/h/cx/cyは1本の動画レンダリング中は不変なため
    キャッシュし、毎フレームのmgrid再計算を避ける。"""
    key = (w, h, round(cx, 1), round(cy, 1))
    cached = _DIST_FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy).astype(np.float32)
    _DIST_FIELD_CACHE[key] = dist
    return dist


# 2026-09-20、ユーザー指示(area_controlの緊張感強化): 安全地帯の外側(危険地帯)にも
# 薄いオーラを描き、「境界の外側は危ない」ことを視覚的に強調する。時間経過とともに
# 色が濃くなり、決着(ZONE_SHRINK_SECONDS経過)が近づくほど危険度が高まって見える。
AREA_CONTROL_AURA_COLOR = np.array([230, 40, 30], dtype=np.float32)
AREA_CONTROL_AURA_OUTER_FRAC = 0.78  # 壁の内側あたりまでを危険地帯の外縁とみなす


def _apply_area_control_danger_aura(img: Image.Image, t: float, layout: dict, scale: float) -> Image.Image:
    """area_control専用: 安全地帯の外側に、時間経過とともに強くなる赤いオーラを重ねる。
    プレイヤー本体を描く前に呼ぶこと(オーラの上にプレイヤーが乗るようにするため)。"""
    w, h = img.size
    cx, cy = _zoom_point((ARENA_CX, ARENA_CY), scale)
    zone_r = _zone_radius(t, layout["zone_start_radius"]) * scale
    outer_r = max(w, h) * AREA_CONTROL_AURA_OUTER_FRAC
    if outer_r <= zone_r:
        return img
    dist = _distance_field(w, h, cx, cy)
    alpha_map = np.clip((dist - zone_r) / (outer_r - zone_r), 0.0, 1.0)
    danger = min(1.0, t / ZONE_SHRINK_SECONDS)
    max_alpha = 0.10 + 0.35 * danger
    alpha_map = (alpha_map * max_alpha)[..., None]
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * (1 - alpha_map) + AREA_CONTROL_AURA_COLOR * alpha_map
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# 2026-09-20、ユーザー指示: トップライト(ゴッドレイ/環境光)。単一の光源(画面上部やや
# 左寄りの1点)から差し込む薄い光条+全体のゆるい陰影グラデーションを常時重ね、
# フラットな背景に奥行きを出す。判定には一切影響しない純粋な演出。
#
# 2026-09-20(続き、ユーザー指示による修正): 「光源は一箇所のみ・枠やプレイヤーにも
# 影を・光量を減らす」との指摘を受け、(1)3本ビーム→単一光源+単一光条に変更、
# (2)光源と反対側が暗くなる全体グラデーション(=枠面の陰影)を追加、
# (3)各プレイヤーに光源と反対方向への柔らかい影を追加、(4)光量(alpha)を0.11→0.05へ
# 大幅に下げた。w,hが不変な1本のレンダリング中はマスク/グラデーションを使い回す。
GODRAY_ALPHA_BASE = 0.05
GODRAY_COLOR = np.array([255, 248, 230], dtype=np.float32)
GODRAY_LIGHT_POS_FRAC = (0.5, -0.10)  # 光源位置(w,hに対する比率。中央上部・yが負=画面の外側)
GODRAY_SHADE_STRENGTH = 0.16  # 光源から最も遠い側をどれだけ暗くするか
GODRAY_ENTITY_SHADOW_STRENGTH = 0.55  # プレイヤーの影の濃さ(壁にかかった時にはっきり見える強さ)
_GODRAY_LAYER_CACHE: dict[tuple, np.ndarray] = {}
_GODRAY_SHADE_CACHE: dict[tuple, np.ndarray] = {}


def _light_position(w: int, h: int) -> tuple[float, float]:
    fx, fy = GODRAY_LIGHT_POS_FRAC
    return w * fx, h * fy


def _build_god_ray_layer(w: int, h: int) -> np.ndarray:
    """単一光源から下方向へ扇状に広がる光条のマスク(0〜1)を1回だけ作る。"""
    key = (w, h)
    cached = _GODRAY_LAYER_CACHE.get(key)
    if cached is not None:
        return cached
    light_x, light_y = _light_position(w, h)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    # 2026-09-20(修正): 「中央上部から、より横に広いライトに」との指示で、光源を中央固定にし
    # 上端・下端とも幅を大きく広げた(以前は片側寄り+細めのビームだった)。
    top_half = w * 0.10
    bottom_half = w * 0.48
    bottom_cx = light_x
    poly = [
        (light_x - top_half, light_y), (light_x + top_half, light_y),
        (bottom_cx + bottom_half, h), (bottom_cx - bottom_half, h),
    ]
    mdraw.polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(8, int(w * 0.03))))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    _GODRAY_LAYER_CACHE[key] = arr
    return arr


def _build_shade_gradient(w: int, h: int) -> np.ndarray:
    """光源から遠いほど暗くなる、画面全体にかかるゆるいグラデーション(乗数、1.0が無変化)。
    枠(壁)自体にも陰影が付いて見えるようにするための、幾何形状に依らない簡易近似。"""
    key = (w, h)
    cached = _GODRAY_SHADE_CACHE.get(key)
    if cached is not None:
        return cached
    light_x, light_y = _light_position(w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(xx - light_x, yy - light_y)
    max_dist = math.hypot(max(light_x, w - light_x), max(0, h - light_y))
    frac = np.clip(dist / max(1.0, max_dist), 0.0, 1.0)
    shade = 1.0 - frac * GODRAY_SHADE_STRENGTH
    _GODRAY_SHADE_CACHE[key] = shade
    return shade


def _apply_directional_shadows(arr: np.ndarray, frame_state: list, light_xy: tuple, scale: float) -> None:
    """各プレイヤーに、光源と反対方向へ落ちる柔らかい影をその場でarrへ焼き込む
    (in-place。呼び出し側で1回だけndarray化して使い回すための設計)。

    2026-09-20(修正): 当初は単純な円形の暗化だったため、背景がほぼ黒(24,24,28)の
    このゲームでは効果がほとんど見えなかった。光の方向に伸びる楕円形にして
    「影らしい」形にした上で濃さも引き上げ、壁(明るい灰色)にかかったときに
    はっきり見えるようにした。"""
    lx, ly = light_xy
    h, w = arr.shape[:2]
    for entry in frame_state:
        zx, zy = _zoom_point((entry["x"], entry["y"]), scale)
        zr = entry["radius"] * scale
        dx, dy = zx - lx, zy - ly
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        vx, vy = -uy, ux  # 影の伸び方向(u)に直交する軸
        long_r = zr * 2.6
        short_r = zr * 0.85
        center_dist = zr * 0.25 + long_r * 0.5  # 影の近い端がプレイヤーのすぐ外側に来るように
        scx, scy = zx + ux * center_dist, zy + uy * center_dist
        pad = max(long_r, short_r) * 1.2
        x0, x1 = max(0, int(scx - pad)), min(w, int(scx + pad))
        y0, y1 = max(0, int(scy - pad)), min(h, int(scy + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rel_x, rel_y = xx - scx, yy - scy
        local_u = rel_x * ux + rel_y * uy
        local_v = rel_x * vx + rel_y * vy
        d = np.hypot(local_u / long_r, local_v / short_r)
        falloff = np.clip(1.0 - d, 0.0, 1.0) ** 1.3
        darken = 1.0 - falloff * GODRAY_ENTITY_SHADOW_STRENGTH
        arr[y0:y1, x0:x1] *= darken[..., None]


DUST_PARTICLE_COUNT = 16
DUST_PARTICLE_RADIUS = 2.5
DUST_PARTICLE_FALL_SPEED = 18.0  # px/秒、ゆっくり下降
DUST_PARTICLE_DRIFT_AMPLITUDE = 10.0  # 左右のゆらぎ幅
_DUST_SEED_CACHE: dict[tuple, list] = {}


def _dust_particle_seeds(w: int, h: int) -> list[tuple[float, float, float, float]]:
    """godrayの光の筋の中を漂う塵パーティクル(指示書「オプション: 塵パーティクル」への対応、
    2026-09-20実装)の初期位置・個体差をキャッシュする。戻り値は(x0, y0, 位相, 速度倍率)。
    位置は画面全体からランダムに撒いておき、可視性をray_maskの値でゲートする
    (ビームの外では自然にほぼ見えなくなるため、ビーム形状を別途再計算する必要がない)。"""
    key = (w, h)
    cached = _DUST_SEED_CACHE.get(key)
    if cached is not None:
        return cached
    rng = random.Random(20260920)
    seeds = [
        (rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, math.tau), rng.uniform(0.7, 1.3))
        for _ in range(DUST_PARTICLE_COUNT)
    ]
    _DUST_SEED_CACHE[key] = seeds
    return seeds


def _apply_dust_particles(arr: np.ndarray, ray_mask: np.ndarray, frame_idx: int) -> None:
    """ビーム内を漂う塵をin-placeで焼き込む。ゆっくり下降しながら左右にゆらぎ、
    画面下まで来たら上端へループする。明るさはray_maskの値(ビーム内で1に近い)で
    ゲートするため、ビームの外ではほとんど見えない。"""
    h, w = arr.shape[:2]
    t = frame_idx / FPS
    for x0, y0, phase, speed_mult in _dust_particle_seeds(w, h):
        y = (y0 + t * DUST_PARTICLE_FALL_SPEED * speed_mult) % h
        x = x0 + math.sin(t * 0.6 + phase) * DUST_PARTICLE_DRIFT_AMPLITUDE
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        visibility = ray_mask[yi, xi]
        if visibility < 0.05:
            continue
        r = DUST_PARTICLE_RADIUS
        x0b, x1b = max(0, int(xi - r * 2)), min(w, int(xi + r * 2 + 1))
        y0b, y1b = max(0, int(yi - r * 2)), min(h, int(yi + r * 2 + 1))
        if x1b <= x0b or y1b <= y0b:
            continue
        yy, xx = np.mgrid[y0b:y1b, x0b:x1b].astype(np.float32)
        d = np.hypot(xx - x, yy - y) / r
        glow = np.clip(1.0 - d, 0.0, 1.0) ** 2
        arr[y0b:y1b, x0b:x1b] += (glow * visibility * 90.0)[..., None]


def _apply_lighting(img: Image.Image, frame_idx: int, frame_state: list, scale: float) -> Image.Image:
    """単一光源のトップライト+全体の陰影グラデーション+各プレイヤーの落ち影+塵パーティクルを
    まとめて適用する(2026-09-20)。1回のndarray変換で済ませ、負荷を抑える。"""
    w, h = img.size
    light_xy = _light_position(w, h)
    arr = np.asarray(img, dtype=np.float32)

    shade = _build_shade_gradient(w, h)
    arr *= shade[..., None]

    _apply_directional_shadows(arr, frame_state, light_xy, scale)

    ray_mask = _build_god_ray_layer(w, h)
    t = frame_idx / FPS
    breathe = 0.9 + 0.1 * math.sin(2 * math.pi * t / 6.0)
    alpha = GODRAY_ALPHA_BASE * breathe
    arr += ray_mask[..., None] * GODRAY_COLOR * alpha

    _apply_dust_particles(arr, ray_mask, frame_idx)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_accel_zone(draw: ImageDraw.ImageDraw, frame_idx: int, accel_zone_rect: tuple, scale: float = 1.0) -> None:
    """加速ゾーンの視覚表現。2026-09-09、ユーザー指摘(上向き矢印だけではダサい)により、
    下から上へ連続して上昇し続ける横波線のアニメーションに変更した。純粋な演出であり、
    当たり判定や実際の力の計算(ACCEL_ZONE_FORCE)には一切影響しない。"""
    x0, y0, x1, y1 = accel_zone_rect
    p0x, p0y = _zoom_point((x0, y0), scale)
    p1x, p1y = _zoom_point((x1, y1), scale)
    draw.rectangle([p0x, p0y, p1x, p1y], outline=(120, 200, 255), width=max(1, int(3 * scale)))

    zone_h = p1y - p0y
    if zone_h <= 0:
        return
    t = frame_idx / FPS
    wave_speed = 80.0 * scale  # 見た目の上昇速度(px/秒)
    n_lines = 3
    spacing = zone_h / n_lines
    amplitude = 6 * scale
    wavelength = max(20.0, (p1x - p0x) / 2.2)
    steps = 16

    for i in range(n_lines):
        y_line = p1y - ((t * wave_speed + i * spacing) % zone_h)  # 下端からループしながら上昇
        points = []
        for s in range(steps + 1):
            px = p0x + (p1x - p0x) * (s / steps)
            py = y_line + math.sin((px / wavelength) * 2 * math.pi + t * 3) * amplitude
            # 2026-09-08修正: sin変位でy_lineが枠の上下端付近にあるとゾーン矩形からはみ出ることが
            # あったため、ゾーン内(p0y〜p1y)にクランプする(見た目は端で波が収まるだけで違和感はない)
            py = min(p1y, max(p0y, py))
            points.append((px, py))
        draw.line(points, fill=(120, 200, 255), width=max(1, int(3 * scale)), joint="curve")


def _draw_flipper_zone(draw: ImageDraw.ImageDraw, layout: dict, scale: float = 1.0) -> None:
    """2026-09-13、pinball専用: 自動パルスフリッパーが作動する帯の位置を示す演出
    (当たり判定・キック自体には影響しない)。ユーザー指摘で、4辺を破線で囲む旧デザインは
    「点線で情報量が多くスマートでない」との指摘を受けたため、上端1本の実線だけに簡略化した
    (下端は壁/穴と重なるため元々冗長、左右の縦線も他の要素と交錯して見づらかった)。"""
    x0, y0 = _zoom_point((ARENA_CX - FLIPPER_ZONE_HALF_WIDTH, layout["bottom"] - FLIPPER_ZONE_HEIGHT), scale)
    x1, _ = _zoom_point((ARENA_CX + FLIPPER_ZONE_HALF_WIDTH, layout["bottom"]), scale)
    draw.line([(x0, y0), (x1, y0)], fill=FLIPPER_COLOR, width=max(1, int(2 * scale)))


def _draw_trap(draw: ImageDraw.ImageDraw, trap_center: tuple, scale: float = 1.0) -> None:
    tx, ty = _zoom_point(trap_center, scale)
    r = TRAP_RADIUS * scale
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(24, 12, 12), outline=(220, 60, 60), width=max(1, int(4 * scale)))
    d = r * 0.55
    draw.line([(tx - d, ty - d), (tx + d, ty + d)], fill=(220, 60, 60), width=max(1, int(3 * scale)))
    draw.line([(tx - d, ty + d), (tx + d, ty - d)], fill=(220, 60, 60), width=max(1, int(3 * scale)))


def _draw_wires(draw: ImageDraw.ImageDraw, wires: list, scale: float = 1.0) -> None:
    """poison_wire(紫)が設置する障害物の描画。2026-09-08、ユーザー指摘対応: 以前は縁が紫でも
    内側の芯が緑寄りの色(60,220,140)で、青っぽく見えて誰のスキルか分かりにくかった。
    芯も同系統の(明るい)紫にして、紫プレイヤーのスキルだと一目で分かるようにした。"""
    for a, b in wires:
        za, zb = _zoom_point(a, scale), _zoom_point(b, scale)
        draw.line([za, zb], fill=(160, 90, 220), width=max(1, int(9 * scale)))
        draw.line([za, zb], fill=(220, 190, 250), width=max(1, int(3 * scale)))


FOOD_COLOR = (255, 200, 60)  # プレイヤー色(COLORS/PALETTES)と重複しない専用の暖色


def _draw_food(draw: ImageDraw.ImageDraw, x: float, y: float, frame_idx: int, scale: float = 1.0) -> None:
    """absorb_growth専用: ランダムに出現する食べ物の描画。「触れると吸収可能になる」ことを
    プレイヤーの色とは無関係な目立つ暖色のダイヤモンド型+パルスで直感的に伝える。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.15 * math.sin(frame_idx * 0.25)
    r = FOOD_PICKUP_RADIUS * scale * pulse
    pts = [(zx, zy - r), (zx + r, zy), (zx, zy + r), (zx - r, zy)]
    draw.polygon(pts, fill=FOOD_COLOR, outline=(255, 255, 255))


def _draw_empowered_aura(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, frame_idx: int, scale: float = 1.0) -> None:
    """absorb_growth専用: 食べ物を取得して「吸収できる側」になっている間、プレイヤーの周囲に
    パルスするリングを表示する。誰が優位なのかを毎フレーム視覚的に明示するための演出。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.25 * abs(((frame_idx % 12) / 12) - 0.5) * 2
    r = radius * scale * 1.3 * pulse
    draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=FOOD_COLOR, width=max(2, int(4 * scale)))


PROTECTION_ZONE_COLOR = (120, 255, 170)  # area_control専用: 保護ゾーンの色(食べ物/銃/シールドと被らないミント色)


def _draw_protection_zone(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, frame_idx: int, scale: float = 1.0) -> None:
    """area_control専用: まだ誰も入っていない保護ゾーン(円形エリア)の描画。破線の円+中心の
    小さな盾マークで「安全地帯の外でも守られる」ことを直感的に伝える。"""
    zx, zy = _zoom_point((x, y), scale)
    r = radius * scale
    pulse = 1.0 + 0.08 * math.sin(frame_idx * 0.2)
    r *= pulse
    n_dashes = 16
    for i in range(n_dashes):
        a0 = (i / n_dashes) * 2 * math.pi
        a1 = a0 + (2 * math.pi / n_dashes) * 0.55
        draw.arc(
            [zx - r, zy - r, zx + r, zy + r],
            math.degrees(a0),
            math.degrees(a1),
            fill=PROTECTION_ZONE_COLOR,
            width=max(2, int(4 * scale)),
        )
    sr = r * 0.28
    draw.ellipse([zx - sr, zy - sr, zx + sr, zy + sr], outline=PROTECTION_ZONE_COLOR, width=max(1, int(2 * scale)))


def _draw_protection_cage(draw: ImageDraw.ImageDraw, shape: str, x: float, y: float, radius: float, scale: float = 1.0) -> None:
    """area_control専用: 保護ゾーンが実体化した後の物理的な囲い。固定座標(空間に固定、
    回転しない)に、外枠と同じshapeの塗りつぶし半透明+実線の囲いを描く(_cage_local_segments
    と同じ形状)。プレイヤーはこの中で跳ね返り続ける(2026-09-09、ユーザー訂正: 無敵で自由に
    動けるのではなく、物理的に閉じ込められる仕様)。"""
    zx, zy = _zoom_point((x, y), scale)
    r = radius * scale
    if shape == "square":
        draw.rectangle([zx - r, zy - r, zx + r, zy + r], outline=PROTECTION_ZONE_COLOR, width=max(2, int(5 * scale)))
    else:
        draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=PROTECTION_ZONE_COLOR, width=max(2, int(5 * scale)))


BARRIER_COLOR = (200, 160, 255)  # goal_reach専用: ゴールを囲むバリアの色(エネルギーシールド風)
GUN_COLOR = (255, 120, 90)  # gun_duel専用: 銃/弾の色
SHIELD_COLOR = (90, 220, 255)  # gun_duel専用: シールドの色


def _draw_goal_barrier(
    draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, hits: int, hits_to_break: int, scale: float = 1.0
) -> None:
    """goal_reach専用: ゴールを囲むバリア。ヒット数が増えるほど薄く&ひびが増えていく見た目にし、
    「あと何回でここを破壊できるか」が説明なしで伝わるようにする(2026-09-08追加)。"""
    zx, zy = _zoom_point((x, y), scale)
    r = radius * scale
    remaining_frac = max(0.0, (hits_to_break - hits) / hits_to_break)
    alpha = 0.35 + 0.65 * remaining_frac
    color = tuple(int(24 + (c - 24) * alpha) for c in BARRIER_COLOR)
    draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=color, width=max(2, int(5 * scale)))
    # ヒットのたびに1本ずつひびを描き足す(x,y固定シードなのでフレームをまたいでも同じ形で安定する)
    crack_rng = random.Random(int(x) * 100003 + int(y))
    for _ in range(hits):
        angle = crack_rng.uniform(0, 2 * math.pi)
        length = r * crack_rng.uniform(0.5, 0.95)
        ex, ey = zx + math.cos(angle) * length, zy + math.sin(angle) * length
        draw.line([(zx, zy), (ex, ey)], fill=color, width=max(1, int(2 * scale)))


# 2026-09-20、ユーザー指示: 「銃だとわかるように」見た目を改善。以前は横棒+縦棒だけの
# T字型で銃だと分かりにくかったため、スライド(銃身)・トリガーガード・グリップを
# 持つ側面シルエットの1本のポリゴンに描き直した(単位長さ1=銃身の半分の長さ、
# +x方向=銃口側)。Tabler Icons(既存の6ch目で使用中、materials/icons/tabler/)に
# gun/pistol相当のアイコンが無いことを確認済み(pistol/gun/weaponはいずれも404、
# sword/axe/bow/target/crosshairは実在)だったため、この用途は座標ベースの
# 自前シルエットで対応する。
GUN_SHAPE_POINTS = [
    (-0.65, -0.15),
    (1.00, -0.15),
    (1.00, 0.15),
    (-0.20, 0.15),
    (-0.20, 0.30),
    (-0.30, 0.30),
    (-0.30, 0.50),
    (-0.48, 0.50),
    (-0.48, 0.30),
    (-0.58, 0.15),
    (-0.78, 1.00),
    (-1.08, 0.95),
    (-1.08, 0.10),
]
GUN_MUZZLE_LOCAL = (1.0, 0.0)  # 銃口の位置(照準線の始点計算に使う)


def _gun_polygon(cx: float, cy: float, angle: float, size: float) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for px, py in GUN_SHAPE_POINTS:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        points.append((cx + rx * size, cy + ry * size))
    return points


def _draw_gun_pickup(draw: ImageDraw.ImageDraw, x: float, y: float, frame_idx: int, scale: float = 1.0) -> None:
    """gun_duel専用: まだ拾われていない銃のアイコン(地面に置かれている状態)。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.15 * math.sin(frame_idx * 0.25)
    size = FOOD_PICKUP_RADIUS * scale * pulse * 0.85
    draw.polygon(_gun_polygon(zx, zy, 0.0, size), fill=GUN_COLOR, outline=(255, 255, 255), width=max(1, int(2 * scale)))


def _draw_held_gun(
    draw: ImageDraw.ImageDraw, x: float, y: float, angle: float, radius: float, charge: float, scale: float = 1.0
) -> None:
    """gun_duel専用: 銃を保持中のプレイヤーに追従する銃(側面シルエット、狙っている方向へ回転)と、
    狙っている方向への破線の照準線。charge(0〜1、拾った直後〜発射直前)が進むほど照準線が
    伸び、緊張感を出す。"""
    zx, zy = _zoom_point((x, y), scale)
    size = radius * 0.95 * scale
    draw.polygon(
        _gun_polygon(zx, zy, angle, size), fill=GUN_COLOR, outline=(255, 255, 255), width=max(1, int(2 * scale))
    )
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    mx, my = GUN_MUZZLE_LOCAL
    bx = zx + (mx * cos_a - my * sin_a) * size
    by = zy + (mx * sin_a + my * cos_a) * size
    aim_len = (60 + charge * 220) * scale
    dash_len, gap_len = 14 * scale, 10 * scale
    dist = 0.0
    while dist < aim_len:
        seg_end = min(dist + dash_len, aim_len)
        p0 = (bx + math.cos(angle) * dist, by + math.sin(angle) * dist)
        p1 = (bx + math.cos(angle) * seg_end, by + math.sin(angle) * seg_end)
        draw.line([p0, p1], fill=GUN_COLOR, width=max(1, int(2 * scale)))
        dist += dash_len + gap_len


def _draw_bullet(draw: ImageDraw.ImageDraw, x: float, y: float, angle: float, scale: float = 1.0) -> None:
    """gun_duel専用: 飛行中の弾。進行方向に短い尾を引かせて速度感を出す。"""
    zx, zy = _zoom_point((x, y), scale)
    r = BULLET_RADIUS * scale
    tail_len = 26 * scale
    tx, ty = zx - math.cos(angle) * tail_len, zy - math.sin(angle) * tail_len
    draw.line([(tx, ty), (zx, zy)], fill=(255, 240, 200), width=max(2, int(4 * scale)))
    draw.ellipse([zx - r, zy - r, zx + r, zy + r], fill=(255, 250, 220), outline=GUN_COLOR)


def _draw_shield_pickup(draw: ImageDraw.ImageDraw, x: float, y: float, frame_idx: int, scale: float = 1.0) -> None:
    """gun_duel専用: まだ拾われていないシールドのアイコン(地面に置かれている状態)。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.15 * math.sin(frame_idx * 0.25)
    r = FOOD_PICKUP_RADIUS * scale * pulse
    pts = [
        (zx, zy - r),
        (zx + r, zy - r * 0.2),
        (zx + r * 0.6, zy + r),
        (zx - r * 0.6, zy + r),
        (zx - r, zy - r * 0.2),
    ]
    draw.polygon(pts, fill=SHIELD_COLOR, outline=(255, 255, 255))


def _draw_shield_aura(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, frame_idx: int, scale: float = 1.0) -> None:
    """gun_duel専用: シールド保持中のプレイヤーの周囲に表示するパルスするリング。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.2 * abs(((frame_idx % 14) / 14) - 0.5) * 2
    r = radius * scale * 1.35 * pulse
    draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=SHIELD_COLOR, width=max(2, int(4 * scale)))


# ============================================================
# weapon_colosseum専用の描画群(2026-09-20、ユーザー指摘で色分けバッジ+文字から刷新)。
# gun_duelの銃(GUN_SHAPE_POINTS)と同じ手法(ローカル座標のポリゴンを回転・拡大縮小して
# ワールド座標へ変換)で、各武器が「それだとわかる」シルエットになるよう専用形状を
# 用意する。+x方向を武器の先端(刃/打撃部)側とする共通ルール。
# ============================================================
WEAPON_COLORS = {
    "sword": (210, 210, 225),
    "spear": (200, 210, 225),
    "hammer": (150, 150, 160),
    "bow": (140, 255, 170),
    "axe": (220, 90, 90),
}
WEAPON_WOOD_COLOR = (140, 100, 60)


def _weapon_transform(cx: float, cy: float, angle: float, size: float):
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        return (cx + (lx * cos_a - ly * sin_a) * size, cy + (lx * sin_a + ly * cos_a) * size)

    return to_world


def _draw_sword_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, size: float) -> None:
    to_world = _weapon_transform(cx, cy, angle, size)
    blade = [to_world(x, y) for x, y in [(1.0, 0.0), (0.25, 0.05), (-0.30, 0.05), (-0.30, -0.05), (0.25, -0.05)]]
    draw.polygon(blade, fill=WEAPON_COLORS["sword"], outline=(255, 255, 255))
    guard = [to_world(x, y) for x, y in [(-0.28, 0.22), (-0.38, 0.22), (-0.38, -0.22), (-0.28, -0.22)]]
    draw.polygon(guard, fill=(150, 150, 160), outline=(255, 255, 255))
    grip = [to_world(x, y) for x, y in [(-0.33, 0.07), (-0.68, 0.07), (-0.68, -0.07), (-0.33, -0.07)]]
    draw.polygon(grip, fill=WEAPON_WOOD_COLOR, outline=(255, 255, 255))


def _draw_spear_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, size: float) -> None:
    """2026-09-20、ユーザー指摘対応: 旧版は柄が細すぎて(幅0.035)実機では穂先しか見えていな
    かった。柄を太く・長くし、はっきり「槍を持っている」とわかるようにした。"""
    to_world = _weapon_transform(cx, cy, angle, size)
    shaft = [to_world(x, y) for x, y in [(0.60, 0.075), (-1.35, 0.075), (-1.35, -0.075), (0.60, -0.075)]]
    draw.polygon(shaft, fill=WEAPON_WOOD_COLOR, outline=(255, 255, 255))
    head = [to_world(x, y) for x, y in [(1.20, 0.0), (0.50, 0.16), (0.50, -0.16)]]
    draw.polygon(head, fill=WEAPON_COLORS["spear"], outline=(255, 255, 255))


def _draw_hammer_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, size: float) -> None:
    """2026-09-21、ユーザー指示: 先端部の幅を広くし、T字型のシルエットにする。"""
    to_world = _weapon_transform(cx, cy, angle, size)
    shaft = [to_world(x, y) for x, y in [(0.45, 0.05), (-0.95, 0.05), (-0.95, -0.05), (0.45, -0.05)]]
    draw.polygon(shaft, fill=WEAPON_WOOD_COLOR, outline=(255, 255, 255))
    head = [to_world(x, y) for x, y in [(0.85, 0.55), (0.45, 0.55), (0.45, -0.55), (0.85, -0.55)]]
    draw.polygon(head, fill=WEAPON_COLORS["hammer"], outline=(255, 255, 255))


def _draw_bow_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, size: float) -> None:
    """弓は塗りつぶしポリゴンではなく弧(折れ線近似)+弦の直線で表現する。
    2026-09-21、ユーザー指摘対応: 旧版は弧の膨らみ(振幅0.15)が浅すぎて弓に見えにくかった
    ため、放物線状の式(x=0.35*(1-y^2))に変更しはっきりした「(」字カーブにした。"""
    to_world = _weapon_transform(cx, cy, angle, size)
    ys = [-1.0 + 2.0 * i / 12 for i in range(13)]
    arc_pts = [to_world(0.35 * (1 - y * y), y) for y in ys]
    draw.line(arc_pts, fill=WEAPON_WOOD_COLOR, width=max(2, int(size * 0.08)))
    draw.line([arc_pts[0], arc_pts[-1]], fill=(235, 235, 225), width=max(1, int(size * 0.03)))


def _draw_axe_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, size: float) -> None:
    """2026-09-21、ユーザー指示: 両刃(斧頭の上下に対称な刃)にする。"""
    to_world = _weapon_transform(cx, cy, angle, size)
    shaft = [to_world(x, y) for x, y in [(0.20, 0.05), (-0.85, 0.05), (-0.85, -0.05), (0.20, -0.05)]]
    draw.polygon(shaft, fill=WEAPON_WOOD_COLOR, outline=(255, 255, 255))
    head_top = [to_world(x, y) for x, y in [(0.20, 0.05), (0.55, 0.45), (0.10, 0.55), (-0.20, 0.28), (-0.05, 0.05)]]
    head_bottom = [to_world(x, y) for x, y in [(0.20, -0.05), (0.55, -0.45), (0.10, -0.55), (-0.20, -0.28), (-0.05, -0.05)]]
    draw.polygon(head_top, fill=WEAPON_COLORS["axe"], outline=(255, 255, 255))
    draw.polygon(head_bottom, fill=WEAPON_COLORS["axe"], outline=(255, 255, 255))


WEAPON_DRAW_FUNCS = {
    "sword": _draw_sword_shape,
    "spear": _draw_spear_shape,
    "hammer": _draw_hammer_shape,
    "bow": _draw_bow_shape,
    "axe": _draw_axe_shape,
}


def _draw_weapon_pickup(draw: ImageDraw.ImageDraw, kind: str, x: float, y: float, frame_idx: int, scale: float = 1.0) -> None:
    """weapon_colosseum専用: まだ拾われていない武器(地面でゆっくり回転しながらパルスする)。"""
    zx, zy = _zoom_point((x, y), scale)
    pulse = 1.0 + 0.1 * math.sin(frame_idx * 0.15)
    size = WEAPON_PICKUP_RADIUS * scale * 1.7 * pulse
    angle = frame_idx * 0.03
    draw_func = WEAPON_DRAW_FUNCS.get(kind)
    if draw_func:
        draw_func(draw, zx, zy, angle, size)


WEAPON_HELD_SIZE_MULT = {
    "spear": 2.0,  # 2026-09-20、ユーザー指摘: 「槍を持てるよう伸ばしてください」への対応
    "axe": 1.7,  # 2026-09-21、ユーザー指示: 「斧そのもののサイズは大きくしてください」への対応
}
SPEAR_FACING_MIN_SPEED = 20.0 * _ARENA_SCALE  # この速さ未満はほぼ静止とみなし、狙っている方向のままにする

# 2026-09-20、ユーザー指示: ハンマーも剣と同様にプレイヤーの周りを周回させ(「回転させて
# 持つ」)、剣より大きめの周回半径で「より前に出す」(見た目の存在感を強くする)。
WEAPON_SPIN_KINDS = {"sword", "hammer"}
WEAPON_SPIN_ORBIT_MULT = {"sword": 1.5, "hammer": 2.3}
WEAPON_SPIN_SIZE_MULT = {"sword": 0.85, "hammer": 0.95}


def _draw_held_weapon(
    draw: ImageDraw.ImageDraw,
    kind: str,
    x: float,
    y: float,
    radius: float,
    angle: float,
    vx: float,
    vy: float,
    frame_idx: int,
    scale: float = 1.0,
    axe_spinning: bool = False,
) -> None:
    """weapon_colosseum専用: 保持中の武器。プレイヤーの脇に表示する。
    2026-09-20、ユーザー指摘: 槍だけは狙っている相手の方向(angle)ではなく、
    自身が進んでいる方向(vx,vy)へ向ける(移動がほぼ止まっている場合はangleにフォールバック)。
    剣・ハンマーは指示書/ユーザー指示「回転しながら攻撃」に対応し、プレイヤーの周りを周回する
    (ハンマーは剣より大きな周回半径で、より前に出て見えるようにする)。
    2026-09-21、ユーザー指示で斧を全面刷新(3回目、最終版): 基本は静止(他の非スピン武器と
    同じ、狙っている相手側へ少し前に出た構え)、攻撃中(axe_spinning)だけ剣/ハンマーより
    さらに速い高速回転にする。"""
    zx, zy = _zoom_point((x, y), scale)
    size = radius * WEAPON_HELD_SIZE_MULT.get(kind, 1.3) * scale
    draw_func = WEAPON_DRAW_FUNCS.get(kind)
    if draw_func is None:
        return
    if kind == "axe" and axe_spinning:
        orbit_r = radius * AXE_HELD_OFFSET_MULT * scale
        spin_angle = frame_idx * DT * AXE_SPIN_SPEED
        bx, by = zx + math.cos(spin_angle) * orbit_r, zy + math.sin(spin_angle) * orbit_r
        draw_func(draw, bx, by, spin_angle, size)
        return
    if kind in WEAPON_SPIN_KINDS:
        orbit_r = radius * WEAPON_SPIN_ORBIT_MULT.get(kind, 1.5) * scale
        spin_angle = frame_idx * DT * SWORD_SPIN_SPEED
        bx, by = zx + math.cos(spin_angle) * orbit_r, zy + math.sin(spin_angle) * orbit_r
        draw_func(draw, bx, by, spin_angle, size * WEAPON_SPIN_SIZE_MULT.get(kind, 0.85))
        return
    face_angle = angle
    if kind == "spear" and math.hypot(vx, vy) > SPEAR_FACING_MIN_SPEED:
        face_angle = math.atan2(vy, vx)
    offset_mult = AXE_HELD_OFFSET_MULT if kind == "axe" else 1.0
    offset = radius * offset_mult * scale
    bx, by = zx + math.cos(face_angle) * offset, zy + math.sin(face_angle) * offset
    draw_func(draw, bx, by, face_angle, size)


def _draw_arrow(draw: ImageDraw.ImageDraw, x: float, y: float, vx: float, vy: float, scale: float = 1.0) -> None:
    """weapon_colosseum(弓矢)専用: 飛行中の矢。進行方向(重力で刻々と変わる)に向けた
    短い線+矢尻で表現する。"""
    zx, zy = _zoom_point((x, y), scale)
    angle = math.atan2(vy, vx)
    length = 30 * scale
    tx, ty = zx - math.cos(angle) * length, zy - math.sin(angle) * length
    draw.line([(tx, ty), (zx, zy)], fill=(230, 220, 190), width=max(2, int(3 * scale)))
    r = ARROW_RADIUS * scale
    draw.ellipse([zx - r, zy - r, zx + r, zy + r], fill=WEAPON_COLORS["bow"])


def _draw_hp_bar(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, hp: int, max_hp: int, scale: float = 1.0) -> None:
    """weapon_colosseum専用: プレイヤー頭上のHP表示(指示書「体力の表記」への対応)。
    バーの色は残量に応じて緑→黄→赤に変化させ、直感的に状況を伝える。"""
    zx, zy = _zoom_point((x, y), scale)
    bar_w = radius * 2.0 * scale
    bar_h = max(3, int(6 * scale))
    top = zy - radius * scale - bar_h - 10 * scale
    left = zx - bar_w / 2
    frac = max(0.0, min(1.0, hp / max_hp))
    color = (90, 220, 110) if frac > 0.6 else (240, 210, 70) if frac > 0.3 else (235, 70, 70)
    draw.rectangle([left, top, left + bar_w, top + bar_h], fill=(40, 40, 46), outline=(15, 15, 18))
    if frac > 0:
        draw.rectangle([left, top, left + bar_w * frac, top + bar_h], fill=color)


_CAPTION_FONT = None
_SUB_FONT = None
_COUNTER_FONT = None


def _load_bold_font(size: int):
    """OS依存のフォント名解決の対策(2026-09-08発見の重大バグ): GitHub Actions(Ubuntu)には
    Windows専用の"arialbd.ttf"が存在せず、常に例外→`ImageFont.load_default()`(サイズ指定不可の
    固定小フォント)にフォールバックしていた。そのためフォントサイズをいくら大きくしても、
    ローカル(Windows)では反映されて見えるのに実際の本番動画(CI上でレンダリング)には
    一切反映されていなかった。Linuxでよく入っているDejaVu Sans Boldも候補に加えたうえで、
    最終的にPillow同梱のサイズ指定可能なデフォルトフォント(`load_default(size=...)`、
    OSに一切依存せず必ず使える)に確実にフォールバックすることで、どの環境でも指定サイズが
    確実に反映されるようにする。"""
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _get_font(size: int, cache_attr: str):
    global _CAPTION_FONT, _SUB_FONT, _COUNTER_FONT
    cache = {"_CAPTION_FONT": _CAPTION_FONT, "_SUB_FONT": _SUB_FONT, "_COUNTER_FONT": _COUNTER_FONT}
    if cache[cache_attr] is not None:
        return cache[cache_attr]
    font = _load_bold_font(size)
    if cache_attr == "_CAPTION_FONT":
        _CAPTION_FONT = font
    elif cache_attr == "_SUB_FONT":
        _SUB_FONT = font
    else:
        _COUNTER_FONT = font
    return font


def _draw_hud(img: Image.Image, rule: str, remaining_count: int | None) -> None:
    """画面上部に、半透明バーでルールの英語キャプションと残数を表示する。
    完全不透明にすると上部にいるプレイヤーが隠れてしまうため半透明にしている。
    """
    # 2026-09-08、ユーザー指摘によりフォントサイズを拡大(34→46, 24→32)。読みやすさ優先だが
    # 画面の邪魔にならないよう、バー自体の高さ(bar_h)と半透明度は維持したまま調整した。
    bar_h = 128
    overlay = Image.new("RGBA", (int(ARENA_W), bar_h), (10, 10, 14, 175))
    odraw = ImageDraw.Draw(overlay)

    caption = RULE_CAPTIONS.get(rule, "")
    font_cap = _get_font(46, "_CAPTION_FONT")
    bbox = odraw.textbbox((0, 0), caption, font=font_cap)
    tw = bbox[2] - bbox[0]
    odraw.text(((ARENA_W - tw) / 2, 12), caption, fill=(240, 240, 240, 255), font=font_cap)

    if remaining_count is not None:
        sub = f"{remaining_count} LEFT"
        font_sub = _get_font(32, "_SUB_FONT")
        bbox2 = odraw.textbbox((0, 0), sub, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        odraw.text(((ARENA_W - tw2) / 2, 74), sub, fill=(190, 190, 200, 255), font=font_sub)

    img.paste(overlay, (0, 0), overlay)


def _draw_crown(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, scale: float = 1.0) -> None:
    """現在の首位(スコア/サイズ)を示す、頭上に浮かぶフラットスタイルの王冠ポリゴン。

    2026-09-20、ユーザー指示: 「誰が有利か」を常時示す視覚サインを置き、最後まで見届ける
    動機付けにする。指標(キル数またはサイズ)が明確に定義できるルールにのみ適用する方針とし、
    現状は現在の半径=サイズがそのまま指標になる`absorb_growth`にのみ使う(呼び出し側で
    ルールを判定する)。hole_fall/goal_reach/area_control/gun_duelには「キル数」に相当する
    自然な指標が無く、安易な代理指標(生存順位等)を導入すると誤解を招くため見送っている。
    """
    zx, zy = _zoom_point((x, y), scale)
    zr = radius * scale
    crown_w, crown_h, gap = zr * 1.3, zr * 0.55, zr * 0.3
    bottom_y = zy - zr - gap
    top_y = bottom_y - crown_h
    band_h = crown_h * 0.35
    left_x, right_x = zx - crown_w / 2, zx + crown_w / 2
    peak_w = crown_w / 3
    points = [
        (left_x, bottom_y),
        (left_x, bottom_y - band_h),
        (left_x + peak_w * 0.5, top_y),
        (left_x + peak_w * 1.0, bottom_y - band_h * 0.5),
        (left_x + peak_w * 1.5, top_y),
        (left_x + peak_w * 2.0, bottom_y - band_h * 0.5),
        (left_x + peak_w * 2.5, top_y),
        (right_x, bottom_y - band_h),
        (right_x, bottom_y),
    ]
    outline_w = max(1, int(2 * scale))
    draw.polygon(points, fill=(255, 215, 0), outline=(120, 80, 0), width=outline_w)


def _draw_player(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: float,
    color: tuple,
    scale: float = 1.0,
    player_shape: str = "circle",
    angle: float = 0.0,
    alpha: float = 1.0,
) -> None:
    """alpha(2026-09-08追加): エピローグで脱落済みプレイヤーをフェードインさせるための不透明度
    (0=完全に見えない〜1=通常表示)。背景色(24,24,28)へ向けて色を線形補間することで実現する
    (Pillowの`ImageDraw`は単体でアルファ合成しないため、既存のability effectの手法を流用)。"""
    if alpha <= 0.0:
        return
    bg = (24, 24, 28)
    fill_color = color if alpha >= 1.0 else tuple(int(bg_c + (c - bg_c) * alpha) for bg_c, c in zip(bg, color))
    outline_color = (255, 255, 255) if alpha >= 1.0 else tuple(int(bg_c + (c - bg_c) * alpha) for bg_c, c in zip(bg, (255, 255, 255)))
    zx, zy = _zoom_point((x, y), scale)
    zr = radius * scale
    outline_w = max(1, int(3 * scale))
    if player_shape == "circle":
        draw.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=fill_color, outline=outline_color, width=outline_w)
        return
    verts = _poly_vertices(player_shape, radius)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # 頂点をangleだけ回転させ、プレイヤー中心(zx,zy)を基準にscale倍して配置する
    points = []
    for lx, ly in verts:
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        points.append((zx + rx * scale, zy + ry * scale))
    draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_w)


EYE_BLINK_PERIOD_FRAMES = int(4.0 * FPS)  # 自然なまばたきの周期(個体ごとに位相をずらす)
EYE_BLINK_DURATION_FRAMES = 6  # まばたきで目を閉じているフレーム数
EXPRESSION_DIZZY_SPEED = 560.0 * _ARENA_SCALE  # この速さを超えたら「目が回る」表情にする


def _draw_dizzy_eye(draw: ImageDraw.ImageDraw, ex: float, ey: float, r: float, frame_idx: int, side: int) -> None:
    """激しく動いている時の「目が回る」表情。白目+回転する渦巻き状の弧2本で表現する。"""
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(255, 255, 255))
    spin = (frame_idx * 22 * side) % 360
    width = max(1, int(r * 0.4))
    draw.arc([ex - r, ey - r, ex + r, ey + r], spin, spin + 130, fill=(20, 20, 24), width=width)
    draw.arc([ex - r, ey - r, ex + r, ey + r], spin + 180, spin + 310, fill=(20, 20, 24), width=width)


def _draw_worried_eye(draw: ImageDraw.ImageDraw, ex: float, ey: float, r: float, side: int) -> None:
    """ピンチの時の「苦しそうな」表情。ハの字の困り眉+小さめの瞳。"""
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(255, 255, 255))
    pupil_r = r * 0.45
    draw.ellipse([ex - pupil_r, ey - pupil_r + r * 0.2, ex + pupil_r, ey + pupil_r + r * 0.2], fill=(20, 20, 24))
    brow_half = r * 1.15
    inner_y, outer_y = ey - r * 1.35, ey - r * 0.75
    if side < 0:
        draw.line([(ex - brow_half, outer_y), (ex + brow_half, inner_y)], fill=(20, 20, 24), width=max(1, int(r * 0.3)))
    else:
        draw.line([(ex - brow_half, inner_y), (ex + brow_half, outer_y)], fill=(20, 20, 24), width=max(1, int(r * 0.3)))


def _draw_happy_eye(draw: ImageDraw.ImageDraw, ex: float, ey: float, r: float) -> None:
    """順調な時の「嬉しそうな」表情。下に凸の弧(細めの笑い目)。"""
    width = max(2, int(r * 0.45))
    draw.arc([ex - r, ey - r * 0.2, ex + r, ey + r * 1.5], 195, 345, fill=(20, 20, 24), width=width)


FACE_DIRECTION_MIN_SPEED = 15.0 * _ARENA_SCALE  # この速さ未満は直近の向きを保持する(停止直後に顔が一瞬で戻らないように)
FACE_DEFAULT_ANGLE = -math.pi / 2  # 一度も動いていない初期状態の既定の向き(画面上向き)


def _draw_player_eyes(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: float,
    scale: float,
    entity_id: int,
    frame_idx: int,
    collision_blink: bool,
    expression: str = "neutral",
    face_angle: float = FACE_DEFAULT_ANGLE,
) -> None:
    """8-2「ミニマルアクセサリー」: 表情を持たせるための最小限の目(2026-09-20実装)。
    普段は個体ごとに位相をずらした自然な周期でまばたきし、衝突の瞬間(画面シェイクが
    発生したフレーム)には追加でまばたきする(「衝突でまばたき」指示への対応)。
    collisionログにエンティティID/座標が記録されていないため、シェイク発生フレームを
    全プレイヤー共通のトリガーとして使う簡易実装(誰と誰がぶつかったかまでは反映されない)。
    小さすぎる図形(吸収されて縮んだ等)には描かない。

    2026-09-20、ユーザー指摘対応: 目を一回り大きく、状況に応じた表情
    (expression: "dizzy"|"worried"|"happy"|"neutral")を追加した。まばたき中はどの表情でも
    一律で閉じ目にする(表情より優先)。
    2026-09-20追記、ユーザー指摘対応: 「目が常にこちら(カメラ)を向いていて不気味」との
    指摘を受け、目のペアを画面に対して固定配置するのをやめ、進んでいる方向(face_angle、
    球体が向きを変えたと見立てる)側へ寄せて配置するようにした(黒目の向きではなく
    「頭の向き」自体を変える設計)。
    2026-09-21、ユーザー指摘で再修正: 目を並べる軸をface_angleに直交させて回転させると、
    移動方向によっては両目が縦に並んでしまい不自然だった。「両目の位置関係はずっと横に
    ある(頭頂部に対して同じ向き)」という指示に沿い、目を並べる軸は常に画面の水平方向で
    固定し、ペアの中心位置だけがface_angle方向(進行方向)へ寄る設計に変更した。"""
    zx, zy = _zoom_point((x, y), scale)
    zr = radius * scale
    if zr < 10 * scale:
        return
    phase = (entity_id * 37) % EYE_BLINK_PERIOD_FRAMES
    idle_blink = (frame_idx + phase) % EYE_BLINK_PERIOD_FRAMES < EYE_BLINK_DURATION_FRAMES
    closed = idle_blink or collision_blink
    cos_f, sin_f = math.cos(face_angle), math.sin(face_angle)
    pair_cx = zx + cos_f * zr * 0.30  # 進行方向側へ「顔」ごと寄せる(水平・垂直とも)
    pair_cy = zy + sin_f * zr * 0.30
    eye_spread = zr * 0.42
    eye_r = max(2.0, zr * 0.22)  # 一回り大きく(旧0.16)
    for side in (-1, 1):
        ex, ey = pair_cx + eye_spread * side, pair_cy  # 目を並べる軸は常に水平固定(頭頂部基準)
        if closed:
            draw.line([(ex - eye_r, ey), (ex + eye_r, ey)], fill=(20, 20, 24), width=max(1, round(2 * scale)))
        elif expression == "dizzy":
            _draw_dizzy_eye(draw, ex, ey, eye_r, frame_idx, side)
        elif expression == "worried":
            _draw_worried_eye(draw, ex, ey, eye_r, side)
        elif expression == "happy":
            _draw_happy_eye(draw, ex, ey, eye_r)
        else:
            draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(255, 255, 255))
            pupil_r = eye_r * 0.5
            draw.ellipse([ex - pupil_r, ey - pupil_r, ex + pupil_r, ey + pupil_r], fill=(20, 20, 24))


def _player_expression(entry: dict, rule: str, frame_state: list) -> str:
    """状況に応じた表情を1つ返す('dizzy'|'worried'|'happy'|'neutral')。
    優先順位: dizzy(激しい移動、全ルール共通) > worried(ピンチ) > happy(順調) > neutral。
    「ピンチ/順調」の自然な指標があるルール(weapon_colosseumのHP、absorb_growthの
    相対サイズ)のみ判定する。hole_fall/goal_reach/gun_duelはそうした自然な指標を持たない
    ため(代理指標を導入すると誤解を招く、_draw_crownと同じ判断基準)dizzy/neutralのみ扱う。"""
    speed = math.hypot(entry.get("vx", 0.0), entry.get("vy", 0.0))
    if speed > EXPRESSION_DIZZY_SPEED:
        return "dizzy"
    if rule == "weapon_colosseum":
        hp = entry.get("hp")
        if hp is not None:
            frac = hp / WEAPON_STARTING_HP
            if frac <= 0.3:
                return "worried"
            if frac >= 0.9:
                return "happy"
    elif rule == "absorb_growth" and len(frame_state) > 1:
        max_r = max(e["radius"] for e in frame_state)
        if entry["radius"] <= max_r * 0.55:
            return "worried"
        if entry["radius"] >= max_r * 0.95:
            return "happy"
    return "neutral"


def _build_shake_by_frame(collisions: list, elimination_order: list | None = None) -> dict[int, float]:
    """8-3: 衝突の衝撃(impulse)に応じたスクリーンシェイクの強度(0-1)をフレームごとに算出する。
    同一フレームに複数の衝突が重なった場合は最大値を採用する。

    2026-09-20、ユーザー指示(撃破インパクト音とスクリーンシェイクの完全同期): 脱落
    (`elimination_order`)は穴落下・ゾーン外判定等、物理衝突(impulse)を伴わないケースも
    多く、従来は脱落の瞬間に必ずしもシェイクが起きていなかった。脱落フレームには原因を
    問わず常に最大強度(1.0)のシェイクを追加保証する(衝突由来のシェイクとは`max()`で合成)。
    """
    shake_by_frame: dict[int, float] = {}
    if collisions:
        max_impulse = max(c["impulse"] for c in collisions)
        for c in collisions:
            intensity = c["impulse"] / max(max_impulse, 1.0)
            for age in range(SHAKE_DURATION_FRAMES):
                f = c["frame"] + age
                decay = 1 - age / SHAKE_DURATION_FRAMES
                shake_by_frame[f] = max(shake_by_frame.get(f, 0.0), intensity * decay)
    for ev in elimination_order or []:
        for age in range(SHAKE_DURATION_FRAMES):
            f = ev["frame"] + age
            decay = 1 - age / SHAKE_DURATION_FRAMES
            shake_by_frame[f] = max(shake_by_frame.get(f, 0.0), decay)
    return shake_by_frame


# 2026-09-20、冒頭演出強化(ユーザー指示): 開始時にカメラを軽くズームした状態から通常表示
# (1.0倍)へ引くことで、開幕の「引き込み」を強める。常時追従の`camera="tracking"`とは異なり、
# 開始直後だけの一回限りの決定論的な変形なので、2026-09-08に廃止したtrackingカメラの
# 不具合(枠回転との整合等)は再発しない(単純な中央クロップ+リサイズのみ)。
#
# 2026-09-20〜21、二転三転した経緯: 当初1.4→1.8倍+放射ブラーまで強化したが「ゲーム画面だと
# 認識される前に離脱している」との指摘を受け1.18倍+弱いブラーへマイルド化。2026-09-21、
# ユーザー指示「スワイプ離脱(広告誤認・無風)防止」で最終的にブラーそのものを撤廃し、
# 「1.2倍→1.0倍への単純なズームのみ」に統一した(視認性最優先。フラッシュ/暗転等の
# マスキング演出も一切使わない方針)。
OPENING_ZOOM_SECONDS = 0.5
OPENING_ZOOM_START = 1.2

# シームレスループ(エピローグ末尾→次ループ冒頭の繋ぎ)。開幕ズームと真逆の動き(1.0倍→
# OPENING_ZOOM_START倍へのズームインのみ)をかけ、動画の最終フレームの状態を冒頭の
# frame_idx=0の状態(zoom=OPENING_ZOOM_START)に一致させる。2026-09-21、ユーザー指示で
# ブラー・継ぎ目フラッシュ(光過敏性配慮のため元々alphaを抑えていたもの)を撤廃したため、
# キャラクターの座標リセットを隠す役割はエピローグの「初期位置へ滑らかに戻る」既存の
# 移動(travel_progress、2026-09-12の元々の設計)のみに委ねる形に戻った。
LOOP_TRANSITION_SECONDS = 0.5


def _crop_zoom(img: Image.Image, zoom: float) -> Image.Image:
    """画像中心を基準に、zoom倍にクロップ+リサイズする(zoom>1で拡大=ズームイン)。
    複数箇所(開幕ズーム・area_controlの緊張ズーム等)で使う共通ヘルパー。"""
    w, h = img.size
    crop_w, crop_h = w / zoom, h / zoom
    cx, cy = w / 2, h / 2
    box = (cx - crop_w / 2, cy - crop_h / 2, cx + crop_w / 2, cy + crop_h / 2)
    return img.crop(box).resize((w, h), Image.BILINEAR)


def _apply_opening_zoom(img: Image.Image, frame_idx: int) -> Image.Image:
    """開始OPENING_ZOOM_SECONDS秒でズームOPENING_ZOOM_START倍→1.0倍(Ease-Out Cubic)。
    ブラー等の演出は一切かけない(視認性最優先)。バトル本体(HUDより前)にのみ適用するため、
    呼び出し側は`_draw_hud`より前に呼ぶこと。"""
    t = frame_idx / FPS
    if t >= OPENING_ZOOM_SECONDS:
        return img
    p = t / OPENING_ZOOM_SECONDS
    eased = 1 - (1 - p) ** 3
    zoom = OPENING_ZOOM_START - (OPENING_ZOOM_START - 1.0) * eased
    if zoom <= 1.001:
        return img
    return _crop_zoom(img, zoom)


def _apply_loop_reverse_zoom(img: Image.Image, phase_t: float) -> Image.Image:
    """エピローグ終盤LOOP_TRANSITION_SECONDS秒で、_apply_opening_zoomと真逆の変化
    (1.0倍→OPENING_ZOOM_START倍へズームイン、Ease-In)をかける。phase_t=1.0
    (エピローグ最終フレーム)で、_apply_opening_zoomのframe_idx=0の状態
    (zoom=OPENING_ZOOM_START)と完全に一致する設計。エピローグのWINテキスト描画より前
    (_apply_opening_zoomがHUDより前に呼ばれるのと同じ理由で、テキストまでズームされて
    しまうのを防ぐため)に呼ぶこと。"""
    start = 1.0 - LOOP_TRANSITION_SECONDS / EPILOGUE_SECONDS
    if phase_t <= start:
        return img
    p = (phase_t - start) / (1.0 - start)
    eased = p**3  # Ease-In cubic(冒頭のEase-Out cubicと対称)
    zoom = 1.0 + (OPENING_ZOOM_START - 1.0) * eased
    if zoom <= 1.001:
        return img
    return _crop_zoom(img, zoom)


# 2026-09-20、ユーザー指示(area_controlの緊張感強化): 安全地帯が縮むにつれ、カメラも
# 少しずつ中央へ寄せる。決着(ZONE_SHRINK_SECONDS経過)に近づくほど寄り方が加速する
# イージングにし、「じわじわ追い詰められる」感覚を強める。開幕ズーム(0.8秒だけ)とは
# 独立した、試合時間全体に渡るゆっくりとした変形。
AREA_CONTROL_TENSION_MAX_ZOOM = 1.55


def _area_control_tension_zoom(img: Image.Image, frame_idx: int) -> Image.Image:
    t = frame_idx / FPS
    frac = min(1.0, t / ZONE_SHRINK_SECONDS)
    eased = frac**1.6  # 終盤にかけて寄る速度が上がる
    zoom = 1.0 + (AREA_CONTROL_TENSION_MAX_ZOOM - 1.0) * eased
    if zoom <= 1.001:
        return img
    return _crop_zoom(img, zoom)


def _apply_screen_shake(img: Image.Image, seed: int, frame_idx: int, intensity: float, scale: float = 1.0) -> Image.Image:
    """8-3: 軽いスクリーンシェイク。結果に一切影響しない純粋な演出のため、シミュレーション用の
    乱数列とは独立した(seed, frame_idx)由来の決定論的な乱数でオフセットを決める(再レンダリング時も
    同じ絵になるようにするため)。タプルをrandom.Random()へ直接渡すとPython 3.11でTypeErrorになる
    (3.10まではhash()フォールバックで動いていたが3.11で廃止された。CI実行のPython 3.11で実際に
    発生・修正した)ため、整数演算だけで決定論的なシードを作る。"""
    seed_val = ((seed % 1_000_003) * 1_000_003 + frame_idx) & 0xFFFFFFFF
    rng = random.Random(seed_val)
    magnitude = SHAKE_MAX_OFFSET * intensity * scale
    dx = round(rng.uniform(-1, 1) * magnitude)
    dy = round(rng.uniform(-1, 1) * magnitude)
    if dx == 0 and dy == 0:
        return img
    shaken = Image.new("RGB", img.size, (24, 24, 28))
    shaken.paste(img, (dx, dy))
    return shaken


def _draw_decision_burst(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """8-3: 決着の瞬間、勝者の位置に一瞬だけ出す色フラッシュ+飛び散るパーティクル。
    結果には一切影響しない純粋な演出(4章の判定ロジックとは独立)。"""
    alpha = max(0.0, 1 - age / DECISION_BURST_FRAMES)
    blended = tuple(int(bg + (col - bg) * alpha) for bg, col in zip((24, 24, 28), color))

    flash_r = (40 + age * 10) * scale
    draw.ellipse(
        [x - flash_r, y - flash_r, x + flash_r, y + flash_r], outline=blended, width=max(1, int(6 * alpha))
    )

    progress = age / DECISION_BURST_FRAMES
    for i in range(DECISION_PARTICLE_COUNT):
        theta = (2 * math.pi / DECISION_PARTICLE_COUNT) * i
        dist = progress * 110 * scale
        px, py = x + math.cos(theta) * dist, y + math.sin(theta) * dist
        pr = max(1.0, 5 * alpha * scale)
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=blended)


def _draw_shockwave_ring(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple, progress: float, scale: float, max_radius: float) -> None:
    """2026-09-12、エピローグ全面改修: 勝者確定の瞬間、勝者の位置から外枠まで一気に広がる
    波紋(勝者カラーの薄いリング)。全画面フラッシュ(光感受性発作のリスクがある)を使わずに
    視聴者の視線を「中央の勝者」から「外枠全体(俯瞰)」へ自然に誘導するための演出で、
    波紋が広がりきってフェードアウトした直後に全プレイヤーの初期配置リセットを行う
    (呼び出し側)ことで、位置の不連続を感じさせにくくする。"""
    alpha = max(0.0, 1.0 - progress)
    if alpha <= 0.0:
        return
    blended = tuple(int(bg + (col - bg) * alpha) for bg, col in zip((24, 24, 28), color))
    radius = progress * max_radius
    width = max(1, int(EPILOGUE_RING_WIDTH * scale))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=blended, width=width)


def _draw_elimination_effects(draw: ImageDraw.ImageDraw, effects: list, scale: float = 1.0) -> None:
    """脱落した瞬間の位置に、数フレームだけ拡散して消えるリング状のエフェクトを描く。
    脱落座標が描画キャンバスの外まで到達していることがあるため、キャンバス内にクランプして
    必ず見えるようにする(実際の脱落点の近似表示になる)。
    """
    for x, y, color, age, base_radius in effects:
        dx = min(max(x, 0), ARENA_W)
        dy = min(max(y, 0), ARENA_H)
        zx, zy = _zoom_point((dx, dy), scale)
        alpha = max(0.0, 1 - age / EFFECT_FRAMES)
        r = (base_radius * 0.6 + age * 4) * scale
        blended = tuple(int(bg + (col - bg) * alpha) for bg, col in zip((24, 24, 28), color))
        width = max(1, int(6 * alpha))
        draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=blended, width=width)


ABILITY_EFFECT_COLOR = {
    "shockwave": (255, 90, 70),
    "vortex": (140, 210, 255),
    "growth_surge": (110, 230, 150),
    "dash": (255, 220, 90),
    "poison_wire": (190, 110, 230),
    "teleport": (120, 230, 220),
    "speed_boost": (255, 150, 60),
    "slam": (190, 190, 200),
    # absorb_growthの「かじり取り」(2026-09-08追加)。特殊能力ではないがability_events/
    # ability_effects_by_frameの汎用フレームワークをそのまま流用する。食べ物と同じ暖色にして
    # 「食べ物/吸収」まわりの演出であることが視覚的に一貫するようにしている。
    "partial_absorb_grow": FOOD_COLOR,
    "partial_absorb_shrink": FOOD_COLOR,
    "pinball_kick": PINBALL_COLOR,
    "pin_hit": PIN_WALL_COLOR,
    "flipper_kick": FLIPPER_COLOR,
    "weapon_hit": (255, 210, 60),  # weapon_colosseum専用: 命中の瞬間の小さなフラッシュ
}


def _draw_target_reticle(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, target_radius: float, scale: float = 1.0) -> None:
    """dash発動時、突進先のターゲットに表示する照準マーク(カメラのオートフォーカス風の四隅ブラケット)。
    「何かが自分を狙っている」ことが説明なしで伝わるよう、リング状の汎用エフェクトとは別の形状にしている。

    2026-09-08、ユーザー指摘対応: gap/lengthが固定ピクセル値(15)だったため、CIRCLE_RADIUSが
    35→25に縮小された後はブラケットの大部分がプレイヤー本体の陰に隠れ、ほぼ見えなくなっていた。
    target_radiusを基準にgapを決めることで、プレイヤーのサイズ(growth_surge等での変化も含む)に
    関わらず、常に本体の外側にはっきり見えるようにする。
    """
    gap = (target_radius + 10) * scale
    length = 18 * scale
    width = max(1, int(3 * scale))
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx, cy = zx + sx * gap, zy + sy * gap
        draw.line([(cx, cy), (cx + sx * length, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + sy * length)], fill=color, width=width)


def _draw_slam_mark(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """slam発動時に表示する、急降下を示す二重の下向き矢印(ゆっくり下に流れながらフェードアウトする)。"""
    drift = (age / ABILITY_EFFECT_FRAMES) * 24 * scale
    size = 16 * scale
    width = max(1, int(4 * scale))
    for dy in (-1.0, 0.5):
        cy = zy + dy * size + drift
        draw.line([(zx - size, cy - size * 0.6), (zx, cy + size * 0.6)], fill=color, width=width)
        draw.line([(zx + size, cy - size * 0.6), (zx, cy + size * 0.6)], fill=color, width=width)


def _draw_growth_mark(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """growth_surge発動時に表示する、中心から外向きの拡張矢印(4隅、「巨大化」を示す)。"""
    progress = age / ABILITY_EFFECT_FRAMES
    inner = (18 + progress * 20) * scale
    length = 16 * scale
    width = max(1, int(4 * scale))
    for dx, dy in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
        bx, by = zx + dx * inner, zy + dy * inner
        tip = (bx + dx * length, by + dy * length)
        draw.line([(bx, by), tip], fill=color, width=width)
        perp = (-dy, dx)
        for sign in (-1, 1):
            wing = (tip[0] - dx * length * 0.5 + sign * perp[0] * length * 0.35, tip[1] - dy * length * 0.5 + sign * perp[1] * length * 0.35)
            draw.line([wing, tip], fill=color, width=width)


def _draw_hazard_mark(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """poison_wire設置の瞬間に表示する、注意喚起のジグザグマーク。"""
    drift = (age / ABILITY_EFFECT_FRAMES) * 6 * scale
    size = 14 * scale
    width = max(1, int(4 * scale))
    pts = [
        (zx - size, zy - size + drift),
        (zx - size * 0.3, zy + size * 0.2 + drift),
        (zx + size * 0.3, zy - size * 0.3 + drift),
        (zx + size, zy + size + drift),
    ]
    draw.line(pts, fill=color, width=width, joint="curve")


def _draw_speed_lines(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """speed_boost発動時に表示する、後方に流れる速度線(加速したことを示す)。
    3本を縦方向にずらして描くことで、束になった1本の線に見えないようにしている。"""
    progress = age / ABILITY_EFFECT_FRAMES
    line_len = (10 + progress * 50) * scale
    width = max(1, int(3 * scale))
    for row, offset in enumerate((-14, 0, 14)):
        oy = offset * scale
        ox = (row - 1) * 6 * scale  # 中央の線を少し前に、両端を少し後ろにずらして矢羽根っぽくする
        draw.line([(zx + ox, zy + oy), (zx + ox - line_len, zy + oy)], fill=color, width=width)


def _draw_nibble_grow_mark(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """absorb_growthの「かじり取り」(2026-09-08追加)で成長した側に表示する、外向きの小さな三角マーク。
    growth_surgeの拡張矢印と紛らわしくならないよう、線ではなく塗りつぶし三角+食べ物と同じ暖色にしている。"""
    progress = age / ABILITY_EFFECT_FRAMES
    inner = (14 + progress * 14) * scale
    size = 9 * scale
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bx, by = zx + dx * inner, zy + dy * inner
        perp = (-dy, dx)
        pts = [
            (bx + dx * size, by + dy * size),
            (bx - perp[0] * size * 0.6, by - perp[1] * size * 0.6),
            (bx + perp[0] * size * 0.6, by + perp[1] * size * 0.6),
        ]
        draw.polygon(pts, fill=color)


def _draw_nibble_shrink_mark(draw: ImageDraw.ImageDraw, zx: float, zy: float, color: tuple, age: int, scale: float = 1.0) -> None:
    """absorb_growthの「かじり取り」(2026-09-08追加)で縮んだ側に表示する、内向きの小さな三角マーク
    (_draw_nibble_grow_markのちょうど逆向き)。"""
    progress = age / ABILITY_EFFECT_FRAMES
    outer = (26 - progress * 10) * scale
    size = 9 * scale
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bx, by = zx + dx * outer, zy + dy * outer
        perp = (-dy, dx)
        pts = [
            (bx - dx * size, by - dy * size),
            (bx - perp[0] * size * 0.6, by - perp[1] * size * 0.6),
            (bx + perp[0] * size * 0.6, by + perp[1] * size * 0.6),
        ]
        draw.polygon(pts, fill=color)


def _draw_ability_effects(draw: ImageDraw.ImageDraw, effects: list, frame_state: list, scale: float = 1.0) -> None:
    """特殊能力発動時のエフェクト。何が起きているか説明なしで伝わるよう、能力ごとに異なる
    固有の形状を使う(色は全てABILITY_EFFECT_COLORのその能力の色に統一する。2026-09-09、
    ユーザー要望により全8種を専用形状にした):
    - dash: ターゲットへ追従する照準マーク
    - slam: 急降下を示す二重矢印
    - growth_surge: 外向きの拡張矢印(4隅)
    - poison_wire: 注意喚起のジグザグマーク(設置される実際のワイヤーは別途_draw_wiresで常時表示)
    - speed_boost: 後方に流れる速度線
    - shockwave: 実際の効果範囲まで広がるリング
    - vortex: 自分に向かって縮んでいくリング(引き寄せ、shockwaveの逆。2026-09-08、内向き矢印
      だと「小さくなる」と誤読されやすいとの指摘で変更)
    - teleport: 消える側は収縮・現れる側は拡大するリング
    """
    positions_by_id = {entry["id"]: (entry["x"], entry["y"], entry["radius"]) for entry in frame_state}

    for eff in effects:
        kind = eff["kind"]
        if kind in ("barrier_hit", "barrier_break"):
            # goal_reachのバリアは_draw_goal_barrier側で専用の見た目を毎フレーム描画済みのため、
            # ここでは汎用リングを重ねて描かない(音はrender_audio側で別途鳴らす)。
            continue
        age = eff["age"]
        alpha = max(0.0, 1 - age / ABILITY_EFFECT_FRAMES)
        color = ABILITY_EFFECT_COLOR.get(kind, (200, 200, 200))
        blended = tuple(int(bg + (col - bg) * alpha) for bg, col in zip((24, 24, 28), color))

        if kind == "dash":
            target_info = positions_by_id.get(eff.get("target_id"))
            if target_info is None:
                continue  # ターゲットが既に脱落済み等
            tx, ty, tradius = target_info
            zx, zy = _zoom_point((tx, ty), scale)
            _draw_target_reticle(draw, zx, zy, blended, tradius * scale, scale)
            continue

        zx, zy = _zoom_point((eff["x"], eff["y"]), scale)

        if kind == "slam":
            _draw_slam_mark(draw, zx, zy, blended, age, scale)
            continue
        if kind == "growth_surge":
            _draw_growth_mark(draw, zx, zy, blended, age, scale)
            continue
        if kind == "poison_wire":
            _draw_hazard_mark(draw, zx, zy, blended, age, scale)
            continue
        if kind == "speed_boost":
            _draw_speed_lines(draw, zx, zy, blended, age, scale)
            continue
        if kind == "partial_absorb_grow":
            _draw_nibble_grow_mark(draw, zx, zy, blended, age, scale)
            continue
        if kind == "partial_absorb_shrink":
            _draw_nibble_shrink_mark(draw, zx, zy, blended, age, scale)
            continue

        if kind == "shockwave":
            r = SHOCKWAVE_RADIUS * (age / ABILITY_EFFECT_FRAMES) * scale
        elif kind == "vortex":
            # 2026-09-08、ユーザー指摘対応: 内向き矢印(growth_surgeの外向き矢印の逆)だと
            # 「縮小(小さくなる)」と誤読されやすい。shockwave(赤・外に広がるリング)の
            # ちょうど逆として、リングが自分に向かって縮んでいく(=引き寄せている)演出にする。
            r = VORTEX_RADIUS * (1 - age / ABILITY_EFFECT_FRAMES) * scale
        elif kind == "teleport":
            progress = age / ABILITY_EFFECT_FRAMES
            r = (18 + (progress if eff.get("phase") == "in" else 1 - progress) * 40) * scale
        else:
            r = (18 + age * 5) * scale
        draw.ellipse([zx - r, zy - r, zx + r, zy + r], outline=blended, width=max(1, int(5 * alpha)))


def _ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def _lerp_angle(a: float, b: float, t: float) -> float:
    """角度の最短経路での線形補間(2π境界をまたいでも遠回りしない)。"""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t


def _initial_entity_states(result: SimResult) -> dict[int, dict]:
    """frame 0時点の各エンティティの状態(x,y,radius,angle)。エピローグで「全員を初期位置に
    戻す」際の行き先として使う(8-1/ループ再生対応、2026-09-09再設計)。"""
    if not result.frames:
        return {}
    return {e["id"]: {"x": e["x"], "y": e["y"], "radius": e["radius"], "angle": e.get("angle", 0.0)} for e in result.frames[0]}


def _last_known_entity_states(result: SimResult) -> dict[int, dict]:
    """決着時点での各エンティティの最後の状態。生存していた個体は最終フレーム、
    脱落済みの個体はelimination_orderに記録された脱落時の位置を使う。"""
    states: dict[int, dict] = {}
    if result.frames:
        for e in result.frames[-1]:
            states[e["id"]] = {"x": e["x"], "y": e["y"], "radius": e["radius"], "angle": e.get("angle", 0.0)}
    for ev in result.elimination_order:
        if ev["id"] not in states:
            states[ev["id"]] = {"x": ev["x"], "y": ev["y"], "radius": ev.get("radius", CIRCLE_RADIUS), "angle": 0.0}
    return states


def _tracking_camera_target(frame_state: list, scale: float) -> tuple:
    """追尾カメラの目標クロップ範囲(中心x,中心y,幅,高さ)を、生存プレイヤー全員が
    余白込みで収まる最小サイズ(ただしCAMERA_MIN_WIDTH以上・アリーナ全体以下)で計算する。
    """
    if not frame_state:
        return ARENA_CX, ARENA_CY, ARENA_W, ARENA_H
    zoomed = [_zoom_point((e["x"], e["y"]), scale) for e in frame_state]
    xs = [p[0] for p in zoomed]
    ys = [p[1] for p in zoomed]
    min_x, max_x = min(xs) - CAMERA_PADDING, max(xs) + CAMERA_PADDING
    min_y, max_y = min(ys) - CAMERA_PADDING, max(ys) + CAMERA_PADDING
    bbox_w, bbox_h = max_x - min_x, max_y - min_y
    target_w = max(bbox_w, bbox_h * (ARENA_W / ARENA_H), CAMERA_MIN_WIDTH)
    target_w = min(target_w, ARENA_W)
    target_h = target_w * (ARENA_H / ARENA_W)
    return (min_x + max_x) / 2, (min_y + max_y) / 2, target_w, target_h


def _epilogue_frame_count(result: SimResult) -> int:
    """勝利演出(エピローグ)のフレーム数。決着しなかった場合は0。"""
    if result.winner_id is not None and result.final_position is not None:
        return EPILOGUE_FRAMES
    return 0


def _iter_rendered_frames(
    result: SimResult, circles: list[CircleEntity], camera: str = "fixed", visual_mismatch: dict[int, float] | None = None
):
    """本編+エピローグの全フレームを1枚ずつ順番に生成するジェネレータ。

    動画全体(40〜55秒×60fps=2400〜3300フレーム)を一度にメモリ上のリストへ溜めると、
    1080x1920換算で数十GB規模のメモリを消費しGitHub Actionsの標準ランナーでOOM Killされる
    (2026-09-09に実測で確認)。そのためrender()側で1フレームずつ消費できるジェネレータに
    している。camera="tracking"の平滑化に使うcam_stateは、ジェネレータのローカル変数として
    自然に「前フレームの状態」を保持できる。

    visual_mismatch(8-5): {entity_id: 見た目上の半径倍率}。物理演算の半径には触れず、
    描画上の大きさだけを変える(意図的な「見た目と実力の不一致」を演出する)。
    """
    visual_mismatch = visual_mismatch or {}
    color_by_id = {c.id: c.color for c in circles}
    scale = _rotation_zoom_scale(result.rotation_speed)
    layout = _arena_layout(result.arena_half_x, result.arena_half_y)  # 2026-09-08、ステージサイズのバリエーション対応
    cam_state = None
    face_angle_by_id: dict[int, float] = {}  # 顔の向き(進行方向)。ほぼ静止中は直近の向きを保持する
    shake_by_frame = _build_shake_by_frame(result.collisions, result.elimination_order)

    effects_by_frame: dict = {}
    for ev in result.elimination_order:
        color = color_by_id.get(ev["id"])
        if color is None:
            continue
        base_radius = ev.get("radius", CIRCLE_RADIUS)
        for age in range(EFFECT_FRAMES):
            target = ev["frame"] + age
            effects_by_frame.setdefault(target, []).append((ev["x"], ev["y"], color, age, base_radius))

    ability_effects_by_frame: dict = {}
    for ev in result.ability_events:
        for age in range(ABILITY_EFFECT_FRAMES):
            target = ev["frame"] + age
            ability_effects_by_frame.setdefault(target, []).append(
                {
                    "x": ev["x"],
                    "y": ev["y"],
                    "kind": ev["type"],
                    "age": age,
                    "target_id": ev.get("target_id"),
                    "phase": ev.get("phase"),
                }
            )

    wires_by_frame: dict = {}
    for w in result.wires:
        for f in range(w["start_frame"], w["end_frame"]):
            wires_by_frame.setdefault(f, []).append((w["a"], w["b"]))

    food_by_frame: dict = {}
    for food in result.food_items:
        end = food["end_frame"] if food["end_frame"] is not None else len(result.frames)
        for f in range(food["start_frame"], end):
            food_by_frame[f] = (food["x"], food["y"])

    protection_zone_by_frame: dict = {}  # 未捕獲(ピックアップ待ち)フェーズ
    protection_cage_by_frame: dict = {}  # 実体化(囲いの中に閉じ込め)フェーズ
    for zone in result.protection_zones:
        end = zone["end_frame"] if zone["end_frame"] is not None else len(result.frames)
        for f in range(zone["start_frame"], end):
            protection_zone_by_frame[f] = (zone["x"], zone["y"], zone["radius"])
        # 2026-09-12バグ修正: end_frameは「捕獲された」場合と「誰にも捕獲されずタイムアウトした」
        # 場合の両方でセットされるが、物理的な囲い(cage_shapes)はsimulate()側で捕獲時にしか
        # 生成されない。end_frame is not Noneだけで判定すると、タイムアウトした回(cage_end_frameは
        # 常にNoneのまま)がlen(result.frames)まで補完され、「誰も入っていないのに動画の最後まで
        # 囲いが表示され続ける」バグになっていた。claimed_by is not Noneで実際に捕獲された回だけに絞る。
        if zone["claimed_by"] is not None:
            cage_end = zone["cage_end_frame"] if zone["cage_end_frame"] is not None else len(result.frames)
            for f in range(zone["end_frame"], cage_end):
                protection_cage_by_frame[f] = (zone["x"], zone["y"], zone["cage_radius"])

    barrier_hit_frames = sorted(ev["frame"] for ev in result.ability_events if ev["type"] == "barrier_hit")

    # 2026-09-13、指示書対応: pinballのポップバンパー、ヒットした瞬間から数フレームだけ
    # 該当バンパーを拡縮表示する(_draw_terrainのflash_local_positions)。on_beginがログした
    # bumper_x/bumper_y(ローカル座標、_pinball_bumper_positionsと同じ基準)で該当バンパーを
    # 特定する。
    pinball_flash_by_frame: dict = {}
    for ev in result.ability_events:
        if ev["type"] != "pinball_kick" or "bumper_x" not in ev:
            continue
        local_pos = (round(ev["bumper_x"], 1), round(ev["bumper_y"], 1))
        for age in range(PINBALL_FLASH_FRAMES):
            pinball_flash_by_frame.setdefault(ev["frame"] + age, set()).add(local_pos)

    gun_by_frame: dict = {}
    for gun in result.guns:
        end = gun["end_frame"] if gun["end_frame"] is not None else len(result.frames)
        for f in range(gun["start_frame"], end):
            gun_by_frame[f] = (gun["x"], gun["y"])

    shield_by_frame: dict = {}
    for shield in result.shields:
        end = shield["end_frame"] if shield["end_frame"] is not None else len(result.frames)
        for f in range(shield["start_frame"], end):
            shield_by_frame[f] = (shield["x"], shield["y"])

    # 弾は直進+一定速度なので、start_frame〜end_frameの間だけ幾何学的に位置を再計算して描画する
    # (シミュレーション側で毎フレーム位置を記録していないため、ここで同じ式を使って復元する)
    bullets_by_frame: dict = {}
    for b in result.bullets:
        end = b["end_frame"] if b["end_frame"] is not None else len(result.frames)
        for f in range(b["start_frame"], end):
            t = (f - b["start_frame"]) * DT
            bx = b["x"] + math.cos(b["angle"]) * BULLET_SPEED * t
            by = b["y"] + math.sin(b["angle"]) * BULLET_SPEED * t
            bullets_by_frame.setdefault(f, []).append((bx, by, b["angle"]))

    # weapon_colosseum専用: 武器ピックアップ(種類ごとに同時出現しうるためリスト)
    weapons_by_frame: dict = {}
    for w in result.weapons:
        end = w["end_frame"] if w["end_frame"] is not None else len(result.frames)
        for f in range(w["start_frame"], end):
            weapons_by_frame.setdefault(f, []).append((w["kind"], w["x"], w["y"]))

    # weapon_colosseum(弓矢)専用: 矢は重力の影響を受ける放物線のため、弾のような単純な直進式では
    # 復元できない。発射時点の初速(vx0/vy0)から等加速度運動の式で毎フレーム位置を再計算する。
    arrows_by_frame: dict = {}
    for a in result.arrows:
        end = a["end_frame"] if a["end_frame"] is not None else len(result.frames)
        for f in range(a["start_frame"], end):
            t = (f - a["start_frame"]) * DT
            ax = a["x"] + a["vx0"] * t
            ay = a["y"] + a["vy0"] * t + 0.5 * ARROW_GRAVITY * t * t
            avy = a["vy0"] + ARROW_GRAVITY * t
            arrows_by_frame.setdefault(f, []).append((ax, ay, a["vx0"], avy))

    for frame_idx, (frame_state, angle) in enumerate(zip(result.frames, result.arena_angles)):
        img = Image.new("RGB", (int(ARENA_W), int(ARENA_H)), (24, 24, 28))
        draw = ImageDraw.Draw(img)
        _draw_arena_walls(draw, angle, result.shape, result.hole_width, result.terrain, layout["half_x"], layout["half_y"], scale)
        _draw_terrain(
            draw, result.terrain, angle, layout["half_x"], layout["half_y"], scale,
            flash_local_positions=pinball_flash_by_frame.get(frame_idx),
        )
        _draw_rule_overlay(draw, result.rule, frame_idx / FPS, layout, scale)
        if result.rule == "area_control":
            img = _apply_area_control_danger_aura(img, frame_idx / FPS, layout, scale)
            draw = ImageDraw.Draw(img)
        if result.terrain == "pinball":
            _draw_flipper_zone(draw, layout, scale)
        if result.accel_zone:
            _draw_accel_zone(draw, frame_idx, layout["accel_zone_rect"], scale)
        if result.trap:
            _draw_trap(draw, layout["trap_center"], scale)
        _draw_wires(draw, wires_by_frame.get(frame_idx, []), scale)
        food_pos = food_by_frame.get(frame_idx)
        if food_pos is not None:
            _draw_food(draw, food_pos[0], food_pos[1], frame_idx, scale)
        pz = protection_zone_by_frame.get(frame_idx)
        if pz is not None:
            _draw_protection_zone(draw, pz[0], pz[1], pz[2], frame_idx, scale)
        pcage = protection_cage_by_frame.get(frame_idx)
        if pcage is not None:
            _draw_protection_cage(draw, result.shape, pcage[0], pcage[1], pcage[2], scale)
        if result.goal_barrier is not None:
            break_frame = result.goal_barrier.get("break_frame")
            if break_frame is None or frame_idx < break_frame:
                hits_so_far = sum(1 for f in barrier_hit_frames if f <= frame_idx)
                _draw_goal_barrier(
                    draw,
                    result.goal_barrier["x"],
                    result.goal_barrier["y"],
                    result.goal_barrier["radius"],
                    hits_so_far,
                    result.goal_barrier["hits_to_break"],
                    scale,
                )
            elif break_frame <= frame_idx < break_frame + DECISION_BURST_FRAMES:
                bzx, bzy = _zoom_point((result.goal_barrier["x"], result.goal_barrier["y"]), scale)
                _draw_decision_burst(draw, bzx, bzy, BARRIER_COLOR, frame_idx - break_frame, scale)
        gun_pos = gun_by_frame.get(frame_idx)
        if gun_pos is not None:
            _draw_gun_pickup(draw, gun_pos[0], gun_pos[1], frame_idx, scale)
        shield_pos = shield_by_frame.get(frame_idx)
        if shield_pos is not None:
            _draw_shield_pickup(draw, shield_pos[0], shield_pos[1], frame_idx, scale)
        for bx, by, bangle in bullets_by_frame.get(frame_idx, []):
            _draw_bullet(draw, bx, by, bangle, scale)
        for wkind, wx, wy in weapons_by_frame.get(frame_idx, []):
            _draw_weapon_pickup(draw, wkind, wx, wy, frame_idx, scale)
        for ax, ay, avx, avy in arrows_by_frame.get(frame_idx, []):
            _draw_arrow(draw, ax, ay, avx, avy, scale)
        _draw_elimination_effects(draw, effects_by_frame.get(frame_idx, []), scale)
        _draw_ability_effects(draw, ability_effects_by_frame.get(frame_idx, []), frame_state, scale)
        leader_id = None
        if result.rule == "absorb_growth" and len(frame_state) > 1:
            leader_id = max(frame_state, key=lambda e: e["radius"])["id"]
        for entry in frame_state:
            mismatch = visual_mismatch.get(entry["id"], 1.0)
            if entry.get("empowered"):
                _draw_empowered_aura(draw, entry["x"], entry["y"], entry["radius"] * mismatch, frame_idx, scale)
            if entry.get("shielded"):
                _draw_shield_aura(draw, entry["x"], entry["y"], entry["radius"] * mismatch, frame_idx, scale)
            if entry.get("holding_gun"):
                others = [e for e in frame_state if e["id"] != entry["id"]]
                if others:
                    target = min(others, key=lambda o: (o["x"] - entry["x"]) ** 2 + (o["y"] - entry["y"]) ** 2)
                    aim_angle = math.atan2(target["y"] - entry["y"], target["x"] - entry["x"])
                else:
                    aim_angle = 0.0
                total_charge_frames = max(1, int(GUN_AIM_SECONDS * FPS))
                charge = 1.0 - (entry.get("gun_fire_frame", frame_idx) - frame_idx) / total_charge_frames
                charge = min(1.0, max(0.0, charge))
                _draw_held_gun(draw, entry["x"], entry["y"], aim_angle, entry["radius"] * mismatch, charge, scale)
            if entry.get("weapon"):
                w_others = [e for e in frame_state if e["id"] != entry["id"]]
                if w_others:
                    w_target = min(w_others, key=lambda o: (o["x"] - entry["x"]) ** 2 + (o["y"] - entry["y"]) ** 2)
                    w_angle = math.atan2(w_target["y"] - entry["y"], w_target["x"] - entry["x"])
                else:
                    w_angle = 0.0
                _draw_held_weapon(
                    draw,
                    entry["weapon"],
                    entry["x"],
                    entry["y"],
                    entry["radius"] * mismatch,
                    w_angle,
                    entry.get("vx", 0.0),
                    entry.get("vy", 0.0),
                    frame_idx,
                    scale,
                    entry.get("axe_spinning", False),
                )
            _draw_player(
                draw,
                entry["x"],
                entry["y"],
                entry["radius"] * mismatch,
                color_by_id[entry["id"]],
                scale,
                result.player_shape,
                entry.get("angle", 0.0),
            )
            collision_blink = any(shake_by_frame.get(frame_idx - k, 0.0) > 0.3 for k in range(3))
            expression = _player_expression(entry, result.rule, frame_state)
            ev_speed = math.hypot(entry.get("vx", 0.0), entry.get("vy", 0.0))
            if ev_speed > FACE_DIRECTION_MIN_SPEED:
                face_angle_by_id[entry["id"]] = math.atan2(entry.get("vy", 0.0), entry.get("vx", 0.0))
            face_angle = face_angle_by_id.get(entry["id"], FACE_DEFAULT_ANGLE)
            _draw_player_eyes(
                draw,
                entry["x"],
                entry["y"],
                entry["radius"] * mismatch,
                scale,
                entry["id"],
                frame_idx,
                collision_blink,
                expression,
                face_angle,
            )
            if entry["id"] == leader_id:
                _draw_crown(draw, entry["x"], entry["y"], entry["radius"] * mismatch, scale)
            if entry.get("hp") is not None:
                _draw_hp_bar(draw, entry["x"], entry["y"], entry["radius"] * mismatch, entry["hp"], entry.get("max_hp") or WEAPON_STARTING_HP, scale)

        # 2026-09-20: 照明(単一光源+陰影+落ち影)は、tracking camera/シェイク/ズームといった
        # 画面座標を変形する処理より前に適用する。frame_state中の座標(_zoom_pointでscale適用
        # 済みだが、まだcrop/resize等はされていない)と一致させるため。後段の変形はこの照明込みの
        # 絵ごと一緒に動かす。
        img = _apply_lighting(img, frame_idx, frame_state, scale)

        if camera == "tracking":
            tx, ty, tw, th = _tracking_camera_target(frame_state, scale)
            if cam_state is None:
                cam_state = [tx, ty, tw, th]
            else:
                for i, target in enumerate((tx, ty, tw, th)):
                    cam_state[i] += (target - cam_state[i]) * CAMERA_SMOOTHING
            ccx, ccy, cw, ch = cam_state
            half_w, half_h = cw / 2, ch / 2
            ccx = min(max(ccx, half_w), ARENA_W - half_w)
            ccy = min(max(ccy, half_h), ARENA_H - half_h)
            img = img.crop((ccx - half_w, ccy - half_h, ccx + half_w, ccy + half_h)).resize((int(ARENA_W), int(ARENA_H)))

        shake_intensity = shake_by_frame.get(frame_idx, 0.0)
        if shake_intensity > 0:
            img = _apply_screen_shake(img, result.seed, frame_idx, shake_intensity, scale)

        img = _apply_opening_zoom(img, frame_idx)
        if result.rule == "area_control":
            img = _area_control_tension_zoom(img, frame_idx)

        _draw_hud(img, result.rule, len(frame_state))
        yield img

    # 勝利演出(エピローグ、2026-09-13 カメラ寄せ廃止): 当初は波紋(shockwave ring)が壁に
    # 到達した瞬間に全員を初期配置へ瞬間ワープさせる方式(0.5秒)にしたが、実機確認で
    # 「瞬間ワープはループ演出として不自然」というフィードバックを受けた。波紋による
    # 視線誘導(全画面フラッシュを使わない光過敏性対策)自体は活かしつつ、瞬間ワープを廃止し、
    # 旧来の「全員が滑らかに初期位置へ戻る移動」を大幅短縮した形で復活させた: 決着直後の
    # 短い静止(EPILOGUE_HOLD_END)の後、残り時間で全員(脱落済み含む)がイージング付きで
    # 初期位置・初期半径・初期角度へ連続的に移動する。さらに2026-09-13、決着時に勝者へ
    # カメラを寄せるズームインも「不自然」との指摘で廃止し、カメラは全体を映したまま固定。
    # 代わりにWINテキストの色を勝者のプレイヤーカラーにして勝者を示す。
    if result.winner_id is not None and result.final_position is not None:
        zwx, zwy = _zoom_point(result.final_position, scale)

        final_t = (result.decided_frame or 0) / FPS
        initial_states = _initial_entity_states(result)
        last_states = _last_known_entity_states(result)
        entity_ids = sorted(initial_states.keys())
        initial_arena_angle = result.arena_angles[0] if result.arena_angles else 0.0
        eliminated_ids = {ev["id"] for ev in result.elimination_order}
        winner_color = color_by_id[result.winner_id]
        # 波紋が確実に外枠まで届く半径(キャンバス対角線の半分。勝者位置によらず余裕を持たせる)
        max_ring_radius = math.hypot(ARENA_W, ARENA_H) / 2

        for i in range(EPILOGUE_FRAMES):
            phase_t = i / max(1, EPILOGUE_FRAMES - 1)

            # 決着直後の短い静止(EPILOGUE_HOLD_END)を過ぎたら、残り時間をかけて
            # 全員を初期位置へ滑らかに(ease-out)戻す。波紋の進行度もこれと同じ時間軸を使う
            # (移動が終わる頃に波紋も消えるよう揃える)。
            if phase_t <= EPILOGUE_HOLD_END:
                travel_progress = 0.0
            else:
                travel_progress = _ease_out((phase_t - EPILOGUE_HOLD_END) / (1 - EPILOGUE_HOLD_END))

            # 2026-09-13、ユーザー指摘対応: 決着時に勝者へカメラを寄せる(ズームイン)演出は廃止。
            # カメラは常に全体を映したまま固定し、代わりに勝者の色をWINテキストに反映することで
            # 誰が勝者かを伝える(下記のtext_color参照)。
            crop_w, crop_h = ARENA_W, ARENA_H
            half_w, half_h = crop_w / 2, crop_h / 2
            cx, cy = ARENA_CX, ARENA_CY

            # 枠の回転角・ルールオーバーレイ(area_controlの安全地帯半径等)も、移動と
            # 同じtravel_progressで決着時点→初期状態へ滑らかに巻き戻す。
            current_arena_angle = _lerp_angle(result.final_arena_angle, initial_arena_angle, travel_progress)
            overlay_t = final_t * (1 - travel_progress)

            base = Image.new("RGB", (int(ARENA_W), int(ARENA_H)), (24, 24, 28))
            draw = ImageDraw.Draw(base)
            _draw_arena_walls(draw, current_arena_angle, result.shape, result.hole_width, result.terrain, layout["half_x"], layout["half_y"], scale)
            # 2026-09-13、ユーザー指摘で修正: 本編側では毎フレーム_draw_terrainを呼んでいたが、
            # エピローグ側はこの呼び出しが漏れており、donut/pegboard/pinball等の内部地形の
            # オブジェクトがエピローグ突入と同時に消えて見えるバグがあった。
            _draw_terrain(draw, result.terrain, current_arena_angle, layout["half_x"], layout["half_y"], scale)
            _draw_rule_overlay(draw, result.rule, overlay_t, layout, scale)

            if i < DECISION_BURST_FRAMES:
                _draw_decision_burst(draw, zwx, zwy, winner_color, i, scale)

            ring_progress = phase_t
            if ring_progress < 1.0:
                _draw_shockwave_ring(draw, zwx, zwy, winner_color, ring_progress, scale, max_ring_radius)

            epilogue_frame_state = []
            for entity_id in entity_ids:
                is_eliminated = entity_id in eliminated_ids
                init = initial_states[entity_id]
                last = last_states.get(entity_id, init)
                x = last["x"] + (init["x"] - last["x"]) * travel_progress
                y = last["y"] + (init["y"] - last["y"]) * travel_progress
                radius = last["radius"] + (init["radius"] - last["radius"]) * travel_progress
                angle = _lerp_angle(last["angle"], init["angle"], travel_progress)
                mismatch = visual_mismatch.get(entity_id, 1.0)
                draw_radius = radius * mismatch
                # 脱落済みプレイヤーは、移動フェーズの進行(=初期位置へ戻る進行)に合わせて
                # フェードインさせる(旧デザインと同じ狙い: 決着直後に唐突に再登場させない)。
                alpha = travel_progress if is_eliminated else 1.0
                if entity_id == result.winner_id:
                    pulse = 1.0 + 0.15 * abs(((i % 20) / 20) - 0.5) * 2
                    draw_radius *= pulse
                _draw_player(draw, x, y, draw_radius, color_by_id[entity_id], scale, result.player_shape, angle, alpha)
                if alpha >= 1.0:
                    _draw_player_eyes(draw, x, y, draw_radius, scale, entity_id, len(result.frames) + i, False)
                epilogue_frame_state.append({"x": x, "y": y, "radius": draw_radius})

            # 2026-09-20、ユーザー指摘対応: 照明(単一光源+陰影+落ち影)がエピローグ側では
            # 一切適用されておらず、決着した瞬間にライトが消えて見えるバグがあった
            # (本編側の_apply_lighting呼び出しがこのエピローグ専用の描画パスに漏れていた)。
            # 常時点灯にするため、本編と同じ_apply_lightingをここでも適用する。
            base = _apply_lighting(base, len(result.frames) + i, epilogue_frame_state, scale)

            crop_box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            cropped = base.crop(crop_box).resize((int(ARENA_W), int(ARENA_H)))

            # シームレスループ復活(2026-09-21): エピローグ終盤で冒頭ズームと真逆の変化
            # (ズームイン+ブラー強化)をかける。WINテキストが一緒にズーム・ブラーされて
            # 読みにくくなるのを防ぐため、テキスト描画より前に適用する。
            cropped = _apply_loop_reverse_zoom(cropped, phase_t)

            # 2026-09-21、ユーザー指示「WIN表示のディレイとフェード」対応: 決着後すぐに表示して
            # 終盤までずっと保持する旧仕様(2026-09-13実装)は、「WINが表示され続けている=
            # もうすぐ動画が終わる」という合図になりスワイプを誘発しているとの判断で撤回。
            # EPILOGUE_TEXT_DELAY(0.5秒)まで一切表示せず、その後ほぼ瞬間的に出現して
            # EPILOGUE_TEXT_FADE_OUT_DURATION(0.3秒)で即座にフェードアウトさせる
            # (=表示は決着後0.5〜0.8秒の間だけの短い演出にする)。
            fade_in_end = EPILOGUE_TEXT_DELAY + EPILOGUE_TEXT_POPIN_FRAMES / max(1, EPILOGUE_FRAMES - 1)
            fade_out_end = EPILOGUE_TEXT_DELAY + EPILOGUE_TEXT_FADE_OUT_DURATION
            if phase_t < EPILOGUE_TEXT_DELAY:
                text_alpha = 0.0
            elif phase_t < fade_in_end:
                text_alpha = (phase_t - EPILOGUE_TEXT_DELAY) / (fade_in_end - EPILOGUE_TEXT_DELAY)
            elif phase_t < fade_out_end:
                text_alpha = 1.0 - (phase_t - fade_in_end) / (fade_out_end - fade_in_end)
            else:
                text_alpha = 0.0

            if text_alpha > 0:
                draw2 = ImageDraw.Draw(cropped)
                text, font = _epilogue_win_label(result)
                bg = (24, 24, 28)
                # 2026-09-13、ユーザー指摘対応: カメラ寄せ演出を廃止した代わりに、WINの文字色を
                # 勝者のプレイヤーカラーにして誰が勝ったかを示す。文字色が暗い色(紺・紫等)だと
                # 黒い影では暗いアリーナ背景に沈んで読みにくくなるため、影の色は文字色の輝度に
                # 応じて黒(明るい文字色向け)/白に近い色(暗い文字色向け)を自動選択し、
                # どのプレイヤーカラーでも視認性を保つ。
                luminance = 0.299 * winner_color[0] + 0.587 * winner_color[1] + 0.114 * winner_color[2]
                shadow_base = (245, 245, 245) if luminance < 140 else (0, 0, 0)
                text_color = tuple(int(c * text_alpha + bg_c * (1 - text_alpha)) for c, bg_c in zip(winner_color, bg))
                shadow_color = tuple(int(c * text_alpha + bg_c * (1 - text_alpha)) for c, bg_c in zip(shadow_base, bg))
                bbox = draw2.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx, ty = (ARENA_W - tw) / 2, ARENA_H * 0.35
                shadow_offset = 8
                draw2.text((tx + shadow_offset, ty + shadow_offset), text, fill=shadow_color, font=font)
                draw2.text((tx, ty), text, fill=text_color, font=font)

            yield cropped


_WIN_FONT_CACHE = None
_EPILOGUE_LABEL_FONT_CACHE: dict[int, object] = {}


def _win_font():
    # 2026-09-08、ユーザー指摘により拡大(90→160)。決着の瞬間の主役表示なので、
    # HUDキャプションよりも大きく目立たせる。
    global _WIN_FONT_CACHE
    if _WIN_FONT_CACHE is None:
        _WIN_FONT_CACHE = _load_bold_font(160)
    return _WIN_FONT_CACHE


def _epilogue_label_font(size: int):
    """_win_font()と同じ太字フォントをsize違いでキャッシュする(match_type別のラベル用)。"""
    cached = _EPILOGUE_LABEL_FONT_CACHE.get(size)
    if cached is None:
        cached = _load_bold_font(size)
        _EPILOGUE_LABEL_FONT_CACHE[size] = cached
    return cached


def _epilogue_win_label(result: "SimResult") -> tuple[str, object]:
    """2026-09-21、ユーザー指示「TEAM WINS/BOSS WINS等の明示ラベルを追加」対応。
    match_typeに応じて表示文言とフォントサイズを変える(「WIN」は3文字前提のfont(160pt)
    のまま、長い文言は800px幅の枠に収まるようフォントを縮小する。実測: 160ptだと
    "TEAM WINS"/"BOSS WINS"は916px、"PLAYERS WIN"は1113pxとARENA_W(800)を超えて
    はみ出すため、事前に幅を計測した上で安全なサイズに固定した)。"""
    if result.match_type == "team":
        return "TEAM WINS", _epilogue_label_font(110)
    if result.match_type == "boss":
        if result.winner_is_boss:
            return "BOSS WINS", _epilogue_label_font(110)
        return "PLAYERS WIN", _epilogue_label_font(90)
    return "WIN", _win_font()


def _synth_pop(volume: float, pitch: float) -> np.ndarray:
    # 8-4「音響設計の精度」: 短く・クリアで余韻を残しすぎない「歯切れの良さ」を狙い、
    # 2026-09-09にdecayを上げて(35→42)テールを短くした(6章の音響ブランディング基準に追加)。
    duration = 0.08
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = np.sin(2 * np.pi * pitch * t) * np.exp(-t * 42) * volume
    return wave.astype(np.float32)


def _synth_eliminate(volume: float = 0.35) -> np.ndarray:
    """脱落の瞬間用の効果音(衝突ポップ・勝利ジングルと区別するため)。
    8-4対応で2026-09-09にdecayを上げ(9→13)、durationを短縮(0.22→0.17)し歯切れをよくした。

    2026-09-20、ユーザー指示(撃破インパクトの強化): 従来の下降ピッチ単体に加えて、
    (1)高域のクリスタルブレイク音(複数の高周波を重ねて減衰させる)と、
    (2)低域のサブベースキック(短時間で立ち上がり減衰する低音の一撃)をレイヤーし、
    より「重み」のある撃破音にした。既存の下降ピッチ(中域)は「らしさ」を保つためそのまま残す。
    """
    duration = 0.22
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    freq = 480 * (120 / 480) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    mid = np.sin(phase) * np.exp(-t * 13)

    crystal = np.zeros_like(t)
    for f in (2200.0, 3100.0, 4200.0):
        crystal += np.sin(2 * np.pi * f * t)
    crystal *= np.exp(-t * 35) * 0.33

    kick_freq = 90 * (40 / 90) ** (t / duration)
    kick_phase = 2 * np.pi * np.cumsum(kick_freq) / SAMPLE_RATE
    kick = np.sin(kick_phase) * np.exp(-t * 18) * 0.9

    wave = (mid + crystal + kick) * volume
    return wave.astype(np.float32)


def _synth_sweep(f_start: float, f_end: float, duration: float, volume: float, decay: float = 6.0) -> np.ndarray:
    """周波数がf_start→f_endへ指数的に変化するプレースホルダー効果音。
    各特殊能力ごとにパラメータを変えて使い回す(ability→サウンドの対応はABILITY_SOUND_PARAMS参照)。"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freq = f_start * (f_end / f_start) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    wave = np.sin(phase) * np.exp(-t * decay) * volume
    return wave.astype(np.float32)


def _synth_wire_place(volume: float = 0.3) -> np.ndarray:
    """毒ワイヤー設置用のブザー風プレースホルダー効果音。"""
    duration = 0.18
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * 180 * t)
    tremolo = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 40 * t))
    wave = carrier * tremolo * np.exp(-t * 8) * volume
    return wave.astype(np.float32)


def _synth_teleport(volume: float = 0.3) -> np.ndarray:
    """瞬間移動用の2連チャープ(ピコッ、ピコッという電子音)。"""
    a = _synth_sweep(400, 900, 0.05, volume, decay=2.0)
    gap = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)
    b = _synth_sweep(900, 1500, 0.05, volume, decay=2.0)
    return np.concatenate([a, gap, b])


def _synth_barrier_hit(volume: float = 0.35) -> np.ndarray:
    """goal_reach専用: ゴールバリアに体当たりした瞬間の硬質な衝突音。"""
    return _synth_sweep(500, 250, 0.12, volume, decay=10.0)


def _synth_barrier_break(volume: float = 0.4) -> np.ndarray:
    """goal_reach専用: バリア破壊の瞬間の弾けるような音。"""
    return _synth_sweep(900, 150, 0.3, volume, decay=4.5)


def _synth_gun_pickup(volume: float = 0.3) -> np.ndarray:
    """gun_duel専用: 銃を拾った瞬間の音。"""
    return _synth_sweep(500, 900, 0.08, volume, decay=8.0)


def _synth_gun_fire(volume: float = 0.4) -> np.ndarray:
    """gun_duel専用: 発射音。"""
    return _synth_sweep(700, 100, 0.15, volume, decay=7.0)


def _synth_shield_pickup(volume: float = 0.3) -> np.ndarray:
    """gun_duel専用: シールドを拾った瞬間の音。"""
    return _synth_sweep(400, 700, 0.2, volume, decay=5.0)


def _synth_bullet_blocked(volume: float = 0.3) -> np.ndarray:
    """gun_duel専用: 弾がシールドで防がれた時の、金属的に弾かれる音。"""
    return _synth_sweep(1000, 600, 0.1, volume, decay=9.0)


# 2026-09-20、ユーザー指示: 武器が命中した瞬間、武器種ごとに異なる音を鳴らす。
_WEAPON_HIT_SOUND_CACHE: dict[str, np.ndarray] = {}


def _synth_weapon_hit(kind: str) -> np.ndarray:
    """weapon_colosseum専用: 武器種ごとのヒット音。近接の刃物系は高めで歯切れよく、
    打撃・斬撃系は低くて重い音にする。"""
    cached = _WEAPON_HIT_SOUND_CACHE.get(kind)
    if cached is not None:
        return cached
    if kind == "sword":
        sound = _synth_pop(0.32, 1700)  # 甲高い金属音
    elif kind == "spear":
        sound = _synth_sweep(1300, 500, 0.09, 0.32, decay=9.0)  # 突き刺す鋭い音
    elif kind == "hammer":
        sound = _synth_sweep(220, 70, 0.18, 0.42, decay=6.5)  # 重い打撃音
    elif kind == "bow":
        sound = _synth_pop(0.28, 900)  # 矢が刺さる音
    elif kind == "axe":
        sound = _synth_sweep(180, 55, 0.22, 0.48, decay=5.5)  # 斧の重い一撃
    else:
        sound = _synth_pop(0.25, 700)
    _WEAPON_HIT_SOUND_CACHE[kind] = sound
    return sound




def _synth_opening_impact(volume: float = 0.55) -> np.ndarray:
    """2026-09-16、冒頭フック強化(ユーザー指示8-6): 動画0.00秒ぴったりに鳴らす、
    重低音インパクト(ドゥン)+高めの倍音(ゴング風)の複合音。無音スタートによる
    フィード離脱を防ぐ目的。"""
    thud = _synth_sweep(95, 45, 0.35, volume, decay=7.0)
    overtone = _synth_sweep(260, 180, 0.15, volume * 0.4, decay=12.0)
    wave = np.zeros(len(thud), dtype=np.float32)
    wave[: len(thud)] += thud
    wave[: len(overtone)] += overtone
    return wave


# 特殊能力ごとの効果音パラメータ(f_start, f_end, duration, volume, decay)。
# dash/shockwave等は_synth_sweepへそのまま渡す。teleport/poison_wireは専用関数。
# 2026-09-09、8-4「音響設計の精度」対応でdecayを底上げ(+2前後)し、歯切れをよくした。
ABILITY_SOUND_PARAMS = {
    "dash": (220, 700, 0.12, 0.3, 6.0),
    "shockwave": (500, 80, 0.2, 0.4, 9.0),
    "vortex": (300, 900, 0.25, 0.28, 6.0),  # 上昇スイープで「吸い込む」印象にする(旧freezeは下降)
    "growth_surge": (300, 600, 0.3, 0.25, 5.0),
    "speed_boost": (300, 900, 0.15, 0.3, 7.0),
    "slam": (300, 90, 0.15, 0.4, 10.0),
}


def _synth_for_ability(kind: str) -> np.ndarray:
    if kind == "poison_wire":
        return _synth_wire_place()
    if kind == "teleport":
        return _synth_teleport()
    # 2026-09-13、指示書対応: pin_wallのピン衝突用に、短く歯切れの良い高音ポップ音を
    # 明示的に割り当てる(指示書が挙げていた"_synth_pin_hit"相当。8-4「音響設計の精度」向けに
    # 既にチューニング済みの_synth_popをそのまま再利用し、重複実装を避けた)。
    # あわせて、pinball(能動キック式バンパー)がABILITY_SOUND_PARAMS未登録のまま
    # 無音になっていた(2026-09-13以前からの見落とし)のもここで修正する。
    if kind == "pin_hit":
        return _synth_pop(0.22, 1500)
    if kind == "pinball_kick":
        return _synth_pop(0.32, 850)
    if kind == "flipper_kick":
        # フリッパーはバンパーよりも低く・太い音にして、由来の違い(自動キック)を耳でも
        # 区別できるようにする。
        return _synth_sweep(220, 500, 0.1, 0.3, decay=6.0)
    params = ABILITY_SOUND_PARAMS.get(kind)
    if params is None:
        return _synth_sweep(300, 300, 0.05, 0.0)
    return _synth_sweep(*params)


def _synth_victory_jingle(volume: float = 0.5) -> np.ndarray:
    notes = [523.25, 659.25, 783.99]  # C5, E5, G5
    note_dur = 0.18
    gap = 0.06
    chunks = []
    for freq in notes:
        t = np.linspace(0, note_dur, int(SAMPLE_RATE * note_dur), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 6) * volume
        chunks.append(wave.astype(np.float32))
        chunks.append(np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32))
    return np.concatenate(chunks)


BGM_VOLUME = 0.11
BGM_MIN_BPM = 70.0
BGM_MAX_BPM = 150.0


def _synth_bgm(total_seconds: float, decided_seconds: float | None, volume: float = BGM_VOLUME) -> np.ndarray:
    """背景に薄く敷くプレースホルダーBGM(企画書3章の「BGMのテンポ・緊張感」軸の簡易実装)。
    持続的な低音パッドの上に、決着に近づくほどテンポが上がる短いパルスを重ねることで
    緊張感の高まりを表現する。本番のブランディング用BGM(6章)ではない。
    """
    n = int(total_seconds * SAMPLE_RATE)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, total_seconds, n, endpoint=False)

    pad = np.sin(2 * np.pi * 65.41 * t) * 0.5 + np.sin(2 * np.pi * 98.00 * t) * 0.35 + np.sin(2 * np.pi * 130.81 * t) * 0.25
    pad *= 0.5

    target = decided_seconds if decided_seconds else total_seconds
    progress = np.clip(t / max(target, 1e-3), 0.0, 1.0)
    bpm = BGM_MIN_BPM + progress * (BGM_MAX_BPM - BGM_MIN_BPM)
    beat_phase = np.cumsum(bpm / 60.0 / SAMPLE_RATE)
    pulse_env = np.clip(1 - 6 * (beat_phase % 1.0), 0.0, 1.0) ** 2
    pulse = np.sin(2 * np.pi * 220 * t) * pulse_env * (0.15 + 0.15 * progress)

    return ((pad + pulse) * volume).astype(np.float32)


def render_audio(result: SimResult, total_frames: int) -> np.ndarray:
    """パイプライン疎通確認用のプレースホルダー合成音
    (衝突ポップ+脱落音+特殊能力音+勝利ジングル)。"""
    n_samples = int(total_frames / FPS * SAMPLE_RATE) + SAMPLE_RATE
    buffer = np.zeros(n_samples, dtype=np.float32)

    # 2026-09-16、冒頭フック強化(ユーザー指示8-6): 0.00秒ぴったりに重低音インパクト音を配置。
    opening_impact = _synth_opening_impact()
    buffer[: len(opening_impact)] += opening_impact

    max_impulse = max((c["impulse"] for c in result.collisions), default=1.0)
    for c in result.collisions:
        volume = float(np.clip(c["impulse"] / max(max_impulse, 1.0), 0.08, 1.0)) * 0.5
        pitch = 260 + 260 * (1 - volume)
        pop = _synth_pop(volume, pitch)
        start = int(c["frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(pop), len(buffer))
        buffer[start:end] += pop[: end - start]

    elim_sound = _synth_eliminate()
    for ev in result.elimination_order:
        start = int(ev["frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(elim_sound), len(buffer))
        if end > start:
            buffer[start:end] += elim_sound[: end - start]

    ability_sounds = {kind: _synth_for_ability(kind) for kind in ABILITY_EFFECT_COLOR}
    for ev in result.ability_events:
        if ev["type"] == "weapon_hit":
            continue  # 専用音を下のweapon_colosseum節で武器種ごとに鳴らす
        sound = ability_sounds.get(ev["type"])
        if sound is None:
            continue
        start = int(ev["frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(sound), len(buffer))
        if end > start:
            buffer[start:end] += sound[: end - start]

    # 2026-09-20、ユーザー指示: weapon_colosseumの武器命中音(武器種ごとに音色を変える)。
    for ev in result.ability_events:
        if ev["type"] != "weapon_hit":
            continue
        sound = _synth_weapon_hit(ev.get("phase", ""))
        start = int(ev["frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(sound), len(buffer))
        if end > start:
            buffer[start:end] += sound[: end - start]

    if result.decided_frame is not None:
        jingle = _synth_victory_jingle()
        start = int(result.decided_frame / FPS * SAMPLE_RATE)
        end = min(start + len(jingle), len(buffer))
        buffer[start:end] += jingle[: end - start]

    # 2026-09-13、ユーザー指摘により削除: エピローグで各プレイヤーが初期位置へ戻り始める瞬間
    # (移動フェーズ開始、旧来「ズームアウト」と呼んでいたタイミング)の効果音は今後付けない。
    # 決着直後のズームイン相当は既存の勝利ジングルが担う。

    # 2026-09-08、ユーザー要望: goal_reachのバリア(体当たりで破壊)の効果音。
    barrier_hit_sound = _synth_barrier_hit()
    barrier_break_sound = _synth_barrier_break()
    for ev in result.ability_events:
        if ev["type"] == "barrier_hit":
            sound = barrier_hit_sound
        elif ev["type"] == "barrier_break":
            sound = barrier_break_sound
        else:
            continue
        start = int(ev["frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(sound), len(buffer))
        if end > start:
            buffer[start:end] += sound[: end - start]

    # 2026-09-08、新ルールgun_duelの効果音(銃/シールドの取得音・発射音・防御音)。
    gun_pickup_sound = _synth_gun_pickup()
    for gun in result.guns:
        if gun["picked_by"] is None or gun["end_frame"] is None:
            continue
        start = int(gun["end_frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(gun_pickup_sound), len(buffer))
        if end > start:
            buffer[start:end] += gun_pickup_sound[: end - start]

    shield_pickup_sound = _synth_shield_pickup()
    for shield in result.shields:
        if shield["picked_by"] is None or shield["end_frame"] is None:
            continue
        start = int(shield["end_frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(shield_pickup_sound), len(buffer))
        if end > start:
            buffer[start:end] += shield_pickup_sound[: end - start]

    gun_fire_sound = _synth_gun_fire()
    bullet_blocked_sound = _synth_bullet_blocked()
    for b in result.bullets:
        start = int(b["start_frame"] / FPS * SAMPLE_RATE)
        end = min(start + len(gun_fire_sound), len(buffer))
        if end > start:
            buffer[start:end] += gun_fire_sound[: end - start]
        if b["blocked"] and b["end_frame"] is not None:
            bstart = int(b["end_frame"] / FPS * SAMPLE_RATE)
            bend = min(bstart + len(bullet_blocked_sound), len(buffer))
            if bend > bstart:
                buffer[bstart:bend] += bullet_blocked_sound[: bend - bstart]

    decided_seconds = result.decided_frame / FPS if result.decided_frame is not None else None
    bgm = _synth_bgm(len(buffer) / SAMPLE_RATE, decided_seconds)
    buffer[: len(bgm)] += bgm

    buffer = np.clip(buffer, -1.0, 1.0)
    return buffer


def _slow_motion_window(decided_frame: int | None, main_frame_count: int) -> tuple[int, int] | None:
    """スローモーションで引き伸ばす本編フレームの区間[start, end)を返す。対象外ならNone。"""
    if decided_frame is None:
        return None
    window_frames = min(int(SLOWMO_WINDOW_SECONDS * FPS), decided_frame, main_frame_count)
    start = max(0, decided_frame - window_frames)
    end = min(decided_frame, main_frame_count)
    if end <= start:
        return None
    return start, end


def _build_frame_plan(total_frames: int, window: tuple[int, int] | None) -> list[int]:
    """出力フレームごとに「どのsource frame indexを描画すればよいか」を表す整数のリストを作る。
    windowで指定した区間だけ、同じindexがSLOWMO_FACTOR回連続する(=スローモーション)。
    整数のリストなので、画像そのものを複製するのと違いメモリはごくわずかしか使わない。

    2026-09-20〜21、ヒットストップをこの仕組み(フレーム保持)で実装したことがあったが、
    ユーザー指示により「ノックバックが無い限り速度をゼロにするだけで十分」という、より単純な
    物理側の実装(_apply_weapon_damage参照)に置き換えたため、この関数は元の形に戻した。
    """
    if window is None:
        return list(range(total_frames))
    start, end = window
    plan = list(range(start))
    for i in range(start, end):
        plan.extend([i] * SLOWMO_FACTOR)
    plan.extend(range(end, total_frames))
    return plan


def _stretch_audio_for_slow_motion(audio_mono: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """音声側もwindowの区間だけ最近傍複製でSLOWMO_FACTOR倍に伸ばす。副次的にピッチが下がるが、
    スロー映像との相性がよい効果として意図的に許容している。"""
    start, end = window
    start_sample = int(start / FPS * SAMPLE_RATE)
    end_sample = int(end / FPS * SAMPLE_RATE)
    stretched_window = np.repeat(audio_mono[start_sample:end_sample], SLOWMO_FACTOR)
    return np.concatenate([audio_mono[:start_sample], stretched_window, audio_mono[end_sample:]])


class _FrameCursor:
    """_iter_rendered_frames()の逐次読み出しをラップし、直近1件をキャッシュする。

    frame_planのsource indexは単調非減少(スローモーション区間で同じindexが連続するだけ)なので、
    「現在位置と同じならキャッシュを返す、進む場合だけ内部ジェネレータをnext()する」という単純な
    ロジックで、画像を一切リストに溜めずに動画エンコードの逐次アクセスに対応できる。
    """

    def __init__(self, frame_iterator) -> None:
        self._iterator = frame_iterator
        self._current_index = -1
        self._current_image: Image.Image | None = None

    def get(self, source_index: int) -> Image.Image:
        while self._current_index < source_index:
            self._current_image = next(self._iterator)
            self._current_index += 1
        if self._current_index != source_index or self._current_image is None:
            raise RuntimeError(
                f"フレームカーソルは巻き戻せません(要求={source_index}, 現在={self._current_index})"
            )
        return self._current_image


def render(
    result: SimResult,
    circles: list[CircleEntity],
    out_path: Path,
    camera: str = "fixed",
    slow_motion: bool = False,
    visual_mismatch: dict[int, float] | None = None,
) -> None:
    """同じシミュレーションログを使って毎フレーム描画・合成音を生成し、mp4として書き出す。
    camera/slow_motionは純粋な描画時の選択(同じシミュレーション結果を使い回せる)。
    visual_mismatchは8-5「意図的な期待の裏切り」用: {entity_id: 見た目上の半径倍率}を渡すと、
    実際の物理演算(半径・当たり判定)には一切影響させず、描画上の大きさだけを変える。

    2026-09-09: 動画全体(数千フレーム)を一度にメモリへ溜める実装だとGitHub Actionsの標準
    ランナーでOOM Killされることを実測で確認したため、moviepyのVideoClip(make_frame=...)で
    1フレームずつ描画とエンコードを進めるストリーミング方式に変更した。
    (2026-09-09追記: 一時導入していたリプレイ挿入は、わかりにくい・自滅脱落まで再度見せてしまう
    という理由でユーザー判断により削除した)
    """
    main_frame_count = len(result.frames)  # エピローグを除いた本編フレーム数
    epilogue_count = _epilogue_frame_count(result)
    total_frames = main_frame_count + epilogue_count

    audio_mono = render_audio(result, total_frames)
    main_end_sample = int(main_frame_count / FPS * SAMPLE_RATE)
    raw_main_audio = audio_mono[:main_end_sample]
    tail_audio = audio_mono[main_end_sample:]  # エピローグ区間分(+末尾のパディング)

    slowmo_window = _slow_motion_window(result.decided_frame, main_frame_count) if slow_motion else None

    main_plan = _build_frame_plan(main_frame_count, slowmo_window)
    epilogue_plan = list(range(main_frame_count, total_frames))
    frame_plan = main_plan + epilogue_plan

    main_audio = _stretch_audio_for_slow_motion(raw_main_audio, slowmo_window) if slowmo_window is not None else raw_main_audio
    audio_mono = np.concatenate([main_audio, tail_audio])

    cursor = _FrameCursor(_iter_rendered_frames(result, circles, camera, visual_mismatch))
    safe_inner_size = (int(OUTPUT_WIDTH * SAFE_ZONE_SCALE), int(OUTPUT_HEIGHT * SAFE_ZONE_SCALE))
    safe_paste_pos = ((OUTPUT_WIDTH - safe_inner_size[0]) // 2, (OUTPUT_HEIGHT - safe_inner_size[1]) // 2)

    def make_frame(t: float) -> np.ndarray:
        idx = min(round(t * FPS), len(frame_plan) - 1)
        inner = cursor.get(frame_plan[idx]).resize(safe_inner_size)
        canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (24, 24, 28))
        canvas.paste(inner, safe_paste_pos)
        return np.array(canvas)

    duration = len(frame_plan) / FPS
    video_clip = VideoClip(make_frame, duration=duration)

    audio_stereo = np.column_stack([audio_mono, audio_mono])
    audio_clip = AudioArrayClip(audio_stereo, fps=SAMPLE_RATE).with_duration(duration)
    video_clip = video_clip.with_audio(audio_clip)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_clip.write_videofile(str(out_path), fps=FPS, codec="libx264", audio_codec="aac", logger=None)


def main():
    parser = argparse.ArgumentParser(description="幾何学バトル 技術検証プロトタイプ")
    parser.add_argument("--n-circles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY)
    parser.add_argument("--elasticity", type=float, default=DEFAULT_ELASTICITY)
    parser.add_argument("--rotation-speed", type=float, default=DEFAULT_ROTATION_SPEED)
    parser.add_argument("--hole-width", type=float, default=DEFAULT_HOLE_WIDTH)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--shape", choices=["square", "circle"], default=DEFAULT_SHAPE)
    parser.add_argument("--rule", choices=["hole_fall", "goal_reach", "area_control", "absorb_growth"], default=DEFAULT_RULE)
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION)
    parser.add_argument("--accel-zone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trap", action="store_true")
    parser.add_argument("--camera", choices=["fixed", "tracking"], default="fixed")
    parser.add_argument("--slow-motion", action="store_true")
    parser.add_argument("--player-shape", choices=["circle", "square", "triangle"], default=DEFAULT_PLAYER_SHAPE)
    parser.add_argument("--palette", choices=list(PALETTES.keys()), default=DEFAULT_PALETTE)
    args = parser.parse_args()

    result, circles = simulate(
        args.n_circles,
        args.seed,
        args.gravity,
        args.elasticity,
        args.rotation_speed,
        args.hole_width,
        args.damping,
        args.shape,
        args.rule,
        args.friction,
        args.accel_zone,
        args.trap,
        args.player_shape,
        args.palette,
    )
    summary = summarize(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.rule}_{args.shape}_seed{args.seed}"
    log_path = OUTPUT_DIR / f"geometry_test_v4_{tag}.json"
    video_path = OUTPUT_DIR / f"geometry_test_v4_{tag}.mp4"

    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    render(result, circles, video_path, args.camera, args.slow_motion)
    print(f"video written: {video_path}")


if __name__ == "__main__":
    main()
