"""
Gemini APIを使って、5キャラクターのディベート動画の台本をJSONで生成する最小実装。

使い方:
    python src/generate_script.py --theme "お風呂は朝派か夜派か"
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from ai_provider import get_text_provider
from character_lore import add_lore_fact, format_lore_for_prompt, load_lore
from characters import CHARACTERS, DADY_APPEARANCE_PROBABILITY, RAP_BATTLE_PROBABILITY, SE_VOCABULARY
from content_self_check import run_self_check
from diversity_guard import check_novelty, is_too_similar, load_recent_scripts, recent_closing_styles, suppressed_choice
from fact_checker import verify_real_data_citation
from ng_word_filter import check_script
from pattern_pool import select_pattern
from schema import SCRIPT_SCHEMA


class NgWordDetected(Exception):
    """台本にNGワード辞書該当語が見つかった場合に送出する（自動投稿を止めるための専用例外）。"""


class SelfCheckFailed(Exception):
    """自己点検（別会話コンテキストでのAI審査）がNG判定を返した場合に送出する。"""


class TooSimilarTheme(Exception):
    """再生成を1回試みても直近投稿とテーマ・切り口の類似度が閾値を超えたままの場合に送出する
    （[[diversity_guard]]参照。入力テーマ自体が直近と本質的に同一の場合、言い回しを変える
    再生成だけでは解消しないため、テーマ選出からやり直す判断を呼び出し元に委ねる）。"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "scripts_templates" / "scripts"
WIN_STATS_PATH = PROJECT_ROOT / "scripts_templates" / "win_stats.json"
MODEL_NAME = "gemini-flash-lite-latest"  # gemini-2.5-flashは新規APIキーでは提供終了(404)。gemini-flash-latestは断続的に503が続いたため、より安価で当面安定しているlite版を採用

# オープニングの「今日のテーマ」表明を任せるキャラ候補（ダディは対象外）
OPENING_SPEAKER_POOL = ["ren", "mailo", "noa", "baku"]
OPENING_TWO_SPEAKER_PROBABILITY = 0.3


def draw_opening_speakers() -> list[str]:
    if random.random() < OPENING_TWO_SPEAKER_PROBABILITY:
        return random.sample(OPENING_SPEAKER_POOL, 2)
    return [random.choice(OPENING_SPEAKER_POOL)]


def load_win_stats() -> dict:
    if WIN_STATS_PATH.exists():
        try:
            return json.loads(WIN_STATS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"ren": 0, "mailo": 0}


def save_win_stats(stats: dict) -> None:
    WIN_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIN_STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def win_bias_hint(stats: dict) -> str:
    """トウマとユズの勝率が偏りすぎないよう、劣勢側を軽く後押しするヒントを返す。"""
    diff = stats.get("ren", 0) - stats.get("mailo", 0)
    if diff >= 2:
        return "mailo"
    if diff <= -2:
        return "ren"
    return ""


def build_system_prompt() -> str:
    lines = [
        "あなたは、日常のくだらないことについてコミカルに対立するディベート動画の台本作家です。",
        "以下の5キャラクター設定に忠実に、指定されたテーマでディベート台本をJSONで生成してください。",
        "",
        "## キャラクター設定",
    ]
    for key, c in CHARACTERS.items():
        lines.append(f"### {c['display_name']}（character: \"{key}\"）")
        lines.append(f"- 役割: {c['role']}")
        lines.append(f"- 性格: {c['personality']}")
        lines.append(f"- 口癖: {'、'.join(c['catchphrases'])}")
        if c["win_move"]:
            lines.append(f"- 勝利時のムーブ: {c['win_move']}")
        if c["lose_move"]:
            lines.append(f"- 敗北時のムーブ: {c['lose_move']}")
        lines.append(f"- 使用可能な emotion（このキャラ以外の値は使わないこと）: {', '.join(c['emotions'])}")
        lines.append("")

    lines += [
        "## 動画の基本構成",
        "- 基本形はトウマ(ren) vs ユズ(mailo)の対面ディベート。",
        "- ソラ(noa)は全員に共感して対立を丸め込む役。これは議論を放り出しているのではなく、",
        "  「勝敗を明確にしないまま両者を肯定して締める」という、このチャンネル独自の着地の仕方",
        "  そのものである。ソラの一言が入ることで、視聴者には『ここで今回の話は収まった』と",
        "  伝わるようにすること（カオスを加速させる賑やかさと、話を締めくくる役割は矛盾しない）。",
        "- カイ(baku)は要所で気の利いたツッコミを挟む常識人。勝敗を言い渡す審判役ではない。",
        "- 稀にソラやカイ自身がディベートの当事者になってもよい。",
        "- 覆面・ダディ(dady)が登場する回（dady_appearance=trueのとき）は、",
        "  議論が最高潮に達した場面にテーマと無関係な一言とともに乱入させ、",
        "  outcome.winner は必ず \"draw\" か \"both_lose\" に固定すること。",
        "",
        "## オープニング（必須・lines の先頭）",
        "- 指定されたキャラクターが、お題そのものを一言だけ短く読み上げること（1行、10文字前後）。",
        "  「本日のテーマ、○○について」のような前置きも最小限にし、長い説明にしないこと。",
        "  画面には別途タイトルカードとしてお題が大きく表示されるため、セリフ側は短い触れ程度でよい。",
        "- 誰が話すかはユーザープロンプトで指定されるので、必ずそのキャラクターを使うこと。",
        "",
        "## 構成の型（オープニングの後、lines は10〜14行程度・短いセリフでテンポよく）",
        "1. トウマが理論・統計でお題への主張を展開（短く）",
        "2. ユズが真っ向否定ではなく、ひょいと話をずらして受け流す（短く）",
        "3. トウマが畳みかける",
        "4. ユズが切り返す",
        "5. （カイが気の利いた一言を挟んでもよい）",
        "6. トウマ・ユズのやり取りがもう1〜2往復",
        "7. （ソラが両者に共感してカオスを加速させてもよい）",
        "8. 勝者側の短いドヤリアクション（win_moveに沿う）",
        "9. 敗者側の短い悔しがりリアクション（lose_moveに沿う）",
        "10. 最後にソラ(noa)が視聴者に向けて一言添えて締める（1行、短く）。これが「両者を",
        "    肯定して有耶無耶にする」という当チャンネル流の着地であることを意識すること。",
        "    ユーザープロンプトで指定された締めのパターンに従うこと。",
        "- 勝敗は「よって〇〇の勝ち」のような台詞で言い渡さないこと。",
        "  8・9のリアクションの温度差だけで、視聴者が状況的になんとなく察せる作りにすること。",
        "",
        "## トーン・マナー（重要）",
        "- 誰かを一方的に否定して終わる、見ていて不快になる展開は避けること。",
        "- ユズはトウマの主張やデータそのものを馬鹿にしたり無価値扱いしない。あくまで笑いに変えて受け流す。",
        "- ただしユズの主張は弱腰にしないこと。トウマの正論に対して、体感・世論・トレンドという",
        "  別の価値基準から真っ向勝負できる、トウマと同じくらい説得力のある強いカードを切ること。",
        "  単なる茶化しや逃げではなく、「それはそれとして俺はこう思う」という芯のある反論にする。",
        "- カイのツッコミは辛口でも愛嬌・知性を感じるものにし、ただ切り捨てるだけの冷たい一言にしない。",
        "  対象は必ず「意見・状況」に向けること（例:「その理屈、都合よすぎん？」はOK）。",
        "  外見・身体的特徴・性別や年齢等の属性・人格そのものへの攻撃（例: 容姿いじり、",
        "  「アホ」「バカ」等の直接的な人格否定ワード）は禁止。",
        "  特に「ブス」「デブ」「チビ」「ババア」「ジジイ」等、容姿を直接罵る具体的な単語は",
        "  理由の如何を問わず絶対に使わないこと（意見への皮肉として容姿の話題に触れることも避ける）。",
        "- キャラクター間の恋愛描写・恋愛的文脈（ときめき、嫉妬、独占欲等）は一切生成しないこと。",
        "  トウマ・ユズ・ソラ・カイ・覆面ダディの関係性は友情・ライバル関係・仲間関係に限定する。",
        "- ソラの台詞・演出は「共感」であり、色気や媚びを含む言い回しにしないこと。",
        "  性的に消費される方向の表現（体のラインを強調する描写、意味深な表情の演出等）は一切書かないこと。",
        "- 全体として「みんな仲良く不毛な言い合いをしているのが微笑ましい」空気を保つこと。",
        "",
        "## セリフの表記ルール（重要）",
        "- text フィールドには実際に声に出すセリフのみを書くこと。",
        "- 「（メガネを光らせる）」のようなト書き・行動描写・効果の説明を text に含めないこと。",
        "  そうした演出は動画編集側で別途付与するため、セリフは純粋な発話だけにすること。",
        "",
        "## ラップバトル形式（battle_style=\"rap_battle\"のとき）",
        "- 通常の会話ではなく、韻を踏んだバース（ラップ）形式でディベートさせること。",
        "- トウマは経験談・ロジック用語で韻を踏み、ユズはトレンド・スラングで韻を踏む。",
        "- ソラは合いの手・フック（コーラス）的な短い一言を挟んでもよい。",
        "- カイは最後に韻を踏んだ気の利いた一言で締める（勝敗の言い渡しはしない）。",
        "- 1行あたりの文字数は通常より短めにし、リズム感を優先すること。",
        "",
        "## 効果音（se）",
        f"- 各lineには se フィールドを付け、次の語彙からのみ選ぶこと: {', '.join(SE_VOCABULARY)}",
        "- attack=攻撃的な主張の開始、logic_ding=トウマの正論の決め台詞、stumble=ずっこけ/呆れ、",
        "  chaos=ダディ登場などのカオス演出、win_fanfare=勝者のドヤ瞬間、lose_buzzer=敗者の脱力瞬間、",
        "  none=該当なし。",
        "- 全lineの3〜5割程度にだけ se を付け、残りは none にすること。毎行付けると耳障りで、",
        "  逆に序盤〜中盤が全部noneだと単調になる。山場（主張の決め台詞、ずっこけ、勝敗の瞬間）に",
        "  絞って、テンポよく間隔を空けて配置すること。連続する2行に同じseを続けて付けないこと。",
        "",
        "## セリフの長さ・テンポ",
        "- 1行は日本語でおおむね12〜25文字程度。短いキャッチボールをたくさん重ねる方を優先し、",
        "  長い説明台詞を1行に詰め込まないこと。",
        "",
        "## 尺の制約",
        "- lines の est_duration_sec（各セリフの読み上げ秒数の見積もり）の合計が、",
        "  target_duration_sec を超えないようにすること。超えそうな場合はセリフを短く削ること。",
        "- 日本語の読み上げは概ね1秒あたり5文字程度（実測に基づく、やや控えめな見積もり）を目安にすること。",
        "",
        "## 実在の事実・データの扱い（重要・コンプライアンス）",
        "- トウマの主張の根拠は、実在の省庁・機関名（例: 国土交通省、総務省等）を出典として語らせないこと。"
        "視聴者が「本当にそのデータがある」と誤解するリスクがあるため、あくまで「俺の経験上」"
        "「論理的に考えて」といった自分自身の経験・推論をベースにした主張にすること。",
        "- 雑学としての価値を高めるため、3〜4回に1回程度の頻度を目安に、誰もが知る一般的な事実・"
        "広く知られた公的統計等、実在し確実に正しいと言える事実を、トウマに限らずどのキャラクターが"
        "語ってもよい（ディベートの実データを用いた討論としての説得力を高める狙い。ただし頻度を"
        "優先するあまり不確かな数値を無理に使わないこと。自信を持って正しいと言える事実が無ければ"
        "無理に使わなくてよい）。使った場合は real_data_citation.used を true にし、character・"
        "fact_summary・source_name（気象庁、総務省統計局等、実在し引用して問題ない広く知られた出典）"
        "を必ず埋めること。",
        "- 曖昧・具体的すぎる・裏取りできない数値（誤りのリスクがあるもの）は real_data_citation では"
        "使わないこと。使わない回は used=false にし、他のフィールドは空文字列でよい。",
        "",
        "## 主張の補強アイコン（visual_aid）",
        "- トウマの統計・データに基づく主張など、決め台詞のlineに限り、内容に最も近いアイコンを",
        "  スキーマのenumから選ぶこと（例: 温度の話ならthermometer、増加傾向ならchart_up、",
        "  コストの話ならmoney等）。該当しない場合は none にすること。",
        "- 多用しないこと。1本の台本につき1〜2箇所程度に絞る。",
        "",
        "## 主張テロップ（debate_positions）",
        "- 討論の主要2キャラ（最も発言数が多い2名。通常はトウマ・ユズ）について、それぞれの立場を",
        "  8〜14文字程度の簡潔なラベルで debate_positions に入れること（例:「効率重視で賛成派」",
        "  「伝統重視で反対派」）。フルセンテンスではなく短い名詞句にすること。",
        "",
        "## 概要欄用の項目",
        "- description_summary には、視聴者向けにテーマを軽く紹介する1〜2文（40〜80文字程度）を書くこと。",
        "  ネタバレしすぎず、興味を引く程度にする。",
        "- hashtags には、テーマに関連する日本語ハッシュタグを3〜5個、#を付けずに入れること",
        "  （例: [\"エアコン\", \"節電\", \"あるある\"]）。",
        "",
        "## その他",
        "- title_candidates と thumbnail_hooks はそれぞれ2件ずつ、クリックしたくなる短い案を出すこと。",
        "- thumbnail_hooks はtitle_candidatesの言い換えにしないこと。タイトルは説明的に、",
        "  thumbnail_hooksは感情を煽る煽り文句（結末への好奇心を刺激する一言）にすること。",
        "  10〜15文字程度に収め、「！？」「…！！」等の感嘆符を活用して目を引くこと",
        "  （例:「パンの耳論争、ついに決着」ではなく「パンの耳論争、ついに決着！？」）。",
        "- bg_query には、背景素材（フリー写真）を検索するための簡潔な英単語を1〜3語で入れること。",
        "  例: テーマが「エアコンの設定温度」なら \"air conditioner room\"。",
        "- 出力は指定されたJSONスキーマに厳密に従うこと。",
    ]

    lore_block = format_lore_for_prompt(load_lore())
    if lore_block:
        lines.append("")
        lines.append(
            "## 裏設定の扱い（Shorts回では控えめに）\n"
            "- Shortsは新規視聴者の獲得が主目的のため、下記の裏設定は「知っている人にはわかる」程度に"
            "留めること。裏設定の説明や強調はせず、多くの回では一切触れなくてよい。\n"
            "- 触れる場合も、初見の視聴者が置いてけぼりにならない範囲の、さりげない一言に留めること。"
        )
        lines.append(lore_block)

    return "\n".join(lines)


CLOSING_STYLE_OPINION = "opinion"
CLOSING_STYLE_TOPIC_REQUEST = "topic_request"
CLOSING_STYLE_PROMPTS = {
    CLOSING_STYLE_OPINION: "「あなたはどっち派？コメントで教えてね！」のような、この回の対立への意見を募るパターン",
    CLOSING_STYLE_TOPIC_REQUEST: "「次に議論してほしいテーマがあったらコメントで教えてね！」のような、次回のお題を募るパターン",
}
# 意見募集の方が基本形として自然なため6:4程度の配分にする（次回お題募集は稀によいアクセントとして使う）
CLOSING_STYLE_TOPIC_REQUEST_PROBABILITY = 0.4


def draw_closing_style(recent_scripts: list[dict] | None = None) -> str:
    """締めパターンを抽選する。直近2本が同じパターン続きの場合は、量産感を避けるため
    その回だけもう片方に固定する（多様性ガード、[[diversity_guard]]参照）。"""
    candidates = [CLOSING_STYLE_OPINION, CLOSING_STYLE_TOPIC_REQUEST]
    if recent_scripts is not None:
        recent_values = recent_closing_styles(recent_scripts)
        candidates = suppressed_choice(candidates, recent_values, max_streak=2)
    if len(candidates) == 1:
        return candidates[0]
    return CLOSING_STYLE_TOPIC_REQUEST if random.random() < CLOSING_STYLE_TOPIC_REQUEST_PROBABILITY else CLOSING_STYLE_OPINION


def build_user_prompt(
    theme: str,
    angle: str,
    fmt: str,
    target_duration: int,
    dady_appearance: bool,
    battle_style: str,
    opening_speakers: list[str],
    win_hint: str = "",
    closing_style: str = CLOSING_STYLE_OPINION,
) -> str:
    opening_desc = "・".join(CHARACTERS[c]["display_name"] for c in opening_speakers)
    parts = [
        f"テーマ: {theme}",
        f"フォーマット: {fmt}",
        f"バトル形式: {'ラップバトル' if battle_style == 'rap_battle' else '通常の会話ディベート'}",
        f"目標尺: {target_duration}秒",
        f"ダディ登場: {'あり' if dady_appearance else 'なし'}",
        f"オープニングでテーマを表明するキャラ: {opening_desc}",
        f"締めのパターン: {CLOSING_STYLE_PROMPTS[closing_style]}",
    ]
    if angle:
        parts.append(f"想定している切り口: {angle}")
    if win_hint:
        name = CHARACTERS[win_hint]["display_name"]
        parts.append(
            f"勝敗バランス調整: シリーズ全体の勝率を均すため、不自然にならない範囲で"
            f"今回は{name}が勝つ展開を優先すること。"
        )
    parts.append("上記の設定で台本を1本作成してください。")
    return "\n".join(parts)


def validate_and_fix(data: dict) -> dict:
    for line in data.get("lines", []):
        char = line.get("character")
        emotion = line.get("emotion")
        allowed = CHARACTERS.get(char, {}).get("emotions", [])
        if emotion not in allowed:
            print(
                f"[警告] {char} に存在しない emotion '{emotion}' が指定されたため 'base' に補正しました。",
                file=sys.stderr,
            )
            line["emotion"] = "base"

    total = sum(line.get("est_duration_sec", 0) for line in data.get("lines", []))
    target = data.get("target_duration_sec", 0)
    if target and total > target:
        print(
            f"[警告] 見積もり合計尺 {total:.1f}秒 が目標尺 {target}秒 を超えています。台本の再生成や手直しを検討してください。",
            file=sys.stderr,
        )

    # NGワード辞書によるルールベースチェック（Geminiのプロンプト指示だけに頼らない別レイヤー）。
    # 検出した場合は警告に留めず、自動投稿を止めるため例外を送出する。
    ng_findings = check_script(data)
    if ng_findings:
        detail = "; ".join(f"{f['location']}:{f['word']}({f['category']})" for f in ng_findings)
        raise NgWordDetected(f"NGワード辞書に該当する語が台本内に見つかりました: {detail}")

    # 生成時とは別のAPI呼び出し・専用system promptによる自己点検（別会話コンテキスト）。
    self_check = run_self_check(data)
    if self_check.get("verdict") == "NG":
        flagged = self_check.get("flagged_items", [])
        detail = "; ".join(f"{f.get('location')}:{f.get('category')}({f.get('reason')})" for f in flagged)
        raise SelfCheckFailed(f"自己点検でNG判定が出ました: {detail}")

    return data


def generate_script(
    theme: str,
    angle: str,
    fmt: str,
    target_duration: int,
    force_dady: bool | None,
    force_rap: bool | None,
) -> dict:
    if force_dady is None:
        dady_appearance = random.random() < DADY_APPEARANCE_PROBABILITY
    else:
        dady_appearance = force_dady

    # ラップバトルとダディ登場は排他（ダディ登場を優先）
    if dady_appearance:
        battle_style = "normal"
    elif force_rap is None:
        battle_style = "rap_battle" if random.random() < RAP_BATTLE_PROBABILITY else "normal"
    else:
        battle_style = "rap_battle" if force_rap else "normal"

    recent_scripts = load_recent_scripts()  # 量産型対策（多様性ガード・新規性チェック）の材料
    opening_speakers = draw_opening_speakers()
    closing_style = draw_closing_style(recent_scripts)

    win_stats = load_win_stats()
    win_hint = "" if dady_appearance else win_bias_hint(win_stats)

    provider = get_text_provider(MODEL_NAME)
    user_prompt = build_user_prompt(
        theme, angle, fmt, target_duration, dady_appearance, battle_style, opening_speakers, win_hint, closing_style
    )
    data = provider.generate_json(build_system_prompt(), user_prompt, SCRIPT_SCHEMA)
    data["dady_appearance"] = dady_appearance  # 抽選結果はコード側の値を正とする
    data["battle_style"] = battle_style
    data["opening_line_count"] = len(opening_speakers)  # 動画側でタイトルカード表示に使う行数

    # 量産型対策: 直近本数のテーマ・切り口との類似度を自己採点させ、似すぎていれば1回だけ再生成する
    # （何度も再生成するとコストが際限なく増えるため、上限は1回に留める）。
    novelty = check_novelty(data.get("theme", theme), data.get("angle", angle), recent_scripts)
    if is_too_similar(novelty):
        print(
            f"[警告] 直近の投稿と類似度が高いため(類似度={novelty['max_similarity']:.2f}, "
            f"類似テーマ「{novelty['most_similar_theme']}」)、1回だけ再生成します。",
            file=sys.stderr,
        )
        retry_prompt = (
            user_prompt
            + f"\n\n注意: 直近の「{novelty['most_similar_theme']}」という回と切り口が似すぎていると"
            "判定されました。今回は違う角度・違う結末にしてください。"
        )
        data = provider.generate_json(build_system_prompt(), retry_prompt, SCRIPT_SCHEMA)
        data["dady_appearance"] = dady_appearance
        data["battle_style"] = battle_style
        data["opening_line_count"] = len(opening_speakers)

        # 再生成後も似ていないか再確認する。入力テーマ自体が直近と本質的に同一の場合、
        # 言い回しを変えるだけの再生成では解消しないことが実測で確認できたため、
        # ここで直らなければテーマ自体を選び直す判断を呼び出し元（daily_pipeline.py）に委ねる。
        novelty = check_novelty(data.get("theme", theme), data.get("angle", angle), recent_scripts)
        if is_too_similar(novelty):
            raise TooSimilarTheme(
                f"再生成後も直近投稿と類似度が高いままでした(類似度={novelty['max_similarity']:.2f}, "
                f"類似テーマ「{novelty['most_similar_theme']}」)。"
            )

    # 実データ引用のWeb検索裏取り（character_canon_v1.md 2-3）。裏取りできなければ不使用に補正する。
    if data.get("real_data_citation", {}).get("used"):
        data["real_data_citation"] = verify_real_data_citation(data["real_data_citation"])

    data = validate_and_fix(data)

    winner = data.get("outcome", {}).get("winner")
    if winner in ("ren", "mailo"):
        win_stats[winner] = win_stats.get(winner, 0) + 1
        save_win_stats(win_stats)

    # 編集パターンのバンディット選定（docs/editing_pattern_engine_v1.md 4章、stage2）。
    # 動画内に登場する各キャラクターについて1件選び、動画組み立て側（assemble_video.py）が
    # 参照できるよう台本データに含めておく。選定結果は投稿後、実測パフォーマンスで
    # record_performance()にフィードバックする（collect_analytics.py参照）。
    appearing_characters = {line["character"] for line in data.get("lines", [])}
    selected_patterns = {}
    for char in appearing_characters:
        record = select_pattern(char)
        selected_patterns[char] = {"id": record["id"], "genes": record["genes"]}
    data["selected_patterns"] = selected_patterns

    return data


def _character_bible_lines() -> list[str]:
    lines = ["## キャラクター設定"]
    for key, c in CHARACTERS.items():
        lines.append(f"### {c['display_name']}（character: \"{key}\"）")
        lines.append(f"- 役割: {c['role']}")
        lines.append(f"- 性格: {c['personality']}")
        lines.append(f"- 口癖: {'、'.join(c['catchphrases'])}")
        lines.append(f"- 使用可能な emotion（このキャラ以外の値は使わないこと）: {', '.join(c['emotions'])}")
        lines.append("")
    return lines


def build_weekly_special_system_prompt() -> str:
    lines = [
        "あなたは、日常のくだらないことについてコミカルに対立するディベート動画の台本作家です。",
        "今回は「週次総集編+拡張ディベート」という長尺（9分程度）の特別回を作成します。",
        "",
        *_character_bible_lines(),
        "## 全体の構成（3部構成、この順番でlinesを1本の配列にまとめること）",
        "1. 【今週のふりかえり】ソラ(noa)かカイ(baku)のどちらかが、ユーザープロンプトで渡される",
        "   今週の投稿ラインナップ（テーマと勝敗）を、1テーマにつき1〜2行程度で軽快に振り返る。",
        "   全部で6〜10行程度。単調な読み上げにせず、コメントを挟みながらテンポよく。",
        "2. 【拡張ディベート本編】ユーザープロンプトで指定されたテーマで、トウマ(ren) vs ユズ(mailo)の",
        "   ディベートを通常回よりも大幅に多いラウンド数（6〜9往復）で展開する。9分の尺を無理な",
        "   水増しではなく、実際の議論の深掘り・具体例の追加・ソラやカイの掛け合いの厚みで満たすこと。",
        "   通常回と同じ構成",
        "   （トウマの主張→ユズの受け流し→畳みかけ→切り返し、カイのツッコミやソラの共感を随所に挟む）を",
        "   長く・深く繰り返し、勝者側の短いドヤリアクション・敗者側の悔しがりリアクションで締める。",
        "   勝敗は台詞で言い渡さず、リアクションの温度差で伝えること。",
        "3. 【週間チャンピオン発表】ユーザープロンプトで指定された今週の勝敗集計をもとに、",
        "   ソラかカイが「今週のチャンピオン」を発表し、「来週も見てね！次のお題も募集中！」のような",
        "   一言を添えたあと、最後に「ご視聴ありがとうございました！」で締めくくること（1行）。",
        "",
        "## トーン・マナー（重要）",
        "- 誰かを一方的に否定して終わる、見ていて不快になる展開は避けること。",
        "- ユズの主張は弱腰にしないこと。体感・世論・トレンドという別の価値基準から、",
        "  トウマと同じくらい説得力のある強いカードを切ること。",
        "- カイのツッコミは辛口でも愛嬌・知性を感じるものにすること。対象は必ず「意見・状況」に",
        "  向けること。外見・身体的特徴・性別や年齢等の属性・人格そのものへの攻撃（容姿いじり、",
        "  「アホ」「バカ」等の直接的な人格否定ワード）は禁止。特に「ブス」「デブ」「チビ」",
        "  「ババア」「ジジイ」等、容姿を直接罵る具体的な単語は絶対に使わないこと。",
        "- キャラクター間の恋愛描写・恋愛的文脈（ときめき、嫉妬、独占欲等）は一切生成しないこと。",
        "  関係性は友情・ライバル関係・仲間関係に限定する。",
        "- ソラの台詞・演出は「共感」であり、色気や媚びを含む言い回し・性的に消費される方向の",
        "  表現は一切書かないこと。",
        "- 全体として「みんな仲良く不毛な言い合いをしているのが微笑ましい」空気を保つこと。",
        "",
        "## セリフの表記ルール（重要）",
        "- text フィールドには実際に声に出すセリフのみを書くこと（ト書き・行動描写は含めない）。",
        "",
        "## 効果音（se）",
        f"- 各lineには se フィールドを付け、次の語彙からのみ選ぶこと: {', '.join(SE_VOCABULARY)}",
        "- 全lineの3〜5割程度にだけ se を付け、山場に絞ってテンポよく配置すること。",
        "  連続する2行に同じseを続けて付けないこと。",
        "",
        "## 尺の制約（重要）",
        "- lines の est_duration_sec の合計が、target_duration_sec の85〜100%の範囲に収まる",
        "  ボリュームにすること。短く済ませすぎないこと（大幅に不足する場合は、拡張ディベート本編の",
        "  ラウンド数を増やす、具体例を足す、キャラの掛け合いを厚くする等で自然に尺を伸ばすこと。",
        "  無理な間延び・水増しはしないが、指定の長尺回として十分な深掘りをすること）。",
        "- 日本語の読み上げは概ね1秒あたり5文字程度を目安にすること。",
        "",
        "## 実在の事実・データの扱い（重要・コンプライアンス）",
        "- トウマの主張の根拠は、実在の省庁・機関名を出典として語らせないこと。あくまで自分自身の",
        "  経験・推論をベースにした主張にすること。",
        "- 3〜4回に1回程度の頻度を目安に、誰もが知る一般的・広く知られた公的統計等、実在し確実に",
        "  正しい事実を、トウマに限らずどのキャラクターが語ってもよい（不確かな数値を無理に使わないこと）。",
        "  使った場合は real_data_citation を必ず埋めること。使わない回は used=false にすること。",
        "",
        "## その他",
        "- battle_style は \"normal\" 固定、dady_appearance は false 固定にすること。",
        "- title_candidates・thumbnail_hooks・description_summary・hashtags・bg_query・debate_positions は",
        "  通常回と同様に埋めること。",
        "- 出力は指定されたJSONスキーマに厳密に従うこと。",
    ]

    lore_block = format_lore_for_prompt(load_lore())
    if lore_block:
        lines.append("")
        lines.append(
            "## 裏設定の扱い（長尺回では積極的に）\n"
            "- 長尺回は既存ファン向けの深掘りコンテンツのため、下記の裏設定はShorts回と異なり"
            "積極的に活用してよい（言及・ネタにする・掛け合いに絡める等）。"
        )
        lines.append(lore_block)

    return "\n".join(lines)


def build_weekly_special_user_prompt(recent_summary: str, champion: str | None, theme: str, target_duration: int) -> str:
    champion_desc = CHARACTERS[champion]["display_name"] if champion in CHARACTERS else "該当なし（僅差・データ不足のため発表を工夫すること）"
    return (
        f"今週の投稿ラインナップ（テーマと勝敗）:\n{recent_summary}\n\n"
        f"今週の勝敗集計から見た週間チャンピオン: {champion_desc}\n\n"
        f"拡張ディベート本編のテーマ: {theme}\n"
        f"目標尺: {target_duration}秒\n\n"
        "上記の設定で、週次総集編+拡張ディベートの台本を1本作成してください。"
    )


def _total_duration(data: dict) -> float:
    return sum(l.get("est_duration_sec", 0) for l in data.get("lines", []))


UNDERFILL_RETRY_RATIO = 0.75  # この割合を下回ったら1回だけ拡張再生成する


def generate_weekly_special(recent_summary: str, champion: str | None, theme: str, target_duration: int) -> dict:
    provider = get_text_provider(MODEL_NAME)

    def call(extra_note: str = "") -> dict:
        user_prompt = build_weekly_special_user_prompt(recent_summary, champion, theme, target_duration)
        if extra_note:
            user_prompt += f"\n\n{extra_note}"
        return provider.generate_json(build_weekly_special_system_prompt(), user_prompt, SCRIPT_SCHEMA)

    data = call()
    total = _total_duration(data)
    if total < target_duration * UNDERFILL_RETRY_RATIO:
        print(f"[警告] 見積もり尺{total:.0f}秒が目標{target_duration}秒に対して不足のため再生成します。", file=sys.stderr)
        data = call(
            f"前回の生成は見積もり尺が{total:.0f}秒と、目標{target_duration}秒に対して不足していました。"
            "拡張ディベート本編のラウンド数を増やし、具体例やキャラの掛け合いを厚くして、"
            "内容を充実させてください。"
        )

    data["format"] = "long"
    data["dady_appearance"] = False
    data["battle_style"] = "normal"
    data["opening_line_count"] = 0  # 総集編は専用のタイトルカード構成のため通常のオープニングカードは使わない
    return validate_and_fix(data)


def build_qa_system_prompt() -> str:
    lines = [
        "あなたは、日常のくだらないことについてコミカルに対立するディベート動画のキャラクターたちが、",
        "視聴者からの質問に答える「Q&A回（裏設定回）」の台本作家です。長尺（9〜15分相当）の特別回です。",
        "",
        *_character_bible_lines(),
        "## 全体の構成",
        "- 冒頭でソラ(noa)かカイ(baku)が「今日はみんなの質問に答えちゃうよ！」のような一言で開始する。",
        "- ユーザープロンプトで渡される質問を順番に取り上げ、宛先が明示されていればそのキャラクターが、",
        "  明示されていなければ内容に最も合うキャラクターが、そのキャラらしい口調で答える。",
        "- 各質問について、一言答えて終わりにしないこと。他のキャラクターの茶々・追加の質問・",
        "  リアクション・軽い掛け合いを4〜6往復程度添えて、1つの質問あたり十分なボリュームを持たせること",
        "  （通常回のテンポ感を踏襲しつつ、内容の掘り下げで長尺回にふさわしい厚みを出す）。",
        "- 目安として、lines全体の行数が「質問数×7〜9行」程度になるようにすること",
        "  （例: 質問が6件なら全体で45〜55行程度）。この行数を下回りそうな場合は、",
        "  上記の掛け合い・リアクションを増やして厚みを持たせること。",
        "- ソラかカイが「また質問募集するね！」のような一言を添えたあと、最後に",
        "  「ご視聴ありがとうございました！」で締めくくること（1行）。",
        "",
        "## 裏設定の一貫性（最重要）",
        "- 既に確定している裏設定（下記にあれば）と絶対に矛盾しないこと。",
        "- 質問への回答で新しい裏設定を作る場合は、今後のブレを避けるため、具体的かつ一貫して",
        "  語れる内容にすること（例:「誕生日は3月14日」等、曖昧にせず断定的に）。",
        "- キャラクターの根本的な性格・役割（トウマ=論理、ユズ=感覚、ソラ=共感、カイ=常識人）とは",
        "  矛盾しない範囲の裏設定にすること。",
        "",
        "## トーン・マナー（重要）",
        "- 誰かを一方的に否定して終わる、見ていて不快になる展開は避けること。",
        "- カイのツッコミは辛口でも愛嬌・知性を感じるものにすること。対象は必ず「意見・状況」に",
        "  向けること。外見・身体的特徴・性別や年齢等の属性・人格そのものへの攻撃（容姿いじり、",
        "  「アホ」「バカ」等の直接的な人格否定ワード）は禁止。特に「ブス」「デブ」「チビ」",
        "  「ババア」「ジジイ」等、容姿を直接罵る具体的な単語は絶対に使わないこと。",
        "- キャラクター間の恋愛描写・恋愛的文脈（ときめき、嫉妬、独占欲等）は一切生成しないこと。",
        "  関係性は友情・ライバル関係・仲間関係に限定する。",
        "- ソラの台詞・演出は「共感」であり、色気や媚びを含む言い回し・性的に消費される方向の",
        "  表現は一切書かないこと。",
        "- 全体として「みんな仲良く」空気を保つこと。",
        "",
        "## セリフの表記ルール（重要）",
        "- text フィールドには実際に声に出すセリフのみを書くこと（ト書き・行動描写は含めない）。",
        "",
        "## 効果音（se）",
        f"- 各lineには se フィールドを付け、次の語彙からのみ選ぶこと: {', '.join(SE_VOCABULARY)}",
        "- 全lineの3〜5割程度にだけ se を付けること。",
        "",
        "## 尺の制約（重要）",
        "- lines の est_duration_sec の合計が、target_duration_sec の85〜100%の範囲に収まる",
        "  ボリュームにすること。短く済ませすぎないこと（不足する場合は質問1件あたりの掛け合いを",
        "  厚くすること。質問の数自体は増減しない）。",
        "- 日本語の読み上げは概ね1秒あたり5文字程度を目安にすること。",
        "",
        "## その他",
        "- battle_style は \"normal\" 固定、dady_appearance は false 固定にすること。",
        "- outcome.winner は \"draw\"、score は \"0:0\" 固定にすること（勝敗を競う回ではないため）。",
        "- real_data_citation は使わない回として used=false 固定でよい（Q&A回では裏設定の一貫性を優先する）。",
        "- title_candidates・thumbnail_hooks・description_summary・hashtags・bg_query・debate_positions は",
        "  通常回と同様に埋めること。",
        "- 出力は指定されたJSONスキーマに厳密に従うこと。",
    ]

    lore_block = format_lore_for_prompt(load_lore())
    if lore_block:
        lines.append("")
        lines.append(lore_block)

    return "\n".join(lines)


def build_qa_user_prompt(questions: list[str], target_duration: int) -> str:
    questions_text = "\n".join(f"- {q}" for q in questions)
    return f"視聴者からの質問:\n{questions_text}\n\n目標尺: {target_duration}秒\n\n上記の質問に答えるQ&A回の台本を1本作成してください。"


LORE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "new_facts": {
            "type": "array",
            "description": "この回答の中で新しく確定した裏設定（無ければ空配列）",
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string", "enum": ["ren", "mailo", "noa", "baku", "dady"]},
                    "fact": {"type": "string", "description": "今後の台本生成でも一貫させるべき裏設定の要約"},
                },
                "required": ["character", "fact"],
            },
        },
    },
    "required": ["new_facts"],
}


def generate_qa_episode(questions: list[str], target_duration: int) -> dict:
    provider = get_text_provider(MODEL_NAME)

    def call(extra_note: str = "") -> dict:
        user_prompt = build_qa_user_prompt(questions, target_duration)
        if extra_note:
            user_prompt += f"\n\n{extra_note}"
        return provider.generate_json(build_qa_system_prompt(), user_prompt, SCRIPT_SCHEMA)

    data = call()
    total = _total_duration(data)
    if total < target_duration * UNDERFILL_RETRY_RATIO:
        print(f"[警告] 見積もり尺{total:.0f}秒が目標{target_duration}秒に対して不足のため再生成します。", file=sys.stderr)
        data = call(
            f"前回の生成は見積もり尺が{total:.0f}秒と、目標{target_duration}秒に対して不足していました。"
            "各質問での掛け合い・リアクション・具体例を増やし、内容を充実させてください"
            "（質問の数自体は変えないこと）。"
        )

    data["format"] = "long"
    data["dady_appearance"] = False
    data["battle_style"] = "normal"
    data["opening_line_count"] = 0
    data = validate_and_fix(data)

    try:
        extraction = provider.generate_json(
            "このQ&A回の台本から、今後の台本生成でも一貫させるべき新しい裏設定を抽出してください。"
            "既に一般的に分かること（性格や役割等）は対象外。具体的な新情報のみ抽出すること。",
            "Q&A回の全セリフ:\n" + "\n".join(f"{l['character']}: {l['text']}" for l in data.get("lines", [])),
            LORE_EXTRACTION_SCHEMA,
        )
        for item in extraction.get("new_facts", []):
            add_lore_fact(item["character"], item["fact"])
    except Exception as e:
        print(f"[警告] 裏設定の抽出・保存に失敗しました: {e}", file=sys.stderr)

    return data


def save_script(data: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}_{data.get('format', 'shorts')}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ディベート動画の台本をGemini APIで生成する")
    parser.add_argument("--theme", required=True, help="ディベートのお題")
    parser.add_argument("--angle", default="", help="新規性のある切り口（省略可）")
    parser.add_argument("--format", default="shorts", choices=["shorts", "long", "compilation"])
    parser.add_argument("--target-duration", type=int, default=55, help="目標尺（秒）")
    dady_group = parser.add_mutually_exclusive_group()
    dady_group.add_argument("--force-dady", action="store_true", help="ダディを強制登場させる")
    dady_group.add_argument("--no-dady", action="store_true", help="ダディを登場させない")
    rap_group = parser.add_mutually_exclusive_group()
    rap_group.add_argument("--force-rap", action="store_true", help="ラップバトル形式を強制する")
    rap_group.add_argument("--no-rap", action="store_true", help="ラップバトル形式にしない")
    args = parser.parse_args()

    force_dady = True if args.force_dady else (False if args.no_dady else None)
    force_rap = True if args.force_rap else (False if args.no_rap else None)

    data = generate_script(args.theme, args.angle, args.format, args.target_duration, force_dady, force_rap)
    out_path = save_script(data)

    total = sum(line.get("est_duration_sec", 0) for line in data.get("lines", []))
    print(f"テーマ: {data['theme']}")
    print(f"バトル形式: {data['battle_style']}")
    print(f"ダディ登場: {data['dady_appearance']}")
    print(f"勝敗: {data['outcome']['winner']} ({data['outcome']['score']})")
    print(f"見積もり尺: {total:.1f}秒 / 目標 {data['target_duration_sec']}秒")
    print(f"出力先: {out_path}")


if __name__ == "__main__":
    main()
