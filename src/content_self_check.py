"""台本生成とは独立した自己点検（fromdeveloper/claude_code_risk_taisaku_shiji.md 項目6）。

台本生成と同じGemini呼び出し・会話文脈では、生成時の勢いや文脈に引きずられて
自己点検が甘くなりうる。この対策として、生成が完了した台本に対して**別のAPI呼び出し
（別の会話コンテキスト）・専用のsystem prompt**で、炎上リスク・著作権リスク・
不快表現・キャラクターの言動線引き（character_canon_v1.md）のみを審査する。

判定結果はOK/NGの構造化データ（SELF_CHECK_SCHEMA）で受け取り、後から集計・分析しやすくする。
審査役は判定のブレを抑えるため、生成時より低いtemperatureで呼び出す。
"""

import json

from ai_provider import get_text_provider
from schema import SELF_CHECK_SCHEMA

# 生成に使うモデルと同じでよいが、呼び出し自体を完全に独立させる
# （新しいAPIリクエスト＝新しい会話コンテキストになるため、これで別文脈の要件を満たす）。
SELF_CHECK_MODEL_NAME = "gemini-flash-lite-latest"
SELF_CHECK_TEMPERATURE = 0.1  # 厳格・一貫した判定にするため低めに設定

SELF_CHECK_SYSTEM_PROMPT = (
    "あなたは、YouTubeディベート動画『日常のくだらないことをコミカルに議論するチャンネル"
    "（トウマ・ユズ・ソラ・カイ・覆面ダディの5キャラクター）』の炎上リスク・著作権リスク・"
    "不快表現を専門に審査する審査官です。台本の面白さ・完成度は一切評価せず、以下の観点のみで"
    "機械的に判定してください。\n"
    "\n"
    "- 実在の個人・団体・企業への誹謗中傷、名誉毀損のおそれのある表現\n"
    "- 実在の統計・機関名を出典として語っているように誤解される表現（架空の裏付けを事実のように語る等）\n"
    "- 差別的表現、性的に不適切な表現、暴力的表現\n"
    "- キャラクター間の恋愛描写・恋愛的文脈（ときめき、嫉妬、独占欲等）。当チャンネルは"
    "友情・ライバル関係に限定する方針であり、恋愛描写は一切禁止\n"
    "- カイの毒舌が「意見・状況への皮肉」の範囲を超え、外見・属性・人格そのものへの"
    "攻撃になっていないか\n"
    "- ソラの台詞・演出が、共感の範囲を超えて色気・媚びを含む言い回しや、性的に消費される"
    "方向の描写になっていないか\n"
    "- 政治・宗教等のセンシティブな話題への偏った言及\n"
    "- 実在の医療・健康法（特定の症状・治療法・薬の是非等）、実在の投資/金融商品、"
    "個別の法律相談、実際に起きたニュース・事件・時事問題そのものを主題とした内容"
    "（誤情報が広告制限につながりやすく、初期は避ける方針のジャンルのため）。\n"
    "  ただし、「もし〜になったら」のような非現実的な前提を1つだけ置いた机上の空論・"
    "雑学的な思考実験（例: 労働時間制度、定年退職制度、睡眠が不要になった場合等を題材にした"
    "空想上の考察）は、実在の政策論争や実際のニュースそのものを扱っているわけではないため"
    "この項目の対象外とする。判定に迷う場合は、非現実的な前提が明示されているかどうかを基準にすること。\n"
    "\n"
    "少しでも懸念があれば verdict=NG とし、flagged_itemsに該当箇所・カテゴリ・理由を"
    "具体的に記載してください。問題が無ければ verdict=OK、flagged_itemsは空配列にしてください。"
    "疑わしきはNGに倒してください（判断に迷う場合は罰則より安全側を優先する）。"
)


def _script_digest(data: dict) -> str:
    """審査対象を、判定に必要な情報だけに絞って渡す（不要な情報でコストを増やさない）。"""
    digest = {
        "theme": data.get("theme"),
        "lines": [
            {"character": line.get("character"), "text": line.get("text")}
            for line in data.get("lines", [])
        ],
        "title_candidates": data.get("title_candidates"),
        "thumbnail_hooks": data.get("thumbnail_hooks"),
        "description_summary": data.get("description_summary"),
    }
    return json.dumps(digest, ensure_ascii=False)


def run_self_check(data: dict) -> dict:
    """台本データを審査し、SELF_CHECK_SCHEMAに従った結果(dict)を返す。
    生成に使ったのとは別のAPI呼び出しになるため、会話文脈は独立している。"""
    provider = get_text_provider(SELF_CHECK_MODEL_NAME)
    return provider.generate_json(
        SELF_CHECK_SYSTEM_PROMPT,
        f"以下の台本を審査してください。\n{_script_digest(data)}",
        SELF_CHECK_SCHEMA,
        temperature=SELF_CHECK_TEMPERATURE,
    )
