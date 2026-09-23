"""コメント採用前の個人情報フィルタ（fromdeveloper/claude_code_risk_taisaku_shiji.md 項目4）。

月末Q&A企画・お題提案として視聴者コメントを採用する前に、本名らしき固有名詞・電話番号・
メールアドレス・SNSアカウント名/URL等を検出し、該当するコメントは採用候補から除外する。

ルールベース（正規表現）のみで判定するシンプルな実装。LLM（Gemini）の一般的な
プライバシー配慮指示だけでは検出漏れがありうるため、それとは別のレイヤーとして
必ずLLM呼び出しより前に実行し、検出した場合はコメント本文をGeminiに送信すらしない
（APIにセンシティブな文字列を渡さないという意味でも安全側）。
"""

import re

# 電話番号（日本国内の主な表記ゆれ: 090-1234-5678 / 03-1234-5678 / 09012345678 等）
_PHONE_RE = re.compile(r"0\d{1,4}[-‐()]?\d{1,4}[-‐()]?\d{3,4}")

# メールアドレス
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# SNSアカウント名・URL（本人特定につながりうるもの）。
# 「@xxxx」単体のハンドル表記はメールアドレスの一部（user@example.com）と誤って
# 重複検出しないよう、直前が英数字/ドットでない場合（＝メールの途中ではない）に限定する。
_SNS_RE = re.compile(
    r"(?:twitter\.com|x\.com|instagram\.com|tiktok\.com|facebook\.com|line\.me)/\S+"
    r"|(?:インスタ|ｲﾝｽﾀ|インスタグラム|ツイッター|X\(旧Twitter\)|LINE\s*ID|ラインID)\s*[:：@]?\s*[A-Za-z0-9_.]{2,}"
    r"|(?<![\w.])@[A-Za-z0-9_]{4,}"  # SNSハンドル(@xxxx)。誤検知を避けるため4文字以上に限定
)

# 住所らしき文字列（郵便番号、都道府県+市区町村+丁目番地パターン）。
# 郵便番号パターンは電話番号中の数字列（例:1234-5678）と誤って重複検出しないよう、
# 直前が数字/ハイフンでない場合（＝電話番号の続きではない）に限定する。
_ADDRESS_RE = re.compile(
    r"(?<![\d\-‐])〒?\d{3}-?\d{4}(?!\d)(?![\-‐]\d)"  # 郵便番号
    r"|(?:北海道|東京都|(?:大阪|京都)府|.{2,3}県).{1,10}(?:市|区|町|村).{0,15}\d+[\-‐]\d+"
)

# 本名らしき自己紹介パターン（例:「田中太郎です」「山田花子と申します」）。
# 姓名の完全な検出は固有名詞辞書がないと困難なため、あくまで簡易ヒューリスティック。
_NAME_SELFINTRO_RE = re.compile(
    r"[一-龥]{1,3}[぀-ゟ]{0,3}[一-龥]{1,3}(?:です|と申します|といいます|でございます)"
)

_PATTERNS: dict[str, re.Pattern] = {
    "phone_number": _PHONE_RE,
    "email": _EMAIL_RE,
    "sns_account": _SNS_RE,
    "address": _ADDRESS_RE,
    "name_selfintro": _NAME_SELFINTRO_RE,
}


def detect_pii(text: str) -> list[str]:
    """検出したPIIカテゴリのリストを返す（何も無ければ空リスト）。"""
    if not text:
        return []
    hits = []
    for category, pattern in _PATTERNS.items():
        if pattern.search(text):
            hits.append(category)
    return hits


def contains_pii(text: str) -> bool:
    return bool(detect_pii(text))
