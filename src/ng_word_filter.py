"""NGワード辞書によるルールベースチェック。

台本生成時のAIによる自己判断だけに頼らず、単純な文字列照合による機械的な
チェックを別レイヤーとして追加する。辞書は scripts_templates/ng_words.json
（カテゴリ名→単語リストのJSON、本リポジトリには含めていない）で手動編集できる。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NG_WORDS_PATH = PROJECT_ROOT / "scripts_templates" / "ng_words.json"

# 誤検知防止用の許可リスト。NGワードは単純な部分文字列一致のため、
# 無害な単語の一部として偶然マッチしてしまうことがある（実例: 特定のキャラクターの口癖が
# あるNGワードを部分文字列として含んでしまい、無関係な回で毎回誤検知していた）。
# 該当のNGワードについて、ここに登録された無害な語をテキストから除去してから判定することで、
# その種の誤検知だけを除外する（除去後もNGワードが残っていれば引き続き正しく検出される）。
#
# 特に「ゴミ」「クズ」「カス」のような短い2文字語は、日常語の複合語（ゴミ箱、カステラ、
# パンくず等）の一部として非常に頻繁に出現しうる。日常生活・食べ物系のテーマも扱う
# チャンネルでは、この種の衝突は今後も起こりうる。
# このリストは網羅的ではなく、実際に発生した誤検知を都度追記していく運用とする
# （「殺す」「消えろ」等の暴力的語は、たとえ対象が虫や物であっても文脈依存の判断が
# 必要なため、意図的にここには含めない＝疑わしきは検出する側に倒す）。
_FALSE_POSITIVE_ALLOWLIST: dict[str, list[str]] = {
    "ブス": ["バイブス"],
    "ゴミ": ["ゴミ箱", "ゴミ袋", "ゴミ出し", "ゴミ収集", "生ゴミ", "粗大ゴミ", "ゴミの分別", "ゴミゼロ"],
    "クズ": ["クズ粉", "クズ湯", "クズきり"],  # くず(葛)湯・くず粉は食材名。「パンくず」等は通常ひらがな表記のため元々マッチしない
    "カス": ["カステラ", "カスタード", "カスタネット", "カスタム", "コーヒーカス"],
}


def load_ng_words() -> dict[str, list[str]]:
    if not NG_WORDS_PATH.exists():
        return {}
    data = json.loads(NG_WORDS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def check_text(text: str, ng_words: dict[str, list[str]] | None = None) -> list[tuple[str, str]]:
    """textに含まれるNGワードを (カテゴリ, 単語) のリストで返す（無ければ空）。"""
    if not text:
        return []
    ng_words = ng_words if ng_words is not None else load_ng_words()
    hits = []
    for category, words in ng_words.items():
        for word in words:
            if not word or word not in text:
                continue
            stripped = text
            for safe in _FALSE_POSITIVE_ALLOWLIST.get(word, []):
                stripped = stripped.replace(safe, "")
            if word in stripped:
                hits.append((category, word))
    return hits


def check_script(data: dict) -> list[dict]:
    """台本JSON全体（セリフ・タイトル案・サムネフック・概要文）をNGワードチェックし、
    検出箇所のリストを返す（無ければ空リスト＝問題なし）。"""
    ng_words = load_ng_words()
    findings = []

    for i, line in enumerate(data.get("lines", [])):
        for category, word in check_text(line.get("text", ""), ng_words):
            findings.append({"location": f"lines[{i}]", "character": line.get("character"), "category": category, "word": word})

    for field in ("title_candidates", "thumbnail_hooks"):
        for i, text in enumerate(data.get(field, []) or []):
            for category, word in check_text(text, ng_words):
                findings.append({"location": f"{field}[{i}]", "category": category, "word": word})

    for category, word in check_text(data.get("description_summary", ""), ng_words):
        findings.append({"location": "description_summary", "category": category, "word": word})

    return findings
