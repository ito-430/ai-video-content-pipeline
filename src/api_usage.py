"""Gemini API使用量(トークン数)を記録し、概算コストを計算する。

各Gemini呼び出し箇所から log_usage() を追加で呼ぶだけの低リスクな仕組み
(既存のGemini呼び出し自体のロジック・戻り値は一切変更しない)。月間の概算コストが
予算上限を超えたら、以降の呼び出しをブロックする(ai_provider.py参照)。

料金は2026-07時点のGemini API公式価格(100万トークンあたり、USD)を参照。
価格改定があれば PRICING_USD_PER_MILLION_TOKENS を更新すること。
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "scripts_templates" / "api_usage_log.json"

JST = timezone(timedelta(hours=9))

# 2026-07時点のGemini API公式価格(USD / 100万トークン、有料枠想定)
PRICING_USD_PER_MILLION_TOKENS = {
    "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    # 2026-09時点のGemini 3.1 Flash Image公式標準価格(ai.google.dev/gemini-api/docs/pricing)。
    # 画像出力は1024px相当で1290トークン程度=1枚あたり約$0.067。DEFAULT_PRICINGで代用すると
    # 150倍近く過小評価されるため、必ずこの表に載せておくこと(新しい画像/動画系モデルを
    # 追加する際も同様に明示登録し、DEFAULT_PRICING任せにしない)。
    "gemini-3.1-flash-image": {"input": 0.50, "output": 60.00},
    # 2026-08時点のAnthropic公式価格。現在は使用箇所なし(台本生成等のテキスト処理は
    # Gemini APIに統一しているため、他プロバイダの価格を混在させても実害はない)。
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}
DEFAULT_PRICING = {"input": 0.10, "output": 0.40}  # 未知モデルはlite相当として概算

# 月間予算の上限と、円換算に使う為替レート。為替は変動するため、実勢より円高寄り
# (=ドル換算した際の上限が実勢より厳しくなる方向)のレートを固定値として使い、
# 安全マージンを確保する。実勢と大きく乖離してきたら見直すこと。
MONTHLY_BUDGET_JPY = 5000
USD_JPY_RATE_FOR_BUDGET = 150
MONTHLY_BUDGET_USD = MONTHLY_BUDGET_JPY / USD_JPY_RATE_FOR_BUDGET


class BudgetExceededError(RuntimeError):
    """当月のGemini API概算コストが月間予算を超えたため、これ以上の呼び出しを止める例外。"""


def log_usage(model: str, usage_metadata) -> None:
    """Gemini呼び出し直後に呼ぶ。usage_metadataはresponse.usage_metadataを渡す。
    失敗してもGemini呼び出し自体には影響させないよう、呼び出し側でtry/exceptすることを推奨。"""
    if usage_metadata is None:
        return
    entries = _load()
    entries.append(
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "model": model,
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
        }
    )
    _save(entries)


def _load() -> list[dict]:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_cost_usd(since: datetime | None = None, until: datetime | None = None) -> float:
    """[since, until) の期間のGemini API利用の概算コスト(USD)を合計する。
    since省略時は下限なし(全期間累計)、until省略時は上限なし。"""
    total = 0.0
    for e in _load():
        try:
            ts = datetime.fromisoformat(e["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if (since is not None and ts < since) or (until is not None and ts >= until):
            continue
        pricing = PRICING_USD_PER_MILLION_TOKENS.get(e.get("model"), DEFAULT_PRICING)
        total += e.get("prompt_tokens", 0) / 1_000_000 * pricing["input"]
        total += e.get("output_tokens", 0) / 1_000_000 * pricing["output"]
    return total


def current_month_cost_usd() -> float:
    """当月(JSTカレンダー月)の概算コスト(USD)を合計する。"""
    now_jst = datetime.now(JST)
    month_start = now_jst.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return summarize_cost_usd(since=month_start)


def enforce_monthly_budget() -> None:
    """当月の概算コストが月間予算(MONTHLY_BUDGET_JPY)を超えていたらBudgetExceededErrorを送出する。

    Gemini API呼び出しの直前に呼ぶ想定(ai_provider.py参照)。呼び出し前にチェックするため、
    予算超過を検知した回の呼び出し自体は実行されない(さらなる支出を防ぐのが目的)。
    月が変わればcurrent_month_cost_usd()の集計対象も自動的に切り替わるため、
    手動リセットなしで翌月には自動的に呼び出しが再開する。
    """
    cost_usd = current_month_cost_usd()
    if cost_usd >= MONTHLY_BUDGET_USD:
        cost_jpy = round(cost_usd * USD_JPY_RATE_FOR_BUDGET)
        raise BudgetExceededError(
            f"今月のGemini API概算利用料が月間予算(¥{MONTHLY_BUDGET_JPY:,})を超えたため、"
            f"これ以降のAPI呼び出しを停止しました(概算: ${cost_usd:.4f} ≒ ¥{cost_jpy:,})。"
            "予算はscripts_templates/api_usage_log.jsonからの概算であり、実際の請求額とは"
            "誤差があり得ます。月が変われば自動的に呼び出しが再開します。"
        )
