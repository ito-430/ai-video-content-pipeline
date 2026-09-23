"""SimuSphere Arena(4ch目)の「1デプロイにつき1仮説」を可視化するための実験ログ。

2026-09-21、ユーザー指示「検証の単一化: 複数パラメータの同時変更を禁止し、今後は
1回のデプロイにつき、1つの仮説のみを変更・検証する体制をコードレベルで強制する」への対応。

正直な前提として、無人パイプラインのコード自体が「このコミットは何を変えたか」を検知して
複数変更を機械的にブロックすることはできない(それをやるなら専用のCIチェックが必要だが、
このチャンネルの規模では過剰)。そのため、ここでは「強制」を*可視化による牽制*として実装する:
- 変更を加えるたびに`log_experiment()`を呼び、仮説・変更したパラメータ・ベースライン値を記録する。
- 週次KPT(geometry_battle_kpt.py)が`current_experiment()`を必ず表示するため、
  「今アクティブな実験が何か」を見ないまま次の変更を加えることが構造的にやりにくくなる。
- 複数の未結論(status="active")の実験が並んでいたら、それ自体が「単一仮説ルールが
  守られていない」というシグナルになる。

2026-09-21のデータ駆動リセットバッチ(投稿時刻/ルール重み/人数上限/ボス比率/CTR改善/
演出調整を一括変更)は、この仕組み導入前の一時的な例外としてstatus="reset_batch"で
記録し、通常の単一仮説カウントには含めない(次回以降の変更から本則を適用する)。
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "scripts_templates" / "geometry_battle_experiment_log.json"


def _load_log() -> list[dict]:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_log(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def log_experiment(
    hypothesis: str,
    changed_params: list[str],
    commit: str | None = None,
    status: str = "active",
) -> None:
    """新しい実験(=1つの仮説の変更)を記録する。status="reset_batch"は複数変更を伴う
    例外的な一括変更用(通常の変更はこれを使わないこと)。"""
    entries = _load_log()
    entries.append(
        {
            "date": datetime.now(timezone.utc).isoformat(),
            "commit": commit,
            "hypothesis": hypothesis,
            "changed_params": changed_params,
            "status": status,
            "result": None,
        }
    )
    _save_log(entries)


def conclude_experiment(index_from_end: int, result: str, status: str = "concluded") -> None:
    """直近からindex_from_end番目(0が最新)の実験に結論を記録する。"""
    entries = _load_log()
    if not entries or index_from_end >= len(entries):
        return
    target = entries[-(index_from_end + 1)]
    target["result"] = result
    target["status"] = status
    _save_log(entries)


def current_experiment() -> dict | None:
    """最新の実験エントリを返す(無ければNone)。"""
    entries = _load_log()
    return entries[-1] if entries else None


def active_experiments() -> list[dict]:
    """status="active"の実験を全て返す。2件以上あれば「単一仮説ルール違反」の兆候。"""
    return [e for e in _load_log() if e.get("status") == "active"]


def summary_for_kpt() -> str:
    """週次KPT向けの短い要約テキスト。"""
    active = active_experiments()
    if not active:
        return "(現在アクティブな実験はありません)"
    lines = []
    if len(active) > 1:
        lines.append(f"⚠️ アクティブな実験が{len(active)}件同時に存在しています(「1デプロイ1仮説」原則から逸脱している可能性)。")
    for e in active:
        params = "、".join(e.get("changed_params") or [])
        lines.append(f"- [{e.get('date', '')[:10]}] {e.get('hypothesis')} (変更: {params})")
    return "\n".join(lines)
