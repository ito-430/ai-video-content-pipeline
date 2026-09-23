"""KPTスレッドでのユーザーとの意見交換を継続し、確定したらSheetsに反映する。

数時間おきに実行する想定:
- 元メッセージに✅/❌リアクションが付いていれば、その時点の最新案で確定（承認/却下）し、
  KPT履歴シートの「ユーザー判断」「実施状況」列に記録する。
- リアクションがまだ無く、スレッドにユーザーからの新しい返信がある場合は、
  これまでの提案+会話ログを踏まえてGeminiが改訂案を作り、スレッドに再提示する。
  刺激的・炎上リスクが少しでもある提案には、変更前後の具体的な台本例を必ず添える。

「承認」が確定した後の、実際の生成パラメータへの反映自体はここでは行わない
（コード変更を伴う判断のため、開発者による別途対応が必要）。
"""

import json
import sys

from ai_provider import get_text_provider
from discord_client import add_reaction_by_id, channel_id, get_messages_by_id, get_reaction_users_by_id, post_to_thread
from generate_kpt import (
    REVISION_SCHEMA,
    _load_state,
    _save_state,
    format_try_item,
    MODEL_NAME,
)
from sheets_client import CHANNEL_SHEET_ID, update_cells
from shared_knowledge import extract_success_pattern

REVISION_SYSTEM_PROMPT = """\
あなたはYouTubeディベート動画チャンネルのディレクターです。
先週提案した「Try」項目について、担当者とのDiscordスレッドで意見交換が行われています。
これまでの提案内容と、会話ログ（ユーザー・Botのやり取り）を踏まえて、Try項目一覧を改訂してください。

## 絶対厳守（改訂後も全項目に適用。少しでも抵触の疑いがあればis_safeをfalseにすること）
1. 永久的な収益の最大化を優先すること（短期バズ狙いで継続成長を犠牲にしない）
2. 炎上・アカウント停止・個人情報漏洩リスクをゼロにすること

## 刺激性のある提案への対応（重要）
「刺激的」「インパクトを強める」等、炎上・賛否のリスクが少しでもある提案には、
risk_levelを"mild"にした上で、変更前(before_example)と変更後(after_example)の
具体的な台本セリフ例を必ず添えること。リスクがない提案はrisk_level="none"でよい。

## ユーザーの意見の反映
会話ログでユーザーが指摘した懸念・要望を可能な限り反映し、納得してもらえる改訂案にすること。
ユーザーが明確に不要と言った項目は削除し、新しい提案が示唆されていれば追加してもよい。
項目数は元の提案からむやみに増やさないこと（2〜4件程度を目安にする）。
"""


def _finalize(state: dict, pending: dict, approve_count: int, reject_count: int) -> None:
    if approve_count > reject_count:
        judgment, status = "承認", "承認（実際のパラメータ反映は開発側で別途対応）"
        note = "✅ この案で承認として記録しました。実際の反映は次回の開発セッションで対応します。"
    elif reject_count > approve_count:
        judgment, status = "却下", "見送り"
        note = "❌ 今回は見送りとして記録しました。"
    else:
        print("[KPT] ✅❌のリアクション数が同数のため、今回は確定を見送ります。")
        return

    update_cells(CHANNEL_SHEET_ID, "KPT履歴", f"F{pending['row']}:G{pending['row']}", [judgment, status])

    if pending.get("thread_id"):
        try:
            post_to_thread(pending["thread_id"], note)
        except Exception as e:
            print(f"[警告] スレッドへの確定通知に失敗しました: {e}", file=sys.stderr)

    if judgment == "承認":
        try:
            extract_success_pattern(pending.get("keep", []))
        except Exception as e:
            print(f"[警告] 成功パターンの抽出に失敗しました: {e}", file=sys.stderr)

    pending["resolved"] = True
    state["pending"] = pending


def _revise_try_items(pending: dict, transcript: str) -> list[dict]:
    contents = (
        f"現在のTry提案一覧:\n{json.dumps(pending.get('try_items', []), ensure_ascii=False, indent=2)}\n\n"
        f"Discordスレッドでの会話ログ:\n{transcript}\n\n"
        "上記を踏まえてTry提案一覧を改訂してください。"
    )
    provider = get_text_provider(MODEL_NAME)
    data = provider.generate_json(REVISION_SYSTEM_PROMPT, contents, REVISION_SCHEMA)
    return [t for t in data.get("try_items", []) if t.get("is_safe")]


def main():
    state = _load_state()
    pending = state.get("pending")
    if not pending or pending.get("resolved"):
        print("[KPT] 意見交換中のKPTはありません。")
        return

    target = pending.get("current_reaction") or {
        "channel_id": channel_id("kpt報告"),
        "message_id": pending["message_id"],
    }
    try:
        checks = get_reaction_users_by_id(target["channel_id"], target["message_id"], "✅")
        crosses = get_reaction_users_by_id(target["channel_id"], target["message_id"], "❌")
    except Exception as e:
        print(f"[警告] リアクション取得に失敗しました: {e}", file=sys.stderr)
        checks, crosses = [], []

    # Bot自身が最初に付けた✅❌の分を差し引く
    approve_count = max(len(checks) - 1, 0)
    reject_count = max(len(crosses) - 1, 0)

    if approve_count > 0 or reject_count > 0:
        _finalize(state, pending, approve_count, reject_count)
        _save_state(state)
        return

    if not pending.get("thread_id"):
        print("[KPT] スレッドが存在しないため、意見交換なしで✅❌の確定待ちです。")
        return

    messages = get_messages_by_id(pending["thread_id"], limit=100)
    user_messages = [m for m in messages if not m.get("author", {}).get("bot") and (m.get("content") or "").strip()]
    last_seen = pending.get("last_seen_thread_msg_id")
    new_user_messages = (
        [m for m in user_messages if int(m["id"]) > int(last_seen)] if last_seen else user_messages
    )

    if not new_user_messages:
        print("[KPT] スレッドに新しい意見はまだありません。")
        return

    transcript = "\n".join(
        f"{'ユーザー' if not m.get('author', {}).get('bot') else 'Bot'}: {m.get('content', '')}" for m in messages
    )

    try:
        revised = _revise_try_items(pending, transcript)
    except Exception as e:
        print(f"[警告] 改訂案の生成に失敗しました: {e}", file=sys.stderr)
        return

    try_text = "\n".join(format_try_item(t) for t in revised) or "(提案なし)"
    reply = (
        "🔄 いただいたご意見を踏まえた改訂案です。\n\n"
        f"■Try（改訂版）\n{try_text}\n\n"
        "この案で進めてよければ✅、まだ調整したい場合はこのスレッドで続けて教えてください。"
    )
    posted_reply = post_to_thread(pending["thread_id"], reply)
    for emoji in ("✅", "❌"):
        try:
            add_reaction_by_id(pending["thread_id"], posted_reply["id"], emoji)
        except Exception:
            pass

    update_cells(CHANNEL_SHEET_ID, "KPT履歴", f"D{pending['row']}", [try_text])

    pending["try_items"] = revised
    pending["last_seen_thread_msg_id"] = max(int(m["id"]) for m in messages)
    # 以降の✅❌確定判定は、今回の改訂返信メッセージに対して行う
    pending["current_reaction"] = {"channel_id": pending["thread_id"], "message_id": posted_reply["id"]}
    state["pending"] = pending
    _save_state(state)
    print("[KPT] 改訂案をスレッドに投稿しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "follow_up_kpt.py")
