"""週次KPT（Keep/Problem/Try）を自動生成し、Discordの#kpt報告へ投稿してユーザーの承認を得る。

## 絶対厳守のガードレール（project_analytics_kpt メモリ参照、KPTのTry提案すべてに優先する）
1. 永久的な収益の最大値（短期バズより右肩上がりの継続を優先）
2. 炎上・アカウント停止・個人情報漏洩リスクをゼロにする
Try提案はこの2点に抵触しないかをGeminiに自己判定させ、抵触の疑いがあるものは機械的に除外し、
Discordには一切出さない。

## 意見交換フロー（2026-07-11更新）
Try提案は一発承認ではなく、投稿メッセージから作成したDiscordスレッドで意見交換を行う。
- ✅❌リアクション: いつでも押せば、その時点の最新案で確定（承認/却下）する。
- スレッドへの返信: フィードバックとして扱い、follow_up_kpt.py が改訂案を再提示する
  （改訂の継続はfollow_up_kpt.py側で行うため、このファイルは新規週の生成のみを担当）。
前週分が意見交換中（未確定）の場合、今週分の新規生成はスキップし重複を避ける。

刺激的・炎上リスクが少しでもある提案には、変更前後の具体的な台本例を必ず添える
（ユーザーが変化の実際のイメージを持てるようにするため）。
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta

from ai_provider import get_text_provider
from discord_client import add_reaction, channel_id, create_thread, get_recent_messages, post_message, snowflake_from_datetime
from sheets_client import CHANNEL_SHEET_ID, append_row, read_rows

PROJECT_ROOT_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "kpt_state.json")
COMPETITOR_CHANNELS_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "competitor_channels.json")
COMPETITOR_STYLE_PATH = os.path.join(PROJECT_ROOT_ENV, "scripts_templates", "competitor_style_patterns.json")
TOP_COMPETITORS_FOR_BENCHMARK = 5
MODEL_NAME = "gemini-flash-lite-latest"

LOOKBACK_DAYS = 7
# ユーザーの生の声（ネタ提案の雑談・システム提案への反応・雑談フィードバック）もKPTに反映する
VOICE_CHANNELS = ["ネタ投稿", "システム提案", "雑談・フィードバック"]
MAX_MESSAGES_PER_CHANNEL = 20

TRY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestion": {"type": "string"},
        "is_safe": {
            "type": "boolean",
            "description": (
                "この提案が「永久的な収益最大化を優先する」「炎上・アカウント停止・"
                "個人情報漏洩リスクをゼロにする」の2大方針に抵触しないと自己判定できるか"
            ),
        },
        "risk_level": {
            "type": "string",
            "enum": ["none", "mild"],
            "description": "「刺激的」「インパクト強化」等、炎上・賛否のリスクが少しでもあるなら mild にすること",
        },
        "before_example": {
            "type": "string",
            "description": "risk_level=mildの場合のみ、変更前の台本セリフ等の具体例（noneの場合は空文字列）",
        },
        "after_example": {
            "type": "string",
            "description": "risk_level=mildの場合のみ、変更後の台本セリフ等の具体例（noneの場合は空文字列）",
        },
    },
    "required": ["suggestion", "is_safe", "risk_level", "before_example", "after_example"],
}

KPT_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}, "description": "うまくいっている点（2〜4件）"},
        "problem": {"type": "array", "items": {"type": "string"}, "description": "課題点（2〜4件）"},
        "try_items": {
            "type": "array",
            "description": "次週以降に試す改善案（2〜4件）",
            "items": TRY_ITEM_SCHEMA,
        },
    },
    "required": ["keep", "problem", "try_items"],
}

REVISION_SCHEMA = {
    "type": "object",
    "properties": {"try_items": {"type": "array", "items": TRY_ITEM_SCHEMA}},
    "required": ["try_items"],
}

SYSTEM_PROMPT = """\
あなたはYouTubeディベート動画チャンネルのデータアナリスト兼ディレクターです。
直近1週間の投稿実績データをもとに、週次KPT（Keep/Problem/Try）を作成してください。

## 観点（最低限これらを問うこと）
- テーマは面白いか／台本は面白いか／演出で視聴者を退屈させていないか／投稿時刻は最適か
- 入力データの「競合チャンネルベンチマーク」と当チャンネルの実績を比較し、投稿頻度・
  タイトルの付け方等で参考にできる差があればKeep/Problem/Tryに反映すること
  （競合の内容そのものを模倣する提案はしないこと。あくまで頻度・傾向レベルの参考に留める）
- YouTubeのAI生成・大量生成コンテンツに関するポリシー（2026年に「repetitious content」から
  「inauthentic content」へ拡大され、テンプレート的・量産的で人間の創意工夫が見えないコンテンツは
  チャンネル単位で凍結されうる）に照らして、量産感・独自性の欠如・マンネリ化の兆候がないかを
  毎回必ずチェックすること。データ上の問題が無くても、この観点は毎週のProblemで検討すること。
- **テーマ種別（通常／机上の空論）のパフォーマンス比較**: 入力データの「テーマ種別比較」を見て、
  どちらが再生数・維持率で優れているか検討すること。差が明確でサンプル数も十分（両方5本以上が目安）
  なら、`scripts_templates/theme_balance_state.json`の`wild_ratio`（既定0.6=通常:突飛2:3）を
  増減させる具体的な数値を伴うTryを提案してよい。サンプルが少ない場合は結論を急がず
  「もう少しデータを溜めてから判断すべき」とProblemまたはKeepで言及するに留めること。

## 絶対厳守（Try提案すべてに優先する。少しでも抵触の疑いがあればis_safeをfalseにすること）
1. 永久的な収益の最大化（短期的なバズ狙いで右肩上がりの継続的成長を犠牲にする提案は禁止）
2. 炎上・アカウント停止・個人情報漏洩リスクをゼロにする（コンプライアンスに反する・攻撃的・
   センシティブな提案、および上記のYouTube AI/量産コンテンツポリシーに抵触しうる提案は禁止）

## 刺激性のある提案への対応（重要）
「刺激的」「インパクトを強める」等、炎上・賛否のリスクが少しでもある提案には、
risk_levelを"mild"にした上で、変更前(before_example)と変更後(after_example)の
具体的な台本セリフ例を必ず添えること。ユーザーが変化の実際のイメージを持てるようにするため。
リスクがない提案はrisk_level="none"とし、before_example/after_exampleは空文字列でよい。

## Discordでのユーザーの声について
入力データには、視聴データに加えて「#ネタ投稿」「#システム提案」「#雑談・フィードバック」での
直近の生の発言も含まれる。これらは定量データでは見えない現場の感覚・要望・不満であるため、
Keep/Problem/Tryの材料として積極的に反映すること（データと発言が矛盾する場合は両方を併記してよい）。

## 出力方針
- Keep/Problemはデータの傾向を踏まえた具体的な言及にすること。
- Tryは次週すぐ試せる具体的なアクションにすること（例: 「テーマXX系の投稿頻度を上げる」
  「投稿時刻を20時台に寄せてみる」等）。
"""


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _row_number_from_range(updated_range: str) -> int:
    # 例: "'KPT履歴'!A5:G5" -> 5
    m = re.search(r"[A-Za-z]+(\d+)", updated_range.split("!")[-1])
    return int(m.group(1))


def _within_lookback(date_str: str, cutoff: datetime) -> bool:
    try:
        return datetime.fromisoformat(date_str) >= cutoff
    except (ValueError, TypeError):
        return False


def _collect_discord_voice(cutoff: datetime) -> str:
    """ネタ投稿・システム提案・雑談フィードバックの生の発言を、KPTの追加コンテキストとして集める。"""
    after_id = snowflake_from_datetime(cutoff)
    lines = []
    for channel_name in VOICE_CHANNELS:
        try:
            messages = get_recent_messages(channel_name, after_id=after_id, limit=100)
        except Exception as e:
            print(f"[警告] {channel_name}の取得に失敗しました: {e}", file=sys.stderr)
            continue
        user_messages = [
            m for m in messages if not m.get("author", {}).get("bot") and (m.get("content") or "").strip()
        ]
        if not user_messages:
            continue
        lines.append(f"◆{channel_name}")
        lines.extend(f"- {m['content']}" for m in user_messages[:MAX_MESSAGES_PER_CHANNEL])
    return "\n".join(lines) if lines else "(直近のDiscord発言はありません)"


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _theme_type_comparison(recent_videos: list[list]) -> str:
    """通常テーマと机上の空論(wild)テーマのパフォーマンスを比較する。Tryでの
    wild_ratio調整提案の根拠になる（サンプルが少ない場合はその旨も明記する）。"""
    by_type: dict[str, list[list]] = {}
    for r in recent_videos:
        padded = r + [""] * (13 - len(r))
        theme_type = padded[12] or "normal"
        by_type.setdefault(theme_type, []).append(padded)

    lines = []
    for theme_type, label in (("normal", "通常"), ("wild", "机上の空論(wild)")):
        rows = by_type.get(theme_type, [])
        if not rows:
            lines.append(f"- {label}: 0本")
            continue
        views = [float(r[6]) for r in rows if str(r[6]).replace(".", "", 1).isdigit()]
        retention = [float(r[7]) for r in rows if str(r[7]).replace(".", "", 1).isdigit()]
        lines.append(f"- {label}: {len(rows)}本, 平均再生数={_avg(views):.0f}, 平均維持率={_avg(retention):.1f}%")

    if min(len(by_type.get("normal", [])), len(by_type.get("wild", []))) < 5:
        lines.append("  （どちらかのサンプル数が5本未満のため、比較の信頼性はまだ低い）")
    return "\n".join(lines)


def _competitor_benchmark_summary() -> str:
    """競合チャンネル・競合動画スタイル分析（[[project-theme-engine]]）の結果を要約し、
    KPTのTry提案が競合比較を踏まえられるようにする。両ファイルとも週次自動更新（discover-competitors.yml）。"""
    if not os.path.exists(COMPETITOR_CHANNELS_PATH):
        return "(競合チャンネルデータはまだありません)"

    try:
        channels = json.loads(open(COMPETITOR_CHANNELS_PATH, encoding="utf-8").read())
    except (json.JSONDecodeError, OSError):
        return "(競合チャンネルデータの読み込みに失敗しました)"

    style_by_id = {}
    if os.path.exists(COMPETITOR_STYLE_PATH):
        try:
            for s in json.loads(open(COMPETITOR_STYLE_PATH, encoding="utf-8").read()):
                style_by_id[s["channel_id"]] = s
        except (json.JSONDecodeError, OSError):
            pass

    top = sorted(channels, key=lambda c: c.get("subscriber_count") or 0, reverse=True)[:TOP_COMPETITORS_FOR_BENCHMARK]
    lines = []
    for c in top:
        style = style_by_id.get(c["channel_id"], {})
        line = f"- {c['title']}（登録者{c.get('subscriber_count') or '非公開'}）"
        if style:
            line += (
                f": 平均再生数={style.get('avg_view_count', '不明')}, "
                f"投稿間隔={style.get('posting_cadence_days', '不明')}日, "
                f"平均タイトル長={style.get('title_pattern', {}).get('avg_title_length', '不明')}文字"
            )
        lines.append(line)
    return "\n".join(lines) if lines else "(競合チャンネルデータはまだありません)"


def build_summary() -> str:
    cutoff = datetime.now().astimezone() - timedelta(days=LOOKBACK_DAYS)

    video_rows = read_rows(CHANNEL_SHEET_ID, "動画データ")[1:]
    recent_videos = [r for r in video_rows if len(r) > 1 and _within_lookback(r[1], cutoff)]

    timing_rows = read_rows(CHANNEL_SHEET_ID, "投稿時刻実験")[1:]
    recent_timing = [r for r in timing_rows if len(r) > 1 and _within_lookback(r[1], cutoff)]

    lines = [f"■直近{LOOKBACK_DAYS}日間の投稿実績（{len(recent_videos)}本）"]
    for r in recent_videos:
        # 動画データ列: video_id, 投稿日時, テーマ, 形式, 勝者, タイトル, 再生数, 視聴維持率, 高評価数, コメント数, 登録者増減, 推定収益, テーマ種別
        padded = r + [""] * (13 - len(r))
        lines.append(
            f"- テーマ「{padded[2]}」(種別={padded[12] or 'normal'}) 形式={padded[3]} 再生数={padded[6]} 維持率={padded[7]} "
            f"高評価={padded[8]} コメント={padded[9]}"
        )

    lines.append("")
    lines.append("■テーマ種別比較（通常 vs 机上の空論）")
    lines.append(_theme_type_comparison(recent_videos))

    lines.append("")
    lines.append("■投稿時刻実験データ")
    for r in recent_timing:
        # 投稿時刻実験列: video_id, 投稿時刻, 曜日, 24h再生数, 7d再生数, 視聴維持率
        padded = r + [""] * (6 - len(r))
        lines.append(f"- 投稿時刻={padded[1]} 曜日={padded[2]} 24h再生数={padded[3]} 7d再生数={padded[4]}")

    if not recent_videos:
        lines.append("(直近期間のデータがまだ十分に溜まっていません)")

    lines.append("")
    lines.append(f"■競合チャンネルベンチマーク（登録者数上位{TOP_COMPETITORS_FOR_BENCHMARK}件、週次自動更新）")
    lines.append(_competitor_benchmark_summary())

    lines.append("")
    lines.append(f"■Discordでのユーザーの声（直近{LOOKBACK_DAYS}日: ネタ投稿・システム提案・雑談フィードバック）")
    lines.append(_collect_discord_voice(cutoff))

    return "\n".join(lines)


def generate_kpt(summary: str) -> dict:
    provider = get_text_provider(MODEL_NAME)
    return provider.generate_json(SYSTEM_PROMPT, summary, KPT_SCHEMA)


def format_try_item(item: dict) -> str:
    text = f"- {item['suggestion']}"
    if item.get("risk_level") == "mild" and (item.get("before_example") or item.get("after_example")):
        text += (
            f"\n  ⚠️炎上リスクに少し配慮が必要な提案です。\n"
            f"  変更前例: {item.get('before_example', '')}\n"
            f"  変更後例: {item.get('after_example', '')}"
        )
    return text


def format_kpt_message(week_label: str, keep: list[str], problem: list[str], safe_tries: list[dict], rejected_count: int) -> str:
    keep_text = "\n".join(f"- {k}" for k in keep)
    problem_text = "\n".join(f"- {p}" for p in problem)
    try_text = "\n".join(format_try_item(t) for t in safe_tries) or "(今週は提案なし)"

    message = f"📊 週次KPT報告（{week_label}）\n\n■Keep\n{keep_text}\n\n■Problem\n{problem_text}\n\n■Try\n{try_text}\n"
    if rejected_count:
        message += f"\n（ガードレール抵触の疑いにより{rejected_count}件の提案を自動除外しました）\n"
    message += (
        "\nこのメッセージから作成したスレッドで、意見や修正の希望を自由に返信してください。"
        "内容を踏まえて改訂案を再提示します。\n"
        "この案で進めてよければ✅、今回は見送るなら❌でリアクションしてください。"
    )
    return message


def post_and_record_kpt(summary: str, kpt: dict, state: dict) -> None:
    safe_tries = [t for t in kpt.get("try_items", []) if t.get("is_safe")]
    rejected_count = len(kpt.get("try_items", [])) - len(safe_tries)

    week_label = datetime.now().strftime("%Y-%m-%d時点の週")
    message = format_kpt_message(week_label, kpt.get("keep", []), kpt.get("problem", []), safe_tries, rejected_count)

    posted = post_message("kpt報告", message)
    for emoji in ("✅", "❌"):
        try:
            add_reaction("kpt報告", posted["id"], emoji)
        except Exception:
            pass

    thread_id = None
    try:
        thread = create_thread("kpt報告", posted["id"], f"KPT意見交換_{week_label}")
        thread_id = thread["id"]
    except Exception as e:
        print(f"[警告] スレッド作成に失敗しました（意見交換なしで承認/却下のみ運用します）: {e}", file=sys.stderr)

    keep_text = "\n".join(f"- {k}" for k in kpt.get("keep", []))
    problem_text = "\n".join(f"- {p}" for p in kpt.get("problem", []))
    try_text = "\n".join(format_try_item(t) for t in safe_tries) or "(今週は提案なし)"

    append_resp = append_row(
        CHANNEL_SHEET_ID,
        "KPT履歴",
        [week_label, keep_text, problem_text, try_text, summary, "", "意見交換中"],
    )
    row_number = _row_number_from_range(append_resp["updates"]["updatedRange"])

    state["pending"] = {
        "message_id": posted["id"],
        "thread_id": thread_id,
        "row": row_number,
        "keep": kpt.get("keep", []),
        "problem": kpt.get("problem", []),
        "try_items": safe_tries,
        "last_seen_thread_msg_id": None,
        "resolved": False,
        # ✅❌の確定判定は「今一番新しく提示している案」に対して行う（最初は元メッセージ、
        # 改訂のたびにfollow_up_kpt.pyがここを最新の返信メッセージへ更新する）
        "current_reaction": {"channel_id": channel_id("kpt報告"), "message_id": posted["id"]},
    }


def main():
    state = _load_state()
    pending = state.get("pending")
    if pending and not pending.get("resolved"):
        print(
            "[KPT] 前回分がまだ意見交換中/未確定のため、今週の新規生成はスキップします。"
            "follow_up_kpt.py が確定を検知するまでお待ちください。"
        )
        return

    summary = build_summary()
    kpt = generate_kpt(summary)
    post_and_record_kpt(summary, kpt, state)

    _save_state(state)
    print("週次KPTを生成し、Discordへ投稿しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "generate_kpt.py")
