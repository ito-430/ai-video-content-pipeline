"""季節のイベント・記念日・国民の祝日に沿ったディベートのお題を自動でテーマプールに追加する。

日次で実行し、数日先までの間近なイベントを検知したら、Geminiでイベントにちなんだ
お題に変換してscripts_templates/theme_pool.json（スコアリング
プール）に追加する。コンプライアンス重視で、宗教的・政治的に踏み込んだ内容や特定ブランドを
想起させる内容は棄却する（ingest_ideas.pyと同じis_usable+theme+categoryゲートを共有）。

固定日のイベント/記念日と、月の第n週の曜日で決まる祝日（ハッピーマンデー制度等）の
両方に対応する。祝日の正確な算出（春分・秋分等）は行わず、広く知られたおおよその日付を
用いる（ブレインストーミング用途のため厳密な暦計算は不要と判断）。
"""

import json
import sys
from datetime import datetime, timedelta

from ai_provider import AITextProvider, get_text_provider
from ingest_ideas import IDEA_SCHEMA_WITH_CATEGORY, PROJECT_ROOT
from theme_pool import add_candidate

STATE_PATH = PROJECT_ROOT / "scripts_templates" / "seasonal_theme_state.json"
MODEL_NAME = "gemini-flash-lite-latest"

LOOKAHEAD_DAYS = 10

# 固定日のイベント・記念日（月, 日, 名称）。特定企業・ブランドに紐づく記念日は避け、
# 広く知られた一般的な季節行事・言葉遊び的な記念日のみを対象にする。
FIXED_EVENTS = [
    (1, 1, "元日"),
    (2, 3, "節分"),
    (2, 11, "建国記念の日"),
    (2, 14, "バレンタインデー"),
    (3, 3, "ひな祭り"),
    (3, 14, "ホワイトデー"),
    (3, 20, "春分の日"),
    (4, 1, "エイプリルフール"),
    (4, 29, "昭和の日"),
    (5, 3, "憲法記念日"),
    (5, 4, "みどりの日"),
    (5, 5, "こどもの日"),
    (6, 21, "夏至"),
    (7, 7, "七夕"),
    (8, 11, "山の日"),
    (8, 13, "お盆"),
    (9, 23, "秋分の日"),
    (10, 31, "ハロウィン"),
    (11, 3, "文化の日"),
    (11, 22, "いい夫婦の日"),
    (11, 23, "勤労感謝の日"),
    (12, 24, "クリスマスイブ"),
    (12, 25, "クリスマス"),
    (12, 31, "大晦日"),
]

# 月の第n○曜日で決まる祝日（ハッピーマンデー制度）。weekday: 0=月曜 … 6=日曜
NTH_WEEKDAY_EVENTS = [
    (1, 0, 2, "成人の日"),
    (7, 0, 3, "海の日"),
    (9, 0, 3, "敬老の日"),
    (10, 0, 2, "スポーツの日"),
]

EVENT_SYSTEM_PROMPT = """\
あなたは、季節のイベント・記念日を『日常のくだらないことをコミカルにディベートする動画』の
お題に変換するアシスタントです。
入力されるのは、間近に迫った季節のイベント・記念日の名称です。
- そのイベントにちなんだ、日常のちょっとした対立軸を「〜べきか」「〜はどっちが正しいか」の
  ようなお題に変換してください。イベントの宗教的・歴史的な重みや政治的な話題には立ち入らず、
  あくまで日常生活の些細な過ごし方・慣習レベルの対立に留めること。
- 特定の企業・商品・ブランドを想起させる内容、宗教的に踏み込んだ内容、政治的な内容は
  is_usable=falseにしてください。
- 医療・健康法・投資/金融・法律相談、ニュース・事件・時事問題に関する内容も、
  誤情報が広告制限につながりやすいジャンルのため is_usable=false にしてください。
- themeが最も当てはまるcategoryも選んでください。
- 例:「クリスマス」→「クリスマスケーキは12/24と25、どっちに食べるべきか」
     「海の日」→「海の日なのに海に行かない人が多いのは、それは悪いことなのか」
"""


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    first_of_month = datetime(year, month, 1)
    delta_days = (weekday - first_of_month.weekday()) % 7
    day = 1 + delta_days + 7 * (n - 1)
    return datetime(year, month, day)


def upcoming_events(today: datetime, within_days: int = LOOKAHEAD_DAYS) -> list[tuple[datetime, str]]:
    window_end = today + timedelta(days=within_days)
    events = []

    for year in (today.year, today.year + 1):
        for month, day, name in FIXED_EVENTS:
            try:
                d = datetime(year, month, day)
            except ValueError:
                continue
            if today <= d <= window_end:
                events.append((d, name))
        for month, weekday, n, name in NTH_WEEKDAY_EVENTS:
            d = _nth_weekday(year, month, weekday, n)
            if today <= d <= window_end:
                events.append((d, name))

    events.sort(key=lambda e: e[0])
    return events


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"queued": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_event_theme(provider: AITextProvider, event_name: str) -> tuple[str, str] | None:
    data = provider.generate_json(EVENT_SYSTEM_PROMPT, f"イベント・記念日: {event_name}", IDEA_SCHEMA_WITH_CATEGORY)
    if data.get("is_usable") and data.get("theme"):
        return data["theme"].strip(), data.get("category", "その他")
    return None


def main():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    state = _load_state()
    queued = set(state.get("queued", []))

    provider = get_text_provider(MODEL_NAME)
    added = 0

    for event_date, event_name in upcoming_events(today):
        key = f"{event_date.strftime('%Y-%m-%d')}:{event_name}"
        if key in queued:
            continue
        try:
            result = _generate_event_theme(provider, event_name)
        except Exception as e:
            print(f"[警告] Gemini呼び出しに失敗しました（{event_name}）: {e}", file=sys.stderr)
            continue

        queued.add(key)
        if result:
            theme, category = result
            add_candidate(theme, source="seasonal", category=category)
            added += 1
            print(f"[季節テーマ] {event_name}({event_date.strftime('%Y-%m-%d')}) → 「{theme}」を追加しました。")
        else:
            print(f"[季節テーマ] {event_name}({event_date.strftime('%Y-%m-%d')}) は採用を見送りました。")

    state["queued"] = sorted(queued)
    _save_state(state)
    print(f"完了: {added}件のイベント連動テーマをテーマプールに追加しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "seasonal_theme_engine.py")
