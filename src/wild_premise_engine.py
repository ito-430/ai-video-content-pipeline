"""「非現実的な前提を1つだけ置き、そこから先は現実の知識・論理で真剣に考察する」
スタイルのディベートお題を生成し、専用の wild_premise_queue.txt へ追記する。

「日常のくだらないテーマ」を主軸にしつつも、「え、そんな発想ある！？」という
驚きのあるテーマも選択肢の一つとして混ぜる狙いがある。
例:「仮に飛行能力を得られたら、背骨矯正が一番使えるよな」のように、ファンタジーな
仮定は1つだけに絞り、そこから先は現実の知識と照らし合わせた大真面目な考察にすることで
「雑学」としての深みを出す（ファンタジー設定そのものを掘り下げる話にはしない）。

通常テーマ(theme_pool.json、スコアリングプール)とは別の
専用キューに貯め、daily_pipeline.py側で通常:突飛=2:3程度の比率になるよう抽選して
消費する（このスタイルだけに偏らせない。wild_premise自体は今回のスコアリング対象外）。
"""

import sys
from pathlib import Path

from ai_provider import AITextProvider, get_text_provider
from diversity_guard import load_recent_scripts, quick_duplicate_check
from ingest_ideas import IDEA_SCHEMA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WILD_QUEUE_PATH = PROJECT_ROOT / "scripts_templates" / "wild_premise_queue.txt"
MODEL_NAME = "gemini-flash-lite-latest"
RECENT_PUBLISHED_WINDOW = 15

SYSTEM_PROMPT = """\
あなたは、YouTubeディベート動画『日常のくだらないことをコミカルに議論するチャンネル』の
お題を考えるブレインストーミング担当です。今回は「非現実的な前提を1つだけ置き、
そこから先は現実の知識・論理で大真面目に考察する」という、机上の空論スタイルのお題を
1つ作ってください。

## スタイルの要点
- 非現実的な前提（ファンタジー・SF的な仮定）は**1つだけ**に絞ること。
  例: 「人類が突然空を飛べるようになったら」「1日だけ動物と会話できるとしたら」
  「瞬間移動が実用化したら」「記憶を1つだけ他人と交換できるとしたら」等。
- 前提を置いた**後**の話は、空想を広げるのではなく、できるだけ現実の知識・慣習・
  制度と照らし合わせて大真面目に考察すること。「え、そんな発想ある！？」という
  意外性は、ファンタジー設定そのものではなく、そこから導かれる現実的な帰結の
  発想の飛躍から生まれるようにする。
- 悪い例（ファンタジー設定を掘り下げるだけ）: 「空を飛べたらどんな空を飛びたいか」
- 良い例（現実的な帰結への飛躍）: 「仮に人類が空を飛べるようになったら、真っ先に
  習得すべきスキルは飛行技術より背骨矯正（着地の衝撃対策）ではないか」
  「もし瞬間移動が実用化したら、新幹線のグリーン車という概念はもう不要になるべきか」
- **簡潔にすること。1文・30〜40文字程度を目安にする。** 「〜という場合」「〜という
  前提において」のような回りくどい言い回しや、二重の『』入れ子、「AすべきかそれともB
  すべきか」のような長い両論併記は避け、既存の短いお題（例:「朝食は必ず食べるべきか」
  「エレベーターで会釈はすべきか」）と同じテンポの一文に収めること。
- 政治・宗教・特定の実在人物や企業への言及、扇動的・攻撃的な内容は避けること。
- 実在の医療・健康法、実在の投資/金融商品、個別の法律相談、実際に起きたニュース・事件
  そのものを主題にすることは避けること（誤情報が広告制限につながりやすいジャンルのため）。
  ただし、労働時間制度・定年退職制度等の一般的な社会制度を、非現実的な前提を通した
  空想上の思考実験として扱うこと自体はこのスタイルの本旨であり問題ない
  （実在の政策論争や時事ニュースそのものを扱っているわけではないため）。
  少しでも迷う場合は is_usable=false にすること。
"""


def _existing_queue_themes() -> list[str]:
    """現在キューに残っている（未消費の）テーマ一覧を返す（重複チェックの材料）。"""
    if not WILD_QUEUE_PATH.exists():
        return []
    lines = WILD_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def _recent_published_themes(n: int = RECENT_PUBLISHED_WINDOW) -> list[str]:
    """直近n本（通常/突飛問わず）の公開済みテーマ一覧を返す（重複チェックの材料）。"""
    return [s["theme"] for s in load_recent_scripts(n) if s.get("theme")]


def generate_wild_premise_theme(provider: AITextProvider, avoid_themes: list[str]) -> str:
    user_prompt = "机上の空論スタイルのお題を1つ作ってください。"
    if avoid_themes:
        user_prompt += (
            "\n\n以下は既にキュー内にある、または最近使用済みの発想です。同じ発想・"
            "似た切り口の焼き直しにならないよう、別の非現実的前提を選んでください:\n"
            + "\n".join(f"- {t}" for t in avoid_themes)
        )
    data = provider.generate_json(SYSTEM_PROMPT, user_prompt, IDEA_SCHEMA)
    if data.get("is_usable") and data.get("theme"):
        return data["theme"].strip()
    return ""


def _append_theme(theme: str) -> None:
    WILD_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WILD_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(theme + "\n")


def main():
    provider = get_text_provider(MODEL_NAME)
    avoid_themes = _existing_queue_themes() + _recent_published_themes()
    try:
        theme = generate_wild_premise_theme(provider, avoid_themes)
    except Exception as e:
        print(f"[警告] Gemini呼び出しに失敗しました: {e}", file=sys.stderr)
        return

    if not theme:
        print("[机上の空論テーマ] 今回は採用を見送りました。")
        return

    # プロンプトで回避を指示しても発生しうるため、生成後にも文字列類似度で足切りする
    # （量産型対策・diversity_guardと同じ判定ロジックを再利用し、重複キューへの追加を防ぐ）。
    duplicate = quick_duplicate_check(theme, avoid_themes)
    if duplicate:
        print(f"[机上の空論テーマ] 「{theme}」は既存の「{duplicate}」と似すぎているため見送りました。")
        return

    _append_theme(theme)
    print(f"[机上の空論テーマ] 「{theme}」を追加しました。")


if __name__ == "__main__":
    from alerting import run_with_alert

    run_with_alert(main, "wild_premise_engine.py")
