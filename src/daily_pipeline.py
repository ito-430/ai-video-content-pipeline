"""毎日1本、テーマプールから台本生成→動画組み立て→YouTube投稿までを自動で行う。

Windowsタスクスケジューラ等で1日1回このスクリプトを実行する想定
（設定手順は別途ドキュメント化している）。

通常テーマは scripts_templates/theme_pool.json（スコアリング
プール。Discord/季節イベント/コメント欄/YouTube急上昇/競合チャンネルの5ソースから
供給される）から都度スコア最良の1件を選ぶ。机上の空論スタイルは今回のスコアリング対象外で、
従来通り scripts_templates/wild_premise_queue.txt からFIFOで消費する。
実行のたびに通常:突飛=既定2:3の比率でどちらから引くかを抽選し（片方が空ならもう片方から補う）。
比率は scripts_templates/theme_balance_state.json で上書きでき、週次KPTの実績分析
（generate_kpt.py）を踏まえて調整する想定（自動反映はしない。KPTでの提案→承認後に
開発者によるレビューを経て反映する既存の運用を踏襲する）。
"""

import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from alerting import alert
from assemble_video import assemble
from generate_script import NgWordDetected, SelfCheckFailed, TooSimilarTheme, generate_script, save_script
from ng_word_filter import check_script
from posting_schedule import mark_posted, should_publish_now
from publish_video import publish
from queue_utils import pop_next_line
from spark_comment import post_spark_comment
from theme_pool import select_and_consume_best
from tts import engine_is_alive
from video_log import record_published_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WILD_QUEUE_PATH = PROJECT_ROOT / "scripts_templates" / "wild_premise_queue.txt"
BALANCE_STATE_PATH = PROJECT_ROOT / "scripts_templates" / "theme_balance_state.json"
DEFAULT_WILD_RATIO = 0.6  # 通常:突飛 = 2:3 が既定（2026-07-15、ユーザー指示によりコメントを稼ぎやすい突飛テーマを主軸に引き上げ）

DEFAULT_TARGET_DURATION = 55
# NGワード検出時に何回まで「別のテーマを選び直して」作り直すか（2026-07-14追加）。
# 同じテーマで再生成しても同じNGワードを再生産することが実測で確認できたため、
# テーマ自体を変えずにリトライしても解消しない前提で、テーマ選出からやり直す設計にした。
MAX_THEME_ATTEMPTS = 3
# 公開設定は環境変数 PUBLISH_PRIVACY で切り替え可能にする（コード変更・再デプロイ不要）。
# GitHub Actionsではリポジトリ変数(Settings > Secrets and variables > Actions > Variables)の
# PUBLISH_PRIVACY を "public" に変更するだけで、次回実行から自動公開に切り替わる。
# 信頼できる実績が積み上がるまでは既定値の private（手動公開）を推奨。
DEFAULT_PRIVACY = os.environ.get("PUBLISH_PRIVACY", "private")


def find_engine_binary() -> Path:
    """OSに応じたVOICEVOXエンジンの実行ファイルを探す（Windowsローカル/Linux CI両対応）。"""
    tools_dir = PROJECT_ROOT / "tools"
    exe_name = "run.exe" if platform.system() == "Windows" else "run"
    matches = sorted(tools_dir.glob(f"*/{exe_name}"))
    if not matches:
        raise FileNotFoundError(
            f"{tools_dir} 内にVOICEVOXエンジンの実行ファイル({exe_name})が見つかりません。"
        )
    return matches[0]


def ensure_engine_running(timeout: int = 60) -> None:
    """無人実行でVOICEVOXエンジンの起動忘れによる失敗を防ぐため、未起動なら自動起動する。"""
    if engine_is_alive():
        return
    print("VOICEVOXエンジンが起動していないため、起動します...")
    engine_exe = find_engine_binary()
    if platform.system() != "Windows":
        engine_exe.chmod(0o755)
    subprocess.Popen([str(engine_exe), "--host", "127.0.0.1", "--port", "50021"], cwd=str(engine_exe.parent))
    start = time.time()
    while time.time() - start < timeout:
        if engine_is_alive():
            print("VOICEVOXエンジンの起動を確認しました。")
            return
        time.sleep(2)
    raise RuntimeError(f"VOICEVOXエンジンの起動待ちがタイムアウトしました。{engine_exe} を確認してください。")


def _load_wild_ratio() -> float:
    if BALANCE_STATE_PATH.exists():
        try:
            return json.loads(BALANCE_STATE_PATH.read_text(encoding="utf-8")).get("wild_ratio", DEFAULT_WILD_RATIO)
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_WILD_RATIO


def _pop_normal() -> tuple[str | None, str, str | None]:
    """テーマプール(theme_pool.py)からスコア最良の1件を選ぶ。"""
    candidate = select_and_consume_best()
    if candidate:
        return candidate["text"], "normal", candidate["category"]
    return None, "normal", None


def _pop_wild() -> tuple[str | None, str, str | None]:
    """机上の空論スタイルは今回のスコアリング対象外。従来通りFIFOキューから消費する。"""
    theme = pop_next_line(WILD_QUEUE_PATH)
    return theme, "wild", None


def pop_next_theme() -> tuple[str | None, str, str | None]:
    """(テーマ, theme_type, category)を返す。theme_typeは'normal'または'wild'
    （wildの場合categoryは常にNone）。通常:突飛=既定2:3の比率で抽選するが、
    選んだ方に候補が無ければもう片方から補う。"""
    wild_ratio = _load_wild_ratio()
    prefer_wild = random.random() < wild_ratio
    primary = _pop_wild if prefer_wild else _pop_normal
    secondary = _pop_normal if prefer_wild else _pop_wild

    theme, theme_type, category = primary()
    if theme:
        return theme, theme_type, category
    return secondary()


def _generate_script_with_retry(theme: str):
    """Gemini API側の一時的な不調（過去に確認済みの断続的な503等）に備え、1回だけ自動リトライする。
    NGワード検出(NgWordDetected)・自己点検NG(SelfCheckFailed)・類似テーマ検出(TooSimilarTheme)は
    同じテーマ・同じ入力での再試行では解消しにくい（実測でも同じ判定を再生産することを確認済み）
    ため、ここではリトライせずそのまま呼び出し元に伝播させ、テーマ自体を選び直す判断に委ねる。"""
    try:
        return generate_script(theme, "", "shorts", DEFAULT_TARGET_DURATION, None, None)
    except (NgWordDetected, SelfCheckFailed, TooSimilarTheme):
        raise
    except Exception as e:
        print(f"[警告] 台本生成に失敗したため、1回だけリトライします: {e}", file=sys.stderr)
        time.sleep(5)
        return generate_script(theme, "", "shorts", DEFAULT_TARGET_DURATION, None, None)


def _assemble_and_recheck(script_path: Path):
    """動画を組み立て、実際にアップロードする内容を投稿直前に再度NGワードチェックする
    （台本ファイルが生成後に手動編集された場合等も検知できるよう、生成時のチェックとは別に行う）。
    ここで検出した場合もNgWordDetectedを送出し、呼び出し元でテーマから作り直す。"""
    video_path = assemble(script_path)
    print(f"動画: {video_path}")

    pre_publish_data = json.loads(script_path.read_text(encoding="utf-8"))
    pre_publish_findings = check_script(pre_publish_data)
    if pre_publish_findings:
        detail = "; ".join(f"{f['location']}:{f['word']}({f['category']})" for f in pre_publish_findings)
        raise NgWordDetected(f"投稿直前チェックでNGワードが検出されました: {detail}")
    return video_path


def main():
    theme = None
    try:
        if not should_publish_now():
            print("[投稿スケジュール] 現在は投稿予定時刻ではないため、今回は何もせず終了します。")
            return

        ensure_engine_running()

        data = script_path = video_path = None
        rejected_attempts: list[tuple[str, str]] = []
        for attempt in range(1, MAX_THEME_ATTEMPTS + 1):
            theme, theme_type, category = pop_next_theme()
            if not theme:
                print("[警告] テーマプール・wild_premise_queue.txtのいずれにも候補がありません。", file=sys.stderr)
                alert(
                    "⚠️ テーマ候補が尽きました。scripts_templates/theme_pool.json"
                    "（手動追加時はsource=\"manual\"）にテーマを追加してください。"
                )
                sys.exit(1)

            print(f"テーマ({theme_type}, 試行{attempt}/{MAX_THEME_ATTEMPTS}): {theme}")
            try:
                data = _generate_script_with_retry(theme)
                data["theme_type"] = theme_type
                if category:
                    data["theme_category"] = category
                script_path = save_script(data)
                print(f"台本: {script_path}")
                video_path = _assemble_and_recheck(script_path)
                break
            except (NgWordDetected, SelfCheckFailed, TooSimilarTheme) as e:
                rejected_attempts.append((theme, str(e)))
                print(f"[警告] {type(e).__name__}のため、このテーマは見送り別のテーマで作り直します: {e}", file=sys.stderr)
                data = script_path = video_path = None
                continue
        else:
            detail = "\n".join(f"- {t}: {err}" for t, err in rejected_attempts)
            raise RuntimeError(
                f"NGワード検出・自己点検NG・類似テーマ検出のため{MAX_THEME_ATTEMPTS}回テーマを変えて"
                f"作り直しましたが、いずれも却下されたため中止しました。\n{detail}"
            )

        video_id = publish(script_path, privacy=DEFAULT_PRIVACY)
        print(f"投稿完了: https://youtu.be/{video_id} (privacy={DEFAULT_PRIVACY})")
        mark_posted()
        record_published_video(video_id, data)

        try:
            post_spark_comment(video_id, data)
        except Exception as comment_error:
            print(f"[警告] スパークコメントの投稿に失敗しました: {comment_error}", file=sys.stderr)
    except SystemExit:
        raise
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        alert(
            "🚨 daily_pipeline.py が失敗しました。\n"
            f"テーマ: {theme or '(取得前に失敗)'}\n"
            f"エラー: {type(e).__name__}: {e}\n"
            f"```\n{tb[-1200:]}\n```"
        )
        raise


if __name__ == "__main__":
    main()
