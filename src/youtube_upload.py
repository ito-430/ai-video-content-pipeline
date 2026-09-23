"""YouTube Data API v3を使った動画アップロード（低レベルAPIラッパー）。

事前準備:
1. Google Cloud Consoleでプロジェクトを作成し、「YouTube Data API v3」を有効化する
2. OAuth同意画面を設定する（公開ステータスは「テスト」でよい。テストユーザーとして
   自分のGoogleアカウント＝チャンネルの管理者アカウントを追加する）
3. 認証情報 → OAuthクライアントIDを作成（アプリケーションの種類：デスクトップアプリ）し、
   JSONをダウンロードして materials/youtube_client_secret.json として保存する
4. 初回のみ以下を実行し、ブラウザでログイン許可する（以後は materials/youtube_token.json が
   自動で使い回されるため、日次実行時にブラウザ操作は不要になる）:
       python src/youtube_upload.py --auth-only

使い方（単体テスト用。通常は publish_video.py 経由で呼ぶ）:
    python src/youtube_upload.py output/xxx.mp4 --title "..." --description-file output/xxx.description.txt
"""

import argparse
import random
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# resumable uploadのnext_chunk()はチャンク単位でHTTPリクエストを送るため、動画1本(数十MB〜)の
# アップロード中に一時的な5xxやネットワーク瞬断が起きる確率は無視できない。Google公式が
# resumable uploadに対して推奨する指数バックオフ+ジッターでのリトライをここに実装し、
# 日次投稿パイプライン全体が一時的な通信エラーだけで丸ごと落ちないようにする。
UPLOAD_RETRIABLE_STATUS_CODES = {500, 502, 503, 504}
UPLOAD_MAX_RETRIES = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_SECRET_PATH = PROJECT_ROOT / "materials" / "youtube_client_secret.json"
TOKEN_PATH = PROJECT_ROOT / "materials" / "youtube_token.json"

# youtube.upload: 動画アップロード / youtube.force-ssl: コメント投稿等の読み書き全般 /
# yt-analytics.readonly: 視聴データの読み取り / youtube.readonly: 競合チャンネル検索・情報取得(search.list, channels.list)。
# 2026-07-11にコメント・分析・競合チャンネル検索機能追加のため拡張。
# スコープを追加した場合、materials/youtube_token.json の再生成（--auth-only の再実行）が必要。
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# 23=Comedy（今回のコンテンツに合わせたデフォルト）。24=Entertainmentも候補。
DEFAULT_CATEGORY_ID = "23"


def _get_credentials(client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH):
    """client_secret_path/token_pathを指定すると、旧チャンネル用の既定パスとは
    別のチャンネル向け認証情報(例: materials/youtube_token_geometry.json)を使い分けられる。
    同じGoogleアカウントが複数のブランドチャンネルを持つ場合、client_secretは共有できるが
    tokenはチャンネル選択の結果が焼き込まれるため、チャンネルごとに別ファイルにすること。
    """
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path.exists():
                raise RuntimeError(
                    f"{client_secret_path} が見つかりません。Google Cloud ConsoleでOAuthクライアント"
                    "（デスクトップアプリ）を作成し、JSONをこのパスに保存してください。"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def get_authenticated_service(client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH):
    """YouTube Data API v3のクライアント（アップロード・コメント等）。"""
    return build("youtube", "v3", credentials=_get_credentials(client_secret_path, token_path))


def get_analytics_service(client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH):
    """YouTube Analytics APIのクライアント（視聴データ読み取り用）。"""
    return build("youtubeAnalytics", "v2", credentials=_get_credentials(client_secret_path, token_path))


def set_thumbnail(video_id: str, thumbnail_path: Path, client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH) -> None:
    """動画にカスタムサムネイルを設定する（長尺動画用）。
    チャンネルの電話番号確認が完了していないと403になる点に注意。"""
    youtube = get_authenticated_service(client_secret_path, token_path)
    media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
    print(f"サムネイル設定完了: {thumbnail_path.name} -> video_id={video_id}")


def set_video_privacy(video_id: str, privacy_status: str, client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH) -> None:
    """既存動画の公開設定を変更する（緊急停止フロー用。誤検知の可能性があるため
    削除は行わず、privacy_statusの変更のみをこの関数の責務とする）。"""
    youtube = get_authenticated_service(client_secret_path, token_path)
    body = {"id": video_id, "status": {"privacyStatus": privacy_status}}
    youtube.videos().update(part="status", body=body).execute()
    print(f"公開設定を変更しました: video_id={video_id} -> privacyStatus={privacy_status}")


def post_comment(video_id: str, text: str, client_secret_path: Path = CLIENT_SECRET_PATH, token_path: Path = TOKEN_PATH) -> dict:
    """動画にトップレベルコメントを投稿する（固定はAPI非対応のため手動で行う必要がある）。"""
    youtube = get_authenticated_service(client_secret_path, token_path)
    body = {
        "snippet": {
            "videoId": video_id,
            "topLevelComment": {"snippet": {"textOriginal": text[:9500]}},
        }
    }
    return youtube.commentThreads().insert(part="snippet", body=body).execute()


def upload_video(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str] | None = None,
    category_id: str = DEFAULT_CATEGORY_ID,
    privacy_status: str = "private",
    client_secret_path: Path = CLIENT_SECRET_PATH,
    token_path: Path = TOKEN_PATH,
) -> str:
    """動画をアップロードし、video_idを返す。

    privacy_status: "private"（デフォルト、安全のためまず非公開）/ "unlisted" / "public"
    """
    youtube = get_authenticated_service(client_secret_path, token_path)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": (tags or [])[:500],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            # COPPA対応: 「子供向けではない」ことの明示は必須項目。記載漏れ防止のため常に明示する。
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    retry_count = 0
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as e:
            if e.resp.status not in UPLOAD_RETRIABLE_STATUS_CODES or retry_count >= UPLOAD_MAX_RETRIES:
                raise
            # 指数バックオフ+ジッター。ジッターを入れるのは、同時刻にスケジュールされた
            # 複数チャンネルの投稿ジョブが同じタイミングで一斉リトライして再び輻輳するのを防ぐため。
            sleep_sec = (2 ** retry_count) + random.uniform(0, 1)
            print(f"アップロード中にHTTP {e.resp.status}。{sleep_sec:.1f}秒後にリトライします({retry_count + 1}/{UPLOAD_MAX_RETRIES})")
            time.sleep(sleep_sec)
            retry_count += 1
            continue

        retry_count = 0  # 直前のチャンクが成功したらリトライカウントをリセットする
        if status:
            print(f"アップロード中... {int(status.progress() * 100)}%")

    video_id = response["id"]
    print(f"アップロード完了: https://youtu.be/{video_id} (privacyStatus={privacy_status})")
    return video_id


def main():
    parser = argparse.ArgumentParser(description="動画をYouTubeにアップロードする（低レベルツール）")
    parser.add_argument("video", nargs="?", help="動画ファイルパス")
    parser.add_argument("--title", help="動画タイトル")
    parser.add_argument("--description-file", help="概要欄テキストファイルのパス")
    parser.add_argument("--tags", nargs="*", default=[], help="タグ（スペース区切り）")
    parser.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    parser.add_argument("--auth-only", action="store_true", help="認証だけ行いtoken.jsonを作成する")
    parser.add_argument("--client-secret", default=str(CLIENT_SECRET_PATH), help="チャンネルごとのOAuthクライアントJSONパス")
    parser.add_argument("--token", default=str(TOKEN_PATH), help="チャンネルごとのtoken.jsonパス")
    args = parser.parse_args()

    client_secret_path = Path(args.client_secret)
    token_path = Path(args.token)

    if args.auth_only:
        get_authenticated_service(client_secret_path, token_path)
        print(f"認証が完了しました。{token_path} を作成しました。")
        return

    if not args.video or not args.title:
        parser.error("videoとtitleは必須です（--auth-onlyを使う場合を除く）")

    description = ""
    if args.description_file:
        description = Path(args.description_file).read_text(encoding="utf-8")

    upload_video(
        Path(args.video), args.title, description, args.tags,
        privacy_status=args.privacy, client_secret_path=client_secret_path, token_path=token_path,
    )


if __name__ == "__main__":
    main()
