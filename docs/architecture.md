# マルチチャンネル対応アーキテクチャ

## 背景

本システムは当初、単一チャンネル(討論バラエティ形式、キャラクター5体、VOICEVOX音声)のみを前提に構築されていた。`CHANNEL_SHEET_ID` / `GUILD_ID` / `CHANNEL_IDS` / キャラクター設定 / VOICEVOX話者 / テーマジャンル / NGワード誤検知許可リスト / 各種ファイルパス等、ほぼ全域が「1チャンネル = 1グローバル状態」としてハードコードされていた。

その後、複数チャンネルを同時運用する方針が定まり、ジャンル・製作フローもチャンネルごとに異なりうる前提で、以下の設計へ移行した。要件は「チャンネル間で状態を混同しないこと」「トークン消費・実装の重複を最小化すること」の2点。

## 核となる設計原則: 「エンジン(共有コード)」と「チャンネルデータ(個別設定)」の分離

単一チャンネル時代の問題は、あるジャンルの**ロジック**と**データ**が未分離なままハードコードされていたこと。

- キャラクターの人格・口癖・VOICEVOX話者ID・色・NGワード許可リスト・テーマジャンル・プロンプト文言 → すべて「そのチャンネル固有の**データ**」
- 台本生成・動画合成・投稿判定・KPT生成の**ロジック**自体は本来チャンネル非依存にできる

そこで、ロジックは `src/` に残して全チャンネルで共有し、データは `channels/<slug>/` 配下に切り出す設計とした。将来別フォーマットのチャンネルが必要になった場合も、ロジック側に差し替え口(`pipeline`指定)を1つ用意しておけば破綻しない。

## フォルダ構成

```
ai-video-pipeline/
├── channels/
│   ├── _template/                 # 新チャンネル追加時のひな形（コピー元）
│   │   ├── channel.yaml           # id, 表示名, pipeline種別, YouTubeチャンネルID,
│   │   │                          # SheetID, Discordカテゴリ, secrets参照名, 有効/無効
│   │   ├── materials/             # characters/ bgm/ se/ fonts/ stock_photos/
│   │   ├── docs/                  # character_canon等 チャンネル固有ドキュメント
│   │   ├── config/                # characters.py・voice_config.py相当のデータ(JSON化)、
│   │   │                          # ng_words_extra.json（共通辞書への追加分のみ）、
│   │   │                          # prompts/（テーマ生成プロンプトの差分）
│   │   └── state/                 # 台本ログ・各種state json
│   │       └── scripts/
│   └── <channel-slug>/            # 各チャンネルはこの並びで追加（同上構成）
│
├── common/                        # 全チャンネル横断で共有する資産・辞書
│   ├── materials/fonts/           # 汎用フォント（チャンネル側で独自フォント追加も可）
│   └── risk/                      # ng_words.json 共通ベース辞書、pii_filterルール
│                                   # （コンプライアンス系は一元管理し、修正が全チャンネルへ
│                                   #   即反映されるようにする＝重複修正の防止）
│
├── src/
│   ├── core/                      # 真に汎用な基盤（channel非依存）
│   │   ├── channel_config.py      # channel.yamlを読んでChannelConfigを組み立てる
│   │   ├── sheets_client.py       # チャンネル別SheetIDを引数化、共有SheetIDのみ定数
│   │   ├── discord_client.py      # ギルドIDは共有定数のまま、CHANNEL_IDSをチャンネル別に解決
│   │   └── alerting.py, ai_provider.py, pii_filter.py, ng_word_filter.py（共通辞書+
│   │       チャンネル追加分をマージ）, api_usage.py（チャンネル別ログ+集計）,
│   │       emergency_control.py, human_review_log.py 等
│   ├── pipelines/
│   │   └── <format>/              # フォーマット別のパイプライン実装
│   │       └── （generate_script.py, assemble_video.py, tts.py, voice_config.py,
│   │            characters.py, daily_pipeline.py, theme_pool.py 等、
│   │            同フォーマットのチャンネルなら丸ごと再利用しデータだけ差し替える）
│   └── shared_knowledge.py        # 全チャンネル共通の学習データ書き込み口
│
├── .github/
│   ├── workflows/
│   │   ├── _reusable-publish.yml      # 重量パイプライン系の共通実体
│   │   ├── _reusable-enqueue.yml      # 軽量エンキュー系
│   │   ├── _reusable-notify.yml       # 軽量通知系
│   │   └── daily-publish.yml 等       # 呼び出し元。matrix.channelで対象チャンネルを列挙し
│   │                                   # reusable workflowを叩く（channel追加時はmatrixに1行足すだけ）
│   └── scripts/commit_and_push.sh
│
└── docs/                          # システム全体のアーキ文書（channel非依存のもののみ）
```

## Google Sheets / Discord の扱い

- **共通ナレッジシート**: 全チャンネル共通の「成功パターン」「投稿時間帯の一般傾向」を全チャンネルから書き込み・参照する。新チャンネルのコールドスタート対策として機能する。
- **チャンネル別シート**: チャンネルごとに1スプレッドシートを作成し、そのIDを`channel.yaml`に記載する（コード内ハードコードを避ける）。
- **Discord**: 1サーバーに集約し、チャンネルごとにカテゴリを複製する方針。`CHANNEL_IDS`はチャンネルごとの辞書として管理する。
- NGワード誤検知・緊急停止などの「リスク系インシデント」の教訓は共通ナレッジシート側に集約し、あるチャンネルで学んだ教訓を他チャンネルにも即反映できるようにする（多層防御の趣旨に合致）。

## GitHub Actions: reusable workflow + matrix でファイル数を抑える

ワークフロー群は「重量publish系 / 軽量enqueue系 / 軽量通知系」の3パターンに整理でき、`workflow_call`によるreusable workflow化と、呼び出し元でのmatrix展開を組み合わせることで、チャンネル数が増えてもワークフローファイル数の増殖(N種類×Mチャンネル)を回避できる。新チャンネル追加時の変更は「matrix配列に1行追加」のみで済む。

secrets命名は、YouTube認証情報(チャンネルごとに別アカウントのため必須)のみチャンネル別に分け、動的参照で解決する。AI APIキー・Discord Botトークン・サービスアカウント等、1つのクレデンシャルで全チャンネルを賄えるものは共有のまま増やさない(無駄な重複・管理コストを避ける)。

### Actions無料枠への影響に関する教訓

過去に「短間隔・24時間ポーリングで月の無料実行時間枠を使い切り、丸1日以上投稿が止まる」障害を経験した。チャンネル数を増やすとこの種の問題は単純計算で悪化するため、以下を設計に組み込んでいる。

1. 各チャンネルの投稿判定ポーリングを個別のワークフロー実行にせず、**1つのスケジューラワークフローが1回の実行内で全チャンネル分をループ判定する**(GitHub Actionsは1ジョブ最低1分課金されるため、チャンネル数が増えるほど一元化の効果が大きい)
2. ポーリング頻度は、チャンネルごとに投稿可能時間帯が異なりうる前提で見直す
3. 新チャンネル追加のたびに、Actions消費分数の概算を立ててから有効化する

## 学習データ(KPT等)の切り分け

| 種別 | 扱い | 理由 |
|---|---|---|
| KPT履歴・テーマプール・勝率統計・投稿スケジュール状態・投稿済み動画ログ・パターンプール状態・競合チャンネル情報 等 | **チャンネル別**(`channels/<slug>/state/`、チャンネル別Sheet) | 中身がジャンル・キャラクター依存の意思決定データで、混ざると誤った学習になる |
| 成功パターン・投稿時間帯の一般傾向(共通ナレッジシート) | **全チャンネル共通のまま** | 新チャンネルのコールドスタート対策として機能する |
| NGワード辞書・PIIフィルタ・リスク対策ロジック | **コアは共通、チャンネル別に追加分のみ**(`common/risk/` + `channels/<slug>/config/ng_words_extra.json`) | 修正を1箇所に集約し全チャンネルへ即反映するため |
| API利用料ログ | **チャンネル別に記録**、月次で合算集計 | 書き込み競合を避けつつ、チャンネル別のコスト把握もできるようにする |

## 新チャンネル追加手順

1. `channels/_template/` を `channels/<new-slug>/` にコピー
2. `channel.yaml`を記入(表示名・pipeline種別・投稿ジャンル方針等)
3. キャラクター素材・BGM/SE・VOICEVOX話者・キャラ設定を配置
4. Google Sheetsをチャンネル別シートのテンプレートから複製し、IDを`channel.yaml`へ
5. Discordに同構成のカテゴリを複製
6. YouTube OAuth認証情報を新規作成し、チャンネル別のsecretsとしてGitHub Secretsに登録
7. `.github/workflows/`の各matrix配列に新チャンネルを追加(初期は無効化・手動トリガーのみにしておく)
8. テスト動画を生成・確認してから、自動スケジュールを有効化する
