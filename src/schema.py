# Gemini構造化出力（responseSchema）用のJSONスキーマ定義

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "description": "ディベートのお題"},
        "angle": {"type": "string", "description": "このお題の新規性のある切り口"},
        "bg_query": {
            "type": "string",
            "description": "背景素材を検索するための、テーマに関連する簡潔な英語のキーワード（1〜3語）",
        },
        "format": {"type": "string", "enum": ["shorts", "long", "compilation"]},
        "battle_style": {"type": "string", "enum": ["normal", "rap_battle"]},
        "target_duration_sec": {"type": "number"},
        "dady_appearance": {"type": "boolean", "description": "覆面・ダディが登場する回かどうか"},
        "outcome": {
            "type": "object",
            "properties": {
                "winner": {
                    "type": "string",
                    "enum": ["touma", "yuzu", "sora", "kai", "dady", "draw", "both_lose"],
                },
                "score": {"type": "string", "description": "例: 7:3"},
            },
            "required": ["winner", "score"],
        },
        "debate_positions": {
            "type": "array",
            "description": (
                "討論の主要2キャラ（最も発言数が多い2名。通常はトウマ・ユズ）の立場を、"
                "視聴者が一目でわかる簡易テロップ用に一言で表す。主要2名分のみ入れること。"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string", "enum": ["touma", "yuzu", "sora", "kai", "dady"]},
                    "stance": {
                        "type": "string",
                        "description": "8〜14文字程度の簡潔な立場ラベル（例: 効率重視で賛成派）。フルセンテンスにしないこと",
                    },
                },
                "required": ["character", "stance"],
            },
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "character": {
                        "type": "string",
                        "enum": ["touma", "yuzu", "sora", "kai", "dady"],
                    },
                    "emotion": {"type": "string"},
                    "text": {"type": "string"},
                    "est_duration_sec": {"type": "number"},
                    "se": {
                        "type": "string",
                        "enum": ["attack", "logic_ding", "stumble", "chaos", "win_fanfare", "lose_buzzer", "none"],
                    },
                    "visual_aid": {
                        "type": "string",
                        "description": "主張を補強するピクトグラム（商標・肖像権リスクを避けるため実写真ではなく自前描画の固定アイコンから選ぶ）",
                        "enum": [
                            "none", "thermometer", "chart_up", "chart_down", "money",
                            "clock", "warning", "eco", "house", "phone", "food", "sleep",
                        ],
                    },
                },
                "required": ["order", "character", "emotion", "text", "est_duration_sec", "se", "visual_aid"],
            },
        },
        "title_candidates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "サムネ/タイトルA案・B案（2件）",
        },
        "thumbnail_hooks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "サムネに載せるキャッチコピー案（2件）",
        },
        "description_summary": {
            "type": "string",
            "description": "概要欄に載せる、テーマについての軽い説明文（1〜2文、40〜80文字程度）",
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "概要欄用のテーマ関連ハッシュタグ（#は付けない、3〜5個、日本語中心）",
        },
        "real_data_citation": {
            "type": "object",
            "description": (
                "実在する事実・データを使った場合のみ記録する（雑学としての価値を担保するための"
                "任意の仕組み。使わない回の方が多い）。ren固有の機能ではなく、どのキャラクターが"
                "使ってもよい。"
            ),
            "properties": {
                "used": {
                    "type": "boolean",
                    "description": "本物の事実・データを使用したか。使っていない回はfalseにし、他のフィールドは空でよい",
                },
                "character": {
                    "type": "string",
                    "enum": ["touma", "yuzu", "sora", "kai", "dady"],
                    "description": "その事実を語ったキャラクター",
                },
                "fact_summary": {
                    "type": "string",
                    "description": "使用した実在の事実の要約（誰もが知る一般的な事実・広く知られた公的統計等に限る）",
                },
                "source_name": {
                    "type": "string",
                    "description": (
                        "情報源の名称（例: 気象庁、総務省統計局等）。実在し、動画のような一般的な"
                        "言及・引用に利用して問題ない、広く知られた公的機関・出典に限ること"
                    ),
                },
            },
            "required": ["used", "character", "fact_summary", "source_name"],
        },
    },
    "required": [
        "theme", "angle", "bg_query", "format", "battle_style", "target_duration_sec", "dady_appearance",
        "outcome", "debate_positions", "lines", "title_candidates", "thumbnail_hooks", "description_summary",
        "hashtags", "real_data_citation",
    ],
}

COMMENT_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["theme_suggestion", "character_question", "none"],
            "description": (
                "theme_suggestion=次回ディベートのお題になりそうな意見、"
                "character_question=キャラクターへの質問（Q&A回向け）、"
                "none=どちらでもない、または採用不可（マナー・コンプライアンス上の懸念含む）"
            ),
        },
        "theme": {
            "type": "string",
            "description": "category=theme_suggestionの場合のみ。「〜べきか」形式のお題に変換。それ以外は空文字列",
        },
        "theme_category": {
            "type": "string",
            "enum": ["生活習慣", "お金", "人間関係", "食べ物", "デジタル・SNS", "仕事・学校", "その他"],
            "description": "category=theme_suggestionの場合のみ。themeが最も当てはまるカテゴリ。それ以外は「その他」でよい",
        },
        "question_text": {
            "type": "string",
            "description": (
                "category=character_questionの場合のみ。コメントの文章そのままではなく、"
                "キャラクターが答えやすい自然な質問文に言い換えたもの。それ以外は空文字列"
            ),
        },
    },
    "required": ["category", "theme", "theme_category", "question_text"],
}

SELF_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["OK", "NG"],
            "description": "台本全体として問題がなければOK、少しでも懸念があればNG",
        },
        "flagged_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "該当箇所（例: lines[3] / title_candidates[0]）"},
                    "category": {
                        "type": "string",
                        "enum": ["炎上リスク", "著作権リスク", "不快表現", "恋愛描写", "個人攻撃", "その他"],
                    },
                    "reason": {"type": "string", "description": "具体的に何が問題かの簡潔な説明"},
                },
                "required": ["location", "category", "reason"],
            },
            "description": "verdict=OKの場合は空配列",
        },
    },
    "required": ["verdict", "flagged_items"],
}

FACT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {
            "type": "boolean",
            "description": "Web検索を踏まえた回答から、元の主張が正確で裏付けが取れると言えるか",
        },
        "reason": {"type": "string", "description": "判定理由の簡潔な説明"},
    },
    "required": ["verified", "reason"],
}

SPARK_COMMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "comment_text": {
            "type": "string",
            "description": "視聴者になりすまして投稿するコメント本文（1〜2文、短めでカジュアルな文体）",
        },
        "is_safe": {
            "type": "boolean",
            "description": "炎上・誹謗中傷・センシティブな話題のリスクがなく、投稿して問題ないと自己判断できるか",
        },
    },
    "required": ["comment_text", "is_safe"],
}
