"""AIテキスト生成のプロバイダー抽象化レイヤー。

現在はGoogle Gemini APIのみを実装しているが、将来的に別のAIモデル・サービスに
差し替えたくなった場合、このモジュール内に新しいプロバイダークラスを追加し、
get_text_provider()の返り値を差し替えるだけで済むようにする（呼び出し側の
generate_script.py等のコードは、プロバイダーの内部実装を一切意識しない）。

使い方:
    from ai_provider import get_text_provider
    provider = get_text_provider(MODEL_NAME)
    data = provider.generate_json(system_instruction, user_prompt, SCHEMA)
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

from api_usage import enforce_monthly_budget, log_usage

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AITextProvider(ABC):
    """テキスト生成AIプロバイダーの共通インターフェース。"""

    @abstractmethod
    def generate_json(
        self, system_instruction: str, user_prompt: str, schema: dict, temperature: float | None = None
    ) -> dict:
        """system_instruction・user_promptを渡し、schema(JSON Schema形式)に従った
        構造化出力(dict)を返す。temperature指定時はその値で上書きする
        （例: 自己点検のような厳格な判定を求める呼び出しで低めに設定する）。"""
        raise NotImplementedError

    @abstractmethod
    def generate_grounded_text(self, prompt: str) -> str:
        """Web検索等の外部情報源で裏付けを取りながら回答するモード（groundingが使える場合のみ）。
        構造化出力(response_schema)とは同時に使えないため、平文の回答を返す
        （事実確認等、検索結果を別途構造化する呼び出しと組み合わせて使う想定）。"""
        raise NotImplementedError

    @abstractmethod
    def generate_json_from_image(
        self,
        system_instruction: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict,
        temperature: float | None = None,
    ) -> dict:
        """画像1枚とテキストプロンプトを渡し、schemaに従った構造化出力(dict)を返す
        （マンガ画像からのコマ/セリフ抽出など、マルチモーダル入力が必要な呼び出し向け）。"""
        raise NotImplementedError

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        reference_images: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str | None = None,
    ) -> bytes:
        """プロンプト(と任意の参照画像)から画像を1枚生成し、PNGバイト列を返す。
        reference_imagesは[(画像バイト列, mime_type), ...]の形式（キャラクターの見た目を
        一貫させるための参照画像同梱など）。"""
        raise NotImplementedError


class GeminiProvider(AITextProvider):
    """Google Gemini API（google-genai SDK）を使った実装。"""

    def __init__(self, model_name: str, api_key: str | None = None):
        from google import genai  # 遅延importでプロバイダー未使用時の依存を減らす

        load_dotenv(PROJECT_ROOT / ".env")
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(".envにGEMINI_API_KEYが設定されていません")

        self.model_name = model_name
        self._client = genai.Client(api_key=resolved_key)

    def generate_json(
        self, system_instruction: str, user_prompt: str, schema: dict, temperature: float | None = None
    ) -> dict:
        from google.genai import types

        config_kwargs = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        enforce_monthly_budget()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        try:
            log_usage(self.model_name, response.usage_metadata)
        except Exception as e:
            print(f"[警告] API使用量ログの記録に失敗しました: {e}", file=sys.stderr)

        return json.loads(response.text)

    def generate_grounded_text(self, prompt: str) -> str:
        from google.genai import types

        enforce_monthly_budget()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        try:
            log_usage(self.model_name, response.usage_metadata)
        except Exception as e:
            print(f"[警告] API使用量ログの記録に失敗しました: {e}", file=sys.stderr)
        return response.text or ""

    def generate_json_from_image(
        self,
        system_instruction: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict,
        temperature: float | None = None,
    ) -> dict:
        from google.genai import types

        config_kwargs = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        enforce_monthly_budget()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), user_prompt],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        try:
            log_usage(self.model_name, response.usage_metadata)
        except Exception as e:
            print(f"[警告] API使用量ログの記録に失敗しました: {e}", file=sys.stderr)

        return json.loads(response.text)

    def generate_image(
        self,
        prompt: str,
        reference_images: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str | None = None,
    ) -> bytes:
        from google.genai import types

        contents: list = []
        for image_bytes, mime_type in reference_images or []:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(prompt)

        config_kwargs: dict = {"response_modalities": ["IMAGE"]}
        if aspect_ratio is not None:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)

        enforce_monthly_budget()
        response = self._client.models.generate_content(
            model=IMAGE_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        try:
            log_usage(IMAGE_MODEL_NAME, response.usage_metadata)
        except Exception as e:
            print(f"[警告] API使用量ログの記録に失敗しました: {e}", file=sys.stderr)

        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                return part.inline_data.data
        raise RuntimeError("Gemini画像生成のレスポンスに画像データが含まれていません")


# 画像生成の既定モデル（2026-09時点の推奨: gemini-2.5-flash-imageはレガシー、
# gemini-3.1-flash-imageが後継。品質重視ならgemini-3-pro-imageへの切り替えを検討）。
IMAGE_MODEL_NAME = "gemini-3.1-flash-image"


# 環境変数 AI_TEXT_PROVIDER で切り替え可能にする（現状はgemini固定だが将来の差し替え口）。
_PROVIDER_REGISTRY = {
    "gemini": GeminiProvider,
}


def get_text_provider(model_name: str) -> AITextProvider:
    provider_key = os.environ.get("AI_TEXT_PROVIDER", "gemini")
    provider_cls = _PROVIDER_REGISTRY.get(provider_key)
    if provider_cls is None:
        raise ValueError(
            f"未知のAI_TEXT_PROVIDER: {provider_key}（利用可能: {', '.join(_PROVIDER_REGISTRY)}）"
        )
    return provider_cls(model_name)
