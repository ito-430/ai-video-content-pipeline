"""VOICEVOXエンジン（ローカルAPIサーバー）を使った音声合成。"""

import requests

from voice_config import STYLE_ID, emotion_to_params

ENGINE_URL = "http://127.0.0.1:50021"


def synthesize(
    text: str, character: str, emotion: str, se: str = "none",
    speaker: int | None = None, emotion_params: dict | None = None,
) -> bytes:
    """1セリフ分のwavバイト列を返す。

    speakerを指定した場合はSTYLE_ID（討論バラエティ5キャラ用）を参照せず、そのIDを直接使う。
    他チャンネルのようにSTYLE_IDに載っていない話者を使う場合に指定する。

    emotion_paramsを指定した場合、voice_config.emotion_to_params()（討論バラエティ専用の
    感情語彙）は使わず、そのパラメータをそのまま適用する。他チャンネルが独自の感情語彙・
    テンポ設計を持つ場合に指定する。
    """
    if speaker is None:
        speaker = STYLE_ID[character]

    query_res = requests.post(
        f"{ENGINE_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    query_res.raise_for_status()
    query = query_res.json()

    query.update(emotion_params if emotion_params is not None else emotion_to_params(emotion, se))

    synth_res = requests.post(
        f"{ENGINE_URL}/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=30,
    )
    synth_res.raise_for_status()
    return synth_res.content


def engine_is_alive() -> bool:
    try:
        return requests.get(f"{ENGINE_URL}/version", timeout=3).ok
    except requests.RequestException:
        return False
