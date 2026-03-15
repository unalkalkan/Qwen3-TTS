"""OpenAI-compatible TTS server on top of the Qwen/Qwen3-TTS HuggingFace Space code.

Implements a small subset of the OpenAI API:
  - GET  /v1/models
  - POST /v1/audio/speech

Notes
-----
* This is intended to be run on your own machine (Docker/VM/bare metal).
* Requires a GPU for practical performance. CPU may work but will be extremely slow.
* You must provide HF_TOKEN (Hugging Face token) if the model repos require auth.

Run
---
  export HF_TOKEN=...
  uvicorn openai_server:app --host 0.0.0.0 --port 8000

Example
-------
  curl http://localhost:8000/v1/models

  curl -X POST http://localhost:8000/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen3-tts-customvoice-1.7b",
      "input": "Hello from Qwen3 TTS",
      "voice": "Ryan",
      "response_format": "wav"
    }' --output out.wav
"""

from __future__ import annotations

import io
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from huggingface_hub import snapshot_download
from pydub import AudioSegment

# Optional: authenticate to HF if token is present
try:
    from huggingface_hub import login

    _HF_TOKEN = os.environ.get("HF_TOKEN")
    if _HF_TOKEN:
        login(token=_HF_TOKEN)
except Exception:
    _HF_TOKEN = os.environ.get("HF_TOKEN")


# ---- Logging ----------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen3_tts_openai")

# ---- Model loading ---------------------------------------------------------

MODEL_SIZES = {"0.6b": "0.6B", "1.7b": "1.7B"}
MODEL_TYPES = {
    "customvoice": "CustomVoice",
    "base": "Base",
    "voicedesign": "VoiceDesign",
}


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    model_type: str
    model_size: str
    hf_repo: str


def resolve_model(model: str) -> ResolvedModel:
    """Map OpenAI-style `model` strings to Qwen3-TTS HF repos."""

    # Supported model id patterns:
    #   qwen3-tts-customvoice-1.7b
    #   qwen3-tts-customvoice-0.6b
    #   qwen3-tts-voicedesign-1.7b
    #   qwen3-tts-base-1.7b
    m = (model or "").strip().lower()
    if not m:
        raise HTTPException(400, "model is required")

    if not m.startswith("qwen3-tts-"):
        raise HTTPException(
            400,
            "Unsupported model. Expected something like 'qwen3-tts-customvoice-1.7b'",
        )

    parts = m.split("-")
    # expected: qwen3 - tts - <type> - <size>
    if len(parts) != 4:
        raise HTTPException(400, f"Unsupported model format: {model}")

    _, _, typ, size = parts
    if typ not in MODEL_TYPES:
        raise HTTPException(400, f"Unsupported model type: {typ}")
    if size not in MODEL_SIZES:
        raise HTTPException(400, f"Unsupported model size: {size}")

    model_type = MODEL_TYPES[typ]
    model_size = MODEL_SIZES[size]
    hf_repo = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"

    return ResolvedModel(model_id=m, model_type=model_type, model_size=model_size, hf_repo=hf_repo)


_loaded_models: Dict[Tuple[str, str], "Qwen3TTSModel"] = {}


def _snapshot_download_with_logs(repo_id: str) -> str:
    """Download HF snapshot with clear logs and progress bars.

    `huggingface_hub.snapshot_download()` uses tqdm internally; in Docker logs this
    is often the easiest way to see progress.
    """
    # Ensure progress bars are enabled
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    logger.info("Downloading model snapshot from Hugging Face: %s", repo_id)
    t0 = time.time()
    path = snapshot_download(repo_id)
    dt = time.time() - t0
    logger.info("Snapshot ready: %s (%.1fs)", path, dt)
    return path


def get_model(model_type: str, model_size: str):
    """Load and cache the underlying Qwen3TTSModel."""
    key = (model_type, model_size)
    if key in _loaded_models:
        return _loaded_models[key]

    from qwen_tts import Qwen3TTSModel

    hf_repo = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
    model_path = _snapshot_download_with_logs(hf_repo)

    # GPU-only by default (since CPU inference will be extremely slow).
    # Set REQUIRE_CUDA=0 to allow CPU fallback (not recommended).
    import torch

    require_cuda = os.environ.get("REQUIRE_CUDA", "1").strip() not in ("0", "false", "False")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. "
            "Make sure you run the container with NVIDIA runtime (e.g. --gpus all) and that CUDA is working."
        )

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )

    _loaded_models[key] = tts
    return tts


# ---- OpenAI-compatible API -------------------------------------------------

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible Server", version="0.1.0")


@app.get("/v1/models")
def list_models():
    data = []
    for typ in ("customvoice", "base", "voicedesign"):
        for size in ("0.6b", "1.7b"):
            # VoiceDesign is only available for 1.7B in the Space.
            if typ == "voicedesign" and size != "1.7b":
                continue
            data.append(
                {
                    "id": f"qwen3-tts-{typ}-{size}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                }
            )
    return {"object": "list", "data": data}


@app.get("/v1/voices")
def list_voices(model: str = "qwen3-tts-customvoice-1.7b"):
    """Return available voices/speakers for the TTS models.
    
    For CustomVoice models, these are predefined speaker names fetched from the model.
    For VoiceDesign models, you can provide any natural language description.
    
    Args:
        model: Model ID to query for available voices (default: qwen3-tts-customvoice-1.7b)
    """
    try:
        resolved = resolve_model(model)
        tts = get_model(resolved.model_type, resolved.model_size)
        speakers = tts.get_supported_speakers()
        
        if speakers is None:
            # Model doesn't provide a speakers list (e.g., VoiceDesign or Base models)
            return {"object": "list", "data": [], "note": f"Model {model} accepts any voice description"}
        
        data = [
            {
                "id": speaker,
                "name": speaker.capitalize(),
                "object": "voice",
            }
            for speaker in speakers
        ]
        
        return {"object": "list", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch voices: {type(e).__name__}: {e}")


# OpenAI TTS: https://platform.openai.com/docs/api-reference/audio/createSpeech
# We'll implement a compatible subset.

ResponseFormat = Literal["wav", "pcm", "mp3"]


def _wav_bytes(audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


@app.post("/v1/audio/speech")
def audio_speech(payload: Dict):
    """OpenAI-compatible TTS endpoint.

    Expected JSON body fields (subset):
      - model: str (required)
      - input: str (required)
      - voice: str (required for CustomVoice; mapped to speaker)
      - response_format: "wav" (default), "pcm" (16-bit little-endian), or "mp3"
      - speed: ignored for now

    For Qwen3-TTS mapping:
      - CustomVoice -> speaker = voice
      - VoiceDesign -> instruct = voice (treat voice string as description)
    """

    model = payload.get("model")
    text = payload.get("input")
    voice = payload.get("voice")
    response_format = (payload.get("response_format") or "wav").lower()

    if not model:
        raise HTTPException(400, "model is required")
    if not text or not str(text).strip():
        raise HTTPException(400, "input is required")

    resolved = resolve_model(model)
    tts = get_model(resolved.model_type, resolved.model_size)

    try:
        if resolved.model_type == "CustomVoice":
            if not voice:
                raise HTTPException(400, "voice is required for customvoice")
            wavs, sr = tts.generate_custom_voice(
                text=str(text).strip(),
                language=payload.get("language") or "Auto",
                speaker=str(voice).strip().lower().replace(" ", "_"),
                instruct=payload.get("style") or payload.get("instruct") or payload.get("instructions"),
                non_streaming_mode=True,
                max_new_tokens=int(payload.get("max_new_tokens") or 2048),
            )
        elif resolved.model_type == "VoiceDesign":
            # Treat `voice` as a natural-language voice description.
            if not voice:
                raise HTTPException(400, "voice (as description) is required for voicedesign")
            wavs, sr = tts.generate_voice_design(
                text=str(text).strip(),
                language=payload.get("language") or "Auto",
                instruct=payload.get("style") or payload.get("instruct") or payload.get("instructions"),
                non_streaming_mode=True,
                max_new_tokens=int(payload.get("max_new_tokens") or 2048),
            )
        else:
            raise HTTPException(
                400,
                "The 'base' (voice clone) model is not exposed via /v1/audio/speech in this minimal server. "
                "Use customvoice or voicedesign.",
            )

        audio = wavs[0]
        audio = np.asarray(audio, dtype=np.float32)

        if response_format == "wav":
            data = _wav_bytes(audio, sr)
            return Response(content=data, media_type="audio/wav")
        elif response_format == "pcm":
            # 16-bit PCM little-endian
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype("<i2")
            return Response(content=pcm16.tobytes(), media_type="application/octet-stream")
        elif response_format == "mp3":
            # Convert to MP3 using pydub
            wav_data = _wav_bytes(audio, sr)
            wav_audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            mp3_buf = io.BytesIO()
            wav_audio.export(mp3_buf, format="mp3", bitrate="128k")
            return Response(content=mp3_buf.getvalue(), media_type="audio/mpeg")
        else:
            raise HTTPException(400, "Unsupported response_format. Use 'wav', 'pcm', or 'mp3'.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {type(e).__name__}: {e}")
