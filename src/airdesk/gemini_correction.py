"""Optional Gemini review for uncertain local handwriting recognition."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os

import cv2
import numpy as np

from .segmented_recognition import MERGED_CLASSES


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.8-flash"
MAX_CORRECTED_CHARACTERS = 200


def gemini_error_message(error: Exception) -> str:
    """Keep actionable errors visible without printing request credentials."""
    code = getattr(error, "code", None)
    if code in (401, 403):
        return "API key rejected or access denied; check GEMINI_API_KEY"
    if code == 429:
        return "quota/rate limit reached; wait and retry"
    if code == 404:
        return "model unavailable; check GEMINI_MODEL"
    if code in (408, 504):
        return "request timed out; thumbs-up to retry"
    if isinstance(code, int) and 500 <= code < 600:
        return f"temporary service error ({code}); thumbs-up to retry"
    return f"{type(error).__name__}; check connection and retry with thumbs-up"


def _can_try_fallback(error: Exception) -> bool:
    import httpx

    code = getattr(error, "code", None)
    return (
        code in (404, 408, 429)
        or (isinstance(code, int) and 500 <= code < 600)
        or isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))
    )


@dataclass(frozen=True)
class GeminiCorrectionResult:
    text: str
    confidence: float
    explanation: str


def prepare_gemini_ink_image(canvas: np.ndarray) -> np.ndarray | None:
    """Crop the white canvas to ink so Gemini sees no webcam or desktop pixels."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim == 3 else canvas
    rows, columns = np.where(gray < 220)
    if len(rows) < 40:
        return None
    padding = 18
    top = max(int(rows.min()) - padding, 0)
    bottom = min(int(rows.max()) + padding + 1, gray.shape[0])
    left = max(int(columns.min()) - padding, 0)
    right = min(int(columns.max()) + padding + 1, gray.shape[1])
    return gray[top:bottom, left:right].copy()


def format_local_candidates(predictions) -> str:
    """Create a compact, position-preserving candidate list for the prompt."""
    rows = []
    text_position = 0
    for prediction in predictions or ():
        if prediction.space_before:
            text_position += 1
        alternatives = ", ".join(
            f"{candidate.character!r}:{candidate.confidence:.3f}"
            for candidate in prediction.alternatives
        )
        rows.append(f"position {text_position}: {alternatives}")
        text_position += 1
    return "\n".join(rows) if rows else "No per-character alternatives available."


def build_review_prompt(local_text: str, predictions) -> str:
    candidates = format_local_candidates(predictions)
    return (
        "Act as a conservative handwriting transcription reviewer. "
        "Transcribe only the visible ink in the attached black-on-white image. "
        "The local CNN classified each character independently, so use its "
        "alternatives as evidence, but correct mistakes when the image supports it.\n\n"
        f"Local transcription: {local_text!r}\n"
        f"Local candidates:\n{candidates}\n\n"
        "Rules:\n"
        "- Preserve exact uppercase/lowercase, digits, spaces, and punctuation.\n"
        "- Do not autocomplete, add words, improve grammar, or invent missing ink.\n"
        "- Unusual names, identifiers, code, and mixed strings may be intentional.\n"
        "- Return the exact transcription, a confidence from 0 to 1, and a short explanation."
    )


class GeminiCorrectionClient:
    """Small testable adapter around Gemini's multimodal structured output API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_model: str = DEFAULT_GEMINI_FALLBACK_MODEL,
        client=None,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is empty")
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.last_model_used = None
        self._preferred_model = model
        self._client = client

    @classmethod
    def from_environment(cls):
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        fallback_model = os.environ.get(
            "GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL
        ).strip()
        return cls(
            api_key=api_key,
            model=model or DEFAULT_GEMINI_MODEL,
            fallback_model=fallback_model or DEFAULT_GEMINI_FALLBACK_MODEL,
        )

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError(
                    "Gemini support needs google-genai; install requirements.txt"
                ) from error
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": 10000, "retry_options": {"attempts": 1}},
            )
        return self._client

    def verify_ready(self) -> None:
        self._get_client()

    def review(self, canvas, local_text: str, predictions=()) -> GeminiCorrectionResult:
        image = prepare_gemini_ink_image(canvas)
        if image is None:
            raise RuntimeError("the handwriting canvas contains too little ink")
        encoded, png = cv2.imencode(".png", image)
        if not encoded:
            raise RuntimeError("could not encode the handwriting canvas")
        try:
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Gemini support needs google-genai; install requirements.txt"
            ) from error
        response_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": ["text", "confidence", "explanation"],
        }
        contents = [
            build_review_prompt(local_text, predictions),
            types.Part.from_bytes(data=png.tobytes(), mime_type="image/png"),
        ]
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=response_schema,
        )
        primary = self._preferred_model
        fallback = self.fallback_model if primary == self.model else self.model
        self.last_model_used = None
        try:
            response = self._generate(primary, contents, config)
        except Exception as error:
            if (
                not _can_try_fallback(error)
                or fallback == primary
            ):
                raise
            print(f"Gemini {primary}: {gemini_error_message(error)}. Trying {fallback}.", flush=True)
            response = self._generate(fallback, contents, config)
        result = self._parse_response(response.text)
        # A working fallback should not pay the primary's timeout every line.
        self._preferred_model = self.last_model_used
        return result

    def _generate(self, model, contents, config):
        response = self._get_client().models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self.last_model_used = model
        return response

    @staticmethod
    def _parse_response(output_text: str) -> GeminiCorrectionResult:
        try:
            payload = json.loads(output_text)
            text = payload["text"]
            confidence = payload["confidence"]
            explanation = payload["explanation"]
            if (
                not isinstance(text, str)
                or not isinstance(explanation, str)
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("invalid field types or confidence")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("Gemini returned an invalid correction response") from error
        allowed = set(MERGED_CLASSES + " ")
        if not text.strip() or len(text) > MAX_CORRECTED_CHARACTERS:
            raise RuntimeError("Gemini returned an empty or excessively long transcription")
        if any(character not in allowed for character in text):
            raise RuntimeError("Gemini returned characters unsupported by AirDesk")
        return GeminiCorrectionResult(
            text=text,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            explanation=explanation[:160],
        )
