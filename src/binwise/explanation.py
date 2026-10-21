"""Optional local Qwen explanation with a deterministic safe fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .schemas import FriendlyExplanation, Guidance
from .gpu import gpu_task


FALLBACKS = {
    "en": "Small sorting choices keep useful materials in circulation.",
    "de": "Richtiges Sortieren hält wertvolle Materialien im Kreislauf.",
    "hi": "सही छंटाई उपयोगी सामग्रियों को दोबारा इस्तेमाल में बनाए रखती है।",
}

BANNED_GUIDANCE_TERMS = {
    "en": (" bin", "collection", "dispose", "disposal", "zurich", "gurugram", "garbage"),
    "de": ("tonne", "sammlung", "entsorg", "zürich", "zurich", "gurugram"),
    "hi": ("डिब्ब", "संग्रह", "निपटान", "ज़्यूरिख", "ज्यूरिख", "गुरुग्राम"),
}


def explanation_is_safe(text: str, language: str) -> bool:
    """The generated sentence must not contain operational disposal guidance."""
    normalized = f" {text.casefold()}"
    return not any(term in normalized for term in BANNED_GUIDANCE_TERMS[language])


class Explainer(Protocol):
    def explain(self, guidance: Guidance) -> tuple[str, bool]: ...


class LocalQwenExplainer:
    """Generate one positive sentence; never generate disposal instructions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        enabled: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.enabled = (
            os.getenv("BINWISE_ENABLE_LLM", "1") == "1" if enabled is None else enabled
        )
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            dtype="auto",
        )

    @gpu_task(duration=20)
    def _generate(self, prompt: str) -> str:
        return self._pipeline(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]

    @staticmethod
    def _extract_json(text: str) -> FriendlyExplanation:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object in model output")
        return FriendlyExplanation.model_validate(json.loads(match.group(0)))

    def explain(self, guidance: Guidance) -> tuple[str, bool]:
        fallback = FALLBACKS[guidance.language]
        if not self.enabled:
            return fallback, True
        prompt = (
            "You are a concise environmental educator. Return JSON only as "
            '{"explanation":"one friendly positive sentence"}. '
            f"Write in language code {guidance.language}. The verified action is: "
            f"{guidance.action} Do not mention bins, collection methods, cities, "
            "or add instructions. Explain only the general environmental benefit."
        )
        try:
            self._load()
            for _ in range(2):
                output = self._generate(prompt)
                try:
                    parsed = self._extract_json(output)
                    if explanation_is_safe(parsed.explanation, guidance.language):
                        return parsed.explanation, False
                except (ValueError, json.JSONDecodeError):
                    continue
        except (ImportError, OSError, RuntimeError, ValueError):
            pass
        return fallback, True

