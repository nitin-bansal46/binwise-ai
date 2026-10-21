"""Pydantic contracts shared by the workflow, UI, tests, and data pipeline."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .constants import CLASS_NAMES


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_label(self) -> "Prediction":
        if self.label not in CLASS_NAMES:
            raise ValueError(f"Unsupported class: {self.label}")
        return self


class LocalizedRule(BaseModel):
    action: str = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    exceptions: list[str] = Field(default_factory=list)


class RuleCard(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    rule_id: str
    city: Literal["zurich", "gurugram"]
    material_class: str
    translations: dict[Literal["en", "de", "hi"], LocalizedRule]
    source_title: str
    source_url: HttpUrl
    last_verified: date

    @model_validator(mode="after")
    def validate_coverage(self) -> "RuleCard":
        if self.material_class not in CLASS_NAMES:
            raise ValueError(f"Unsupported material class: {self.material_class}")
        if set(self.translations) != {"en", "de", "hi"}:
            raise ValueError("Every rule must contain en, de, and hi translations")
        return self


class FriendlyExplanation(BaseModel):
    explanation: str = Field(min_length=8, max_length=240)


class Guidance(BaseModel):
    rule_id: str
    city: Literal["zurich", "gurugram"]
    material_class: str
    language: Literal["en", "de", "hi"]
    action: str
    steps: list[str]
    exceptions: list[str]
    explanation: str
    source_title: str
    source_url: HttpUrl
    last_verified: date
    translation_notice: str | None = None
    explanation_fallback_used: bool = False


class TraceEvent(BaseModel):
    node: str
    duration_ms: float = Field(ge=0)
    details: dict[str, object] = Field(default_factory=dict)

