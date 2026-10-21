"""Typed LangGraph workflow with interrupt/resume classification confirmation."""

from __future__ import annotations

import io
import time
from typing import NotRequired, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from PIL import Image

from .constants import (
    CITIES,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    LANGUAGES,
    MARGIN_THRESHOLD,
)
from .explanation import Explainer, FALLBACKS, explanation_is_safe
from .schemas import Guidance, Prediction, RuleCard, TraceEvent


OFFICIAL_FALLBACKS = {
    "zurich": (
        "City of Zurich — Correct waste disposal in Zurich",
        "https://www.stadt-zuerich.ch/content/dam/web/de/umwelt-energie/entsorgung/"
        "dokumente/correct-waste-disposal-zurich.pdf",
    ),
    "gurugram": (
        "Municipal Corporation Gurugram — official portal",
        "https://mcg.gov.in/",
    ),
}


class Classifier(Protocol):
    def predict(self, image: Image.Image) -> list[Prediction]: ...


class Retriever(Protocol):
    def retrieve(
        self, city: str, material_class: str, optional_note: str = "", k: int = 3
    ) -> list[RuleCard]: ...


class BinWiseState(TypedDict):
    image: bytes
    city: str
    language: str
    optional_note: str
    predictions: list[dict]
    confirmed_class: str | None
    needs_confirmation: bool
    retrieved_rule_ids: list[str]
    retrieved_rules: list[dict]
    guidance: dict | None
    trace: list[dict]
    error: str | None
    forced_class: NotRequired[str | None]


def _trace(node: str, started: float, **details: object) -> dict:
    return TraceEvent(
        node=node,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        details=details,
    ).model_dump(mode="json")


def _translation_notice(language: str) -> str | None:
    if language == "en":
        return None
    if language == "de":
        return "Maschinell unterstützte Übersetzung; maßgeblich ist die verlinkte offizielle Quelle."
    return "यह मशीन-सहायता से किया गया अनुवाद है; आधिकारिक जानकारी के लिए दिए गए स्रोत को देखें।"


class BinWiseWorkflow:
    def __init__(
        self,
        classifier: Classifier,
        retriever: Retriever,
        explainer: Explainer,
        checkpointer=None,
    ) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.explainer = explainer
        self.graph = self._build(checkpointer or InMemorySaver())

    def _build(self, checkpointer):
        builder = StateGraph(BinWiseState)
        builder.add_node("validate_input", self.validate_input)
        builder.add_node("classify_image", self.classify_image)
        builder.add_node("confirm_prediction", self.confirm_prediction)
        builder.add_node("retrieve_rules", self.retrieve_rules)
        builder.add_node("generate_explanation", self.generate_explanation)
        builder.add_node("validate_grounding", self.validate_grounding)
        builder.add_node("render_result", self.render_result)

        builder.add_edge(START, "validate_input")
        builder.add_edge("validate_input", "classify_image")
        builder.add_conditional_edges(
            "classify_image",
            self.confidence_router,
            {
                "confirm_prediction": "confirm_prediction",
                "retrieve_rules": "retrieve_rules",
            },
        )
        builder.add_edge("confirm_prediction", "retrieve_rules")
        builder.add_edge("retrieve_rules", "generate_explanation")
        builder.add_edge("generate_explanation", "validate_grounding")
        builder.add_edge("validate_grounding", "render_result")
        builder.add_edge("render_result", END)
        return builder.compile(checkpointer=checkpointer)

    def validate_input(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        if state["city"] not in CITIES:
            raise ValueError(f"Unsupported city: {state['city']}")
        if state["language"] not in LANGUAGES:
            raise ValueError(f"Unsupported language: {state['language']}")
        if not state["image"]:
            raise ValueError("Please upload a waste-item image.")
        try:
            with Image.open(io.BytesIO(state["image"])) as candidate:
                candidate.verify()
        except Exception as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc
        return {
            "trace": state["trace"]
            + [_trace("validate_input", started, city=state["city"])],
            "error": None,
        }

    def classify_image(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        forced = state.get("forced_class")
        if forced:
            if forced not in CLASS_NAMES:
                raise ValueError(f"Unsupported override: {forced}")
            predictions = [{"label": forced, "confidence": 1.0}]
            needs_confirmation = False
            confirmed_class = forced
            mode = "user_override"
        else:
            image = Image.open(io.BytesIO(state["image"])).convert("RGB")
            values = self.classifier.predict(image)
            if len(values) < 3:
                raise ValueError("The classifier must return three predictions.")
            predictions = [item.model_dump(mode="json") for item in values[:3]]
            top = predictions[0]["confidence"]
            margin = top - predictions[1]["confidence"]
            needs_confirmation = top < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD
            confirmed_class = None if needs_confirmation else predictions[0]["label"]
            mode = "model"
        return {
            "predictions": predictions,
            "needs_confirmation": needs_confirmation,
            "confirmed_class": confirmed_class,
            "trace": state["trace"]
            + [
                _trace(
                    "classify_image",
                    started,
                    mode=mode,
                    top_prediction=predictions[0]["label"],
                    confidence=predictions[0]["confidence"],
                    needs_confirmation=needs_confirmation,
                )
            ],
        }

    @staticmethod
    def confidence_router(state: BinWiseState) -> str:
        return "confirm_prediction" if state["needs_confirmation"] else "retrieve_rules"

    def confirm_prediction(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        selection = interrupt(
            {
                "kind": "classification_confirmation",
                "message": "The image is uncertain. Please choose the closest material.",
                "options": state["predictions"],
            }
        )
        if selection not in CLASS_NAMES:
            raise ValueError("The confirmation must be one of the supported materials.")
        return {
            "confirmed_class": selection,
            "needs_confirmation": False,
            "trace": state["trace"]
            + [_trace("confirm_prediction", started, selected=selection)],
        }

    def retrieve_rules(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        selected = state["confirmed_class"]
        if not selected:
            raise ValueError("A material class must be confirmed before retrieval.")
        rules = self.retriever.retrieve(
            city=state["city"],
            material_class=selected,
            optional_note=state["optional_note"],
            k=3,
        )
        if not rules:
            title, url = OFFICIAL_FALLBACKS[state["city"]]
            messages = {
                "en": "No reviewed item-specific rule is available. Open the official municipal guide before disposal.",
                "de": "Keine geprüfte objektspezifische Regel ist verfügbar. Vor der Entsorgung den offiziellen Leitfaden öffnen.",
                "hi": "इस वस्तु के लिए सत्यापित नियम उपलब्ध नहीं है। निपटान से पहले आधिकारिक नगर गाइड खोलें।",
            }
            fallback = RuleCard.model_validate(
                {
                    "rule_id": f"{state['city']}-official-source-fallback",
                    "city": state["city"],
                    "material_class": selected,
                    "translations": {
                        language: {
                            "action": message,
                            "steps": [message],
                            "exceptions": ["Do not guess when the official rule is unavailable."],
                        }
                        for language, message in messages.items()
                    },
                    "source_title": title,
                    "source_url": url,
                    "last_verified": "2026-10-20",
                }
            )
            rules = [fallback]
        if any(rule.city != state["city"] for rule in rules):
            raise ValueError("Retriever returned a rule for the wrong city.")
        return {
            "retrieved_rule_ids": [rule.rule_id for rule in rules],
            "retrieved_rules": [rule.model_dump(mode="json") for rule in rules],
            "trace": state["trace"]
            + [
                _trace(
                    "retrieve_rules",
                    started,
                    city=state["city"],
                    rule_ids=[rule.rule_id for rule in rules],
                )
            ],
        }

    def generate_explanation(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        selected = state["confirmed_class"]
        rules = [RuleCard.model_validate(item) for item in state["retrieved_rules"]]
        rule = next(
            (item for item in rules if item.material_class == selected), rules[0]
        )
        localized = rule.translations[state["language"]]
        base = Guidance(
            rule_id=rule.rule_id,
            city=rule.city,
            material_class=rule.material_class,
            language=state["language"],
            action=localized.action,
            steps=localized.steps,
            exceptions=localized.exceptions,
            explanation="Pending",
            source_title=rule.source_title,
            source_url=rule.source_url,
            last_verified=rule.last_verified,
            translation_notice=_translation_notice(state["language"]),
        )
        explanation, fallback_used = self.explainer.explain(base)
        if not explanation_is_safe(explanation, state["language"]):
            explanation = FALLBACKS[state["language"]]
            fallback_used = True
        guidance = Guidance.model_validate(
            {
                **base.model_dump(mode="json"),
                "explanation": explanation,
                "explanation_fallback_used": fallback_used,
            }
        )
        return {
            "guidance": guidance.model_dump(mode="json"),
            "trace": state["trace"]
            + [
                _trace(
                    "generate_explanation",
                    started,
                    fallback_used=fallback_used,
                )
            ],
        }

    def validate_grounding(self, state: BinWiseState) -> dict:
        started = time.perf_counter()
        guidance = Guidance.model_validate(state["guidance"])
        if guidance.rule_id not in state["retrieved_rule_ids"]:
            raise ValueError("Guidance is not grounded in a retrieved rule.")
        source_rule = next(
            RuleCard.model_validate(item)
            for item in state["retrieved_rules"]
            if item["rule_id"] == guidance.rule_id
        )
        localized = source_rule.translations[guidance.language]
        deterministic_fields_match = (
            guidance.action == localized.action
            and guidance.steps == localized.steps
            and guidance.exceptions == localized.exceptions
            and str(guidance.source_url).rstrip("/")
            == str(source_rule.source_url).rstrip("/")
        )
        if not deterministic_fields_match:
            raise ValueError("Generated guidance changed a sourced instruction.")
        return {
            "trace": state["trace"]
            + [
                _trace(
                    "validate_grounding", started, rule_id=guidance.rule_id
                )
            ]
        }

    @staticmethod
    def render_result(state: BinWiseState) -> dict:
        started = time.perf_counter()
        return {
            "trace": state["trace"]
            + [_trace("render_result", started, status="ready")]
        }

    def start(self, state: BinWiseState, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(state, config=config)

    def resume(self, selection: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(Command(resume=selection), config=config)

