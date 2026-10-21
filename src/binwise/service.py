"""Application-facing workflow service."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

from PIL import Image

from .classifier import VisionClassifier
from .explanation import LocalQwenExplainer
from .retrieval import RuleRetriever
from .workflow import BinWiseState, BinWiseWorkflow


@dataclass
class WorkflowResponse:
    status: str
    thread_id: str
    state: dict
    confirmation: dict | None = None


def image_to_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


class BinWiseService:
    def __init__(self, workflow: BinWiseWorkflow | None = None) -> None:
        self.workflow = workflow or BinWiseWorkflow(
            classifier=VisionClassifier(),
            retriever=RuleRetriever(),
            explainer=LocalQwenExplainer(),
        )

    @staticmethod
    def _response(result: dict, thread_id: str) -> WorkflowResponse:
        interrupts = result.get("__interrupt__", ())
        if interrupts:
            payload = interrupts[0].value
            return WorkflowResponse(
                status="needs_confirmation",
                thread_id=thread_id,
                state=result,
                confirmation=payload,
            )
        return WorkflowResponse(status="complete", thread_id=thread_id, state=result)

    def analyze(
        self,
        image: Image.Image,
        city: str,
        language: str,
        optional_note: str = "",
        thread_id: str | None = None,
        forced_class: str | None = None,
    ) -> WorkflowResponse:
        current_thread = thread_id or str(uuid.uuid4())
        state: BinWiseState = {
            "image": image_to_bytes(image),
            "city": city,
            "language": language,
            "optional_note": optional_note.strip(),
            "predictions": [],
            "confirmed_class": None,
            "needs_confirmation": False,
            "retrieved_rule_ids": [],
            "retrieved_rules": [],
            "guidance": None,
            "trace": [],
            "error": None,
            "forced_class": forced_class,
        }
        result = self.workflow.start(state, current_thread)
        return self._response(result, current_thread)

    def resume(self, thread_id: str, selected_class: str) -> WorkflowResponse:
        result = self.workflow.resume(selected_class, thread_id)
        return self._response(result, thread_id)

