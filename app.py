"""Hugging Face Spaces / local Gradio entry point for BinWise AI."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

try:
    # Import before any model loads so ZeroGPU can enable CUDA emulation.
    import spaces
except ImportError:
    spaces = None

from binwise.constants import CLASS_NAMES
from binwise.schemas import Guidance
from binwise.service import BinWiseService, WorkflowResponse


_SERVICE: BinWiseService | None = None


def service() -> BinWiseService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = BinWiseService()
    return _SERVICE


def format_result(response: WorkflowResponse) -> str:
    guidance = Guidance.model_validate(response.state["guidance"])
    predictions = response.state.get("predictions", [])
    confidence = predictions[0]["confidence"] if predictions else 1.0
    prediction_text = (
        f"**Material:** {guidance.material_class}  ·  **Model confidence:** {confidence:.1%}"
    )
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(guidance.steps, 1))
    exceptions = "\n".join(f"- {item}" for item in guidance.exceptions) or "- None listed"
    notice = f"\n\n_{guidance.translation_notice}_" if guidance.translation_notice else ""
    fallback = (
        "\n\n_The local language model was unavailable, so a reviewed fallback sentence was used._"
        if guidance.explanation_fallback_used
        else ""
    )
    return f"""
{prediction_text}

### What to do

{guidance.action}

### Prepare it

{steps}

### Important exceptions

{exceptions}

### Why it helps

{guidance.explanation}

**Official source:** [{guidance.source_title}]({guidance.source_url})  
**Rule ID:** `{guidance.rule_id}` · **Last verified:** {guidance.last_verified}{notice}{fallback}
""".strip()


def error_outputs(message: str, thread_id: str):
    return (
        f"### Could not complete the check\n\n{message}",
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(value=None),
        [],
        thread_id,
    )


def response_outputs(response: WorkflowResponse):
    if response.status == "needs_confirmation":
        options = response.confirmation["options"]
        labels = [item["label"] for item in options]
        choices = [
            (f"{item['label']} ({item['confidence']:.1%})", item["label"])
            for item in options
        ]
        return (
            "### Please confirm the material\n\nThe image classifier is uncertain, so no disposal advice has been generated yet.",
            gr.update(visible=True),
            gr.update(choices=choices, value=labels[0]),
            gr.update(visible=False),
            gr.update(value=None),
            response.state.get("trace", []),
            response.thread_id,
        )
    return (
        format_result(response),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(visible=True),
        gr.update(choices=CLASS_NAMES, value=response.state["confirmed_class"]),
        response.state.get("trace", []),
        response.thread_id,
    )


def analyze(image, city, language, note, thread_id):
    if image is None:
        return error_outputs("Please upload or photograph one waste item.", thread_id)
    try:
        response = service().analyze(
            image=image,
            city=city,
            language=language,
            optional_note=note or "",
            thread_id=thread_id or str(uuid.uuid4()),
        )
        return response_outputs(response)
    except Exception as exc:
        return error_outputs(str(exc), thread_id)


def confirm(selected, thread_id):
    if not selected or not thread_id:
        return error_outputs("The pending classification session was lost. Please analyse again.", thread_id)
    try:
        return response_outputs(service().resume(thread_id, selected))
    except Exception as exc:
        return error_outputs(str(exc), thread_id)


def override(image, city, language, note, selected):
    if image is None or not selected:
        return error_outputs("Keep the image and choose a material before applying an override.", "")
    try:
        response = service().analyze(
            image=image,
            city=city,
            language=language,
            optional_note=note or "",
            thread_id=str(uuid.uuid4()),
            forced_class=selected,
        )
        return response_outputs(response)
    except Exception as exc:
        return error_outputs(str(exc), "")


def build_demo() -> gr.Blocks:
    configured = bool(os.getenv("BINWISE_MODEL_REPO") or os.getenv("BINWISE_WEIGHTS_PATH"))
    banner = (
        "The trained vision model is configured."
        if configured
        else "Demo setup required: set `BINWISE_MODEL_REPO` or `BINWISE_WEIGHTS_PATH` after training."
    )
    with gr.Blocks(title="BinWise AI — Where Does This Go?") as demo:
        gr.Markdown(
            "# ♻️ BinWise AI — Where Does This Go?\n"
            "Upload one item for sourced disposal guidance in Zurich or Gurugram. "
            "Rules can change, so always follow the linked official source.\n\n"
            f"> {banner}"
        )
        thread_id = gr.State("")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="Waste-item photo",
                    height=330,
                )
                city = gr.Dropdown(
                    choices=[("Zurich", "zurich"), ("Gurugram", "gurugram")],
                    value="zurich",
                    label="City",
                )
                language = gr.Dropdown(
                    choices=[("English", "en"), ("Deutsch", "de"), ("हिन्दी", "hi")],
                    value="en",
                    label="Language",
                )
                note = gr.Textbox(
                    label="Optional note",
                    placeholder="e.g. greasy pizza box, broken bottle",
                    max_lines=2,
                )
                analyze_button = gr.Button("Analyse item", variant="primary")
            with gr.Column(scale=1):
                result = gr.Markdown("Upload an image to begin.")
                with gr.Group(visible=False) as confirmation_group:
                    confirmation_choice = gr.Dropdown(label="Closest material")
                    confirmation_button = gr.Button("Use this material", variant="primary")
                with gr.Group(visible=False) as override_group:
                    override_choice = gr.Dropdown(CLASS_NAMES, label="Correct the material")
                    override_button = gr.Button("Apply correction")
                with gr.Accordion("Workflow trace", open=False):
                    trace = gr.JSON(label="Executed nodes (images are never logged)")

        outputs = [
            result,
            confirmation_group,
            confirmation_choice,
            override_group,
            override_choice,
            trace,
            thread_id,
        ]
        analyze_button.click(
            analyze,
            inputs=[image, city, language, note, thread_id],
            outputs=outputs,
        )
        confirmation_button.click(
            confirm,
            inputs=[confirmation_choice, thread_id],
            outputs=outputs,
        )
        override_button.click(
            override,
            inputs=[image, city, language, note, override_choice],
            outputs=outputs,
        )
        gr.Markdown(
            "**Privacy:** uploaded images are processed in memory and are not stored, "
            "tracked, or used for analytics. Translations are machine-assisted."
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()

