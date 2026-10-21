import json
import unittest

from binwise.explanation import FALLBACKS, LocalQwenExplainer
from binwise.schemas import Guidance


class FakePipeline:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = 0

    def __call__(self, messages, **_kwargs):
        self.calls += 1
        assert messages[-1]["role"] == "user"
        return [{"generated_text": next(self.texts)}]


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.guidance = Guidance(
            rule_id="zurich-glass",
            city="zurich",
            material_class="Glass",
            language="en",
            action="Use the official collection point.",
            steps=["Empty the container."],
            exceptions=[],
            explanation="Pending",
            source_title="City of Zurich",
            source_url="https://www.stadt-zuerich.ch/",
            last_verified="2026-10-20",
        )

    def test_valid_json_is_used(self):
        explanation = "Sorting saves resources for future use."
        explainer = LocalQwenExplainer(enabled=True)
        explainer._pipeline = FakePipeline([json.dumps({"explanation": explanation})])
        self.assertEqual(explainer.explain(self.guidance), (explanation, False))

    def test_malformed_output_retries_once_then_falls_back(self):
        explainer = LocalQwenExplainer(enabled=True)
        explainer._pipeline = FakePipeline(["not JSON", "still not JSON"])
        self.assertEqual(explainer.explain(self.guidance), (FALLBACKS["en"], True))
        self.assertEqual(explainer._pipeline.calls, 2)

    def test_unsafe_output_is_rejected_before_retry(self):
        explanation = "Sorting saves resources for future use."
        explainer = LocalQwenExplainer(enabled=True)
        explainer._pipeline = FakePipeline(
            [
                json.dumps({"explanation": "Use a different bin in Gurugram."}),
                json.dumps({"explanation": explanation}),
            ]
        )
        self.assertEqual(explainer.explain(self.guidance), (explanation, False))
        self.assertEqual(explainer._pipeline.calls, 2)

    def test_english_instruction_cannot_bypass_hindi_guard(self):
        guidance = self.guidance.model_copy(update={"language": "hi"})
        unsafe = json.dumps({"explanation": "The correct response should be 'Do not collect these items.'"})
        explainer = LocalQwenExplainer(enabled=True)
        explainer._pipeline = FakePipeline([unsafe, unsafe])
        self.assertEqual(explainer.explain(guidance), (FALLBACKS["hi"], True))

    def test_hindi_explanation_requires_devanagari(self):
        guidance = self.guidance.model_copy(update={"language": "hi"})
        wrong_language = json.dumps({"explanation": "Sorting saves resources for future use."})
        explainer = LocalQwenExplainer(enabled=True)
        explainer._pipeline = FakePipeline([wrong_language, wrong_language])
        self.assertEqual(explainer.explain(guidance), (FALLBACKS["hi"], True))


if __name__ == "__main__":
    unittest.main()
