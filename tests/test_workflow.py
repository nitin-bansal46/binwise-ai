from pathlib import Path
import unittest

from PIL import Image

from binwise.explanation import FALLBACKS
from binwise.retrieval import RuleRetriever
from binwise.rules import load_rules
from binwise.schemas import Prediction
from binwise.service import BinWiseService
from binwise.workflow import BinWiseWorkflow


ROOT = Path(__file__).resolve().parents[1]


class FakeClassifier:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, _image):
        return [Prediction(label=label, confidence=confidence) for label, confidence in self.predictions]


class FakeExplainer:
    def explain(self, guidance):
        return FALLBACKS[guidance.language], True


class MaliciousExplainer:
    def explain(self, _guidance):
        return "Put this in a different bin in Gurugram.", False


class EmptyRetriever:
    def retrieve(self, **_kwargs):
        return []


def make_service(predictions, retriever=None):
    retriever = retriever or RuleRetriever(
        rules=load_rules(ROOT / "data" / "rules.json"), use_embeddings=False
    )
    workflow = BinWiseWorkflow(FakeClassifier(predictions), retriever, FakeExplainer())
    return BinWiseService(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32), color="green")

    def test_high_confidence_completes_without_interrupt(self):
        service = make_service([("Glass", 0.82), ("Plastic", 0.10), ("Metal", 0.08)])
        response = service.analyze(self.image, "zurich", "en")
        self.assertEqual(response.status, "complete")
        self.assertEqual(response.state["guidance"]["rule_id"], "zurich-glass")
        self.assertEqual(response.state["retrieved_rule_ids"][0], "zurich-glass")

    def test_low_confidence_interrupts_and_confirmation_replaces_prediction(self):
        service = make_service([("Plastic", 0.52), ("Glass", 0.34), ("Metal", 0.14)])
        pending = service.analyze(self.image, "gurugram", "hi")
        self.assertEqual(pending.status, "needs_confirmation")
        completed = service.resume(pending.thread_id, "Glass")
        self.assertEqual(completed.status, "complete")
        self.assertEqual(completed.state["confirmed_class"], "Glass")
        self.assertEqual(completed.state["guidance"]["rule_id"], "gurugram-glass")

    def test_small_top_two_margin_interrupts_even_above_threshold(self):
        service = make_service([("Paper", 0.76), ("Cardboard", 0.68), ("Plastic", 0.02)])
        response = service.analyze(self.image, "zurich", "de")
        self.assertEqual(response.status, "needs_confirmation")

    def test_user_can_override_high_confidence_prediction(self):
        service = make_service([("Paper", 0.90), ("Cardboard", 0.07), ("Plastic", 0.03)])
        response = service.analyze(
            self.image, "zurich", "en", forced_class="Textile Trash"
        )
        self.assertEqual(response.state["guidance"]["rule_id"], "zurich-textile-trash")

    def test_empty_retrieval_uses_official_source_fallback(self):
        service = make_service(
            [("Paper", 0.90), ("Cardboard", 0.07), ("Plastic", 0.03)],
            retriever=EmptyRetriever(),
        )
        response = service.analyze(self.image, "gurugram", "en")
        guidance = response.state["guidance"]
        self.assertEqual(guidance["rule_id"], "gurugram-official-source-fallback")
        self.assertEqual(guidance["source_url"], "https://mcg.gov.in/")

    def test_corrupt_image_returns_a_helpful_validation_error(self):
        service = make_service(
            [("Paper", 0.90), ("Cardboard", 0.07), ("Plastic", 0.03)]
        )
        state = {
            "image": b"not-an-image",
            "city": "zurich",
            "language": "en",
            "optional_note": "",
            "predictions": [],
            "confirmed_class": None,
            "needs_confirmation": False,
            "retrieved_rule_ids": [],
            "retrieved_rules": [],
            "guidance": None,
            "trace": [],
            "error": None,
        }
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            service.workflow.start(state, "corrupt-image")

    def test_trace_has_requested_nodes_and_no_image_data(self):
        service = make_service([("Metal", 0.90), ("Glass", 0.06), ("Plastic", 0.04)])
        response = service.analyze(self.image, "zurich", "en")
        names = [event["node"] for event in response.state["trace"]]
        self.assertEqual(
            names,
            [
                "validate_input",
                "classify_image",
                "retrieve_rules",
                "generate_explanation",
                "validate_grounding",
                "render_result",
            ],
        )
        self.assertTrue(
            all("image" not in event["details"] for event in response.state["trace"])
        )

    def test_explainer_cannot_introduce_a_bin_city_or_collection_method(self):
        retriever = RuleRetriever(
            rules=load_rules(ROOT / "data" / "rules.json"), use_embeddings=False
        )
        workflow = BinWiseWorkflow(
            FakeClassifier([("Glass", 0.90), ("Metal", 0.06), ("Plastic", 0.04)]),
            retriever,
            MaliciousExplainer(),
        )
        response = BinWiseService(workflow).analyze(self.image, "zurich", "en")
        guidance = response.state["guidance"]
        self.assertEqual(guidance["explanation"], FALLBACKS["en"])
        self.assertTrue(guidance["explanation_fallback_used"])


if __name__ == "__main__":
    unittest.main()

