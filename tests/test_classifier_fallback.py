from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from binwise.classifier import VisionClassifier
from binwise.schemas import Prediction


class ClassifierFallbackTests(unittest.TestCase):
    def setUp(self):
        self.classifier = VisionClassifier()
        self.classifier._load = Mock()
        self.classifier._device = SimpleNamespace(type="cuda")
        self.predictions = [
            Prediction(label="Glass", confidence=0.80),
            Prediction(label="Metal", confidence=0.12),
            Prediction(label="Plastic", confidence=0.08),
        ]
        self.classifier._forward_cpu = Mock(return_value=self.predictions)
        self.image = object()

    def test_missing_cuda_uses_real_cpu_inference(self):
        with patch(
            "binwise.classifier._classify_prepared",
            side_effect=RuntimeError("No CUDA GPUs are available"),
        ):
            self.assertEqual(self.classifier.predict(self.image), self.predictions)
        self.classifier._forward_cpu.assert_called_once_with(self.image)
        self.assertEqual(self.classifier.last_inference_backend, "cpu_fallback")
        self.assertEqual(self.classifier.last_fallback_reason, "RuntimeError")

    def test_wrapped_worker_failure_also_uses_cpu(self):
        with patch(
            "binwise.classifier._classify_prepared", side_effect=KeyError("RuntimeError")
        ):
            self.assertEqual(self.classifier.predict(self.image), self.predictions)
        self.classifier._forward_cpu.assert_called_once_with(self.image)

    def test_gpu_success_does_not_run_cpu(self):
        with patch("binwise.classifier._classify_prepared", return_value=self.predictions):
            self.assertEqual(self.classifier.predict(self.image), self.predictions)
        self.classifier._forward_cpu.assert_not_called()
        self.assertEqual(self.classifier.last_inference_backend, "gpu")

    def test_cpu_machine_never_requests_gpu(self):
        self.classifier._device = SimpleNamespace(type="cpu")
        with patch("binwise.classifier._classify_prepared") as gpu:
            self.assertEqual(self.classifier.predict(self.image), self.predictions)
        gpu.assert_not_called()
        self.assertEqual(self.classifier.last_inference_backend, "cpu")

    def test_cpu_model_error_is_not_hidden(self):
        self.classifier._forward_cpu.side_effect = ValueError("Invalid model input")
        with patch(
            "binwise.classifier._classify_prepared",
            side_effect=RuntimeError("No CUDA GPUs are available"),
        ):
            with self.assertRaisesRegex(ValueError, "Invalid model input"):
                self.classifier.predict(self.image)


if __name__ == "__main__":
    unittest.main()
