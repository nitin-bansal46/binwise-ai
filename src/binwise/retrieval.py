"""City-filtered retrieval over the structured rule corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import CITIES, CLASS_NAMES
from .rules import load_rules, rule_text
from .schemas import RuleCard


class RuleRetriever:
    """Retrieve rules with a mandatory city filter before similarity ranking.

    Embeddings and FAISS are lazy and optional at runtime. Exact structured
    matching remains available if those dependencies cannot load.
    """

    def __init__(
        self,
        rules: Sequence[RuleCard] | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ) -> None:
        self.rules = list(rules or load_rules())
        self.model_name = model_name
        self.use_embeddings = use_embeddings
        self._model = None
        self._indexes: dict[str, object] = {}
        self._city_rules: dict[str, list[RuleCard]] = {
            city: [rule for rule in self.rules if rule.city == city] for city in CITIES
        }

    def _ensure_indexes(self) -> bool:
        if not self.use_embeddings:
            return False
        if self._indexes:
            return True
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            # The tiny corpus does not need GPU capacity. In particular,
            # retrieval must not allocate CUDA outside a ZeroGPU worker.
            self._model = SentenceTransformer(self.model_name, device="cpu")
            for city, candidates in self._city_rules.items():
                vectors = self._model.encode(
                    [rule_text(rule) for rule in candidates],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype("float32")
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                self._indexes[city] = index
            return True
        except (ImportError, OSError, RuntimeError):
            self.use_embeddings = False
            return False

    def retrieve(
        self,
        city: str,
        material_class: str,
        optional_note: str = "",
        k: int = 3,
    ) -> list[RuleCard]:
        if city not in CITIES:
            raise ValueError(f"Unsupported city: {city}")
        if material_class not in CLASS_NAMES:
            raise ValueError(f"Unsupported material: {material_class}")

        # Security/grounding boundary: candidates are filtered by city first.
        candidates = self._city_rules[city]
        exact = next(rule for rule in candidates if rule.material_class == material_class)

        ranked: list[RuleCard]
        if self._ensure_indexes():
            query = f"{material_class}. {optional_note}".strip()
            vector = self._model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")
            limit = min(max(k, 1), len(candidates))
            _, indices = self._indexes[city].search(vector, limit)
            ranked = [candidates[index] for index in indices[0] if index >= 0]
        else:
            ranked = [exact] + [
                rule for rule in candidates if rule.rule_id != exact.rule_id
            ]

        # Classification is confirmed, so its exact RuleCard must remain first.
        ranked = [exact] + [rule for rule in ranked if rule.rule_id != exact.rule_id]
        return ranked[: max(k, 1)]

    def save_indexes(self, output_dir: str | Path) -> None:
        if not self._ensure_indexes():
            raise RuntimeError("FAISS and sentence-transformers are required to save indexes")
        import faiss

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for city, index in self._indexes.items():
            faiss.write_index(index, str(destination / f"{city}.faiss"))
            metadata = [rule.rule_id for rule in self._city_rules[city]]
            (destination / f"{city}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

