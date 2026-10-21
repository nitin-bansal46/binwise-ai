"""Loading and validating the structured municipal rule corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import CITIES, CLASS_NAMES
from .schemas import RuleCard


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "rules.json"


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> list[RuleCard]:
    source = Path(path)
    records = json.loads(source.read_text(encoding="utf-8"))
    rules = [RuleCard.model_validate(record) for record in records]
    validate_rule_corpus(rules)
    return rules


def validate_rule_corpus(rules: list[RuleCard]) -> None:
    expected = {(city, material) for city in CITIES for material in CLASS_NAMES}
    actual = {(rule.city, rule.material_class) for rule in rules}
    duplicates = len(actual) != len(rules)
    missing = expected - actual
    extra = actual - expected
    if duplicates or missing or extra:
        raise ValueError(
            "Invalid rule corpus: "
            f"duplicates={duplicates}, missing={sorted(missing)}, extra={sorted(extra)}"
        )


def rule_text(rule: RuleCard) -> str:
    """Build embedding text without losing the structured source of truth."""
    chunks = [rule.city, rule.material_class]
    for language in ("en", "de", "hi"):
        localized = rule.translations[language]
        chunks.extend(
            [localized.action, *localized.steps, *localized.exceptions]
        )
    return " | ".join(chunks)

