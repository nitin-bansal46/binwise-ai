from pathlib import Path
import unittest

from binwise.constants import CITIES, CLASS_NAMES
from binwise.retrieval import RuleRetriever
from binwise.rules import load_rules


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = load_rules(ROOT / "data" / "rules.json")
        cls.retriever = RuleRetriever(rules=rules, use_embeddings=False)

    def test_city_filter_is_never_crossed(self):
        for city in CITIES:
            for material in CLASS_NAMES:
                results = self.retriever.retrieve(city, material, "sample note", k=3)
                self.assertTrue(results)
                self.assertTrue(all(rule.city == city for rule in results))

    def test_confirmed_material_is_always_first(self):
        results = self.retriever.retrieve("zurich", "Glass", "broken green bottle")
        self.assertEqual(results[0].rule_id, "zurich-glass")


if __name__ == "__main__":
    unittest.main()

