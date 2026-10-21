from pathlib import Path
import unittest

from binwise.constants import CITIES, CLASS_NAMES, LANGUAGES
from binwise.rules import load_rules


ROOT = Path(__file__).resolve().parents[1]


class RuleCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules(ROOT / "data" / "rules.json")

    def test_has_one_rule_for_every_city_material_pair(self):
        self.assertEqual(len(self.rules), 18)
        actual = {(rule.city, rule.material_class) for rule in self.rules}
        expected = {(city, material) for city in CITIES for material in CLASS_NAMES}
        self.assertEqual(actual, expected)

    def test_every_rule_has_all_languages_and_official_source(self):
        for rule in self.rules:
            self.assertEqual(set(rule.translations), set(LANGUAGES))
            self.assertTrue(str(rule.source_url).startswith("https://"))
            for localized in rule.translations.values():
                self.assertTrue(localized.action)
                self.assertTrue(localized.steps)


if __name__ == "__main__":
    unittest.main()

