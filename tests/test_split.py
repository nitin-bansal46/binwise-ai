import csv
import tempfile
from pathlib import Path
import unittest

from PIL import Image

from binwise.constants import CLASS_NAMES


class SplitTests(unittest.TestCase):
    def test_stratified_split_has_no_overlap(self):
        try:
            from binwise.training import create_split_manifest
        except ImportError:
            self.skipTest("Training dependencies are optional")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in CLASS_NAMES:
                folder = root / class_name
                folder.mkdir()
                for index in range(20):
                    Image.new("RGB", (8, 8)).save(folder / f"{index}.png")
            manifest = create_split_manifest(root, root / "split.csv")
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            paths = [row["path"] for row in rows]
            self.assertEqual(len(paths), len(set(paths)))
            counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
            self.assertEqual(counts, {"train": 126, "validation": 27, "test": 27})


if __name__ == "__main__":
    unittest.main()

