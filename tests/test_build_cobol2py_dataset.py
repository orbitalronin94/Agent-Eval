import unittest

from scripts.build_cobol2py_dataset import group_key, split_families


class DatasetSplitTests(unittest.TestCase):
    def test_semantically_identical_feature_groups_stay_together(self):
        cases = [
            {"id": "display-a", "features": ["DISPLAY literal"]},
            {"id": "bench-01", "features": ["DISPLAY literal"]},
            {"id": "numeric-a", "features": ["ADD numeric item"]},
        ]
        splits = split_families(cases)
        self.assertEqual(splits[group_key(cases[0])], splits[group_key(cases[1])])

    def test_split_is_balanced_and_deterministic(self):
        cases = [
            {"id": f"case-{index:03d}", "features": [f"independent semantic feature {index:03d}"]}
            for index in range(100)
        ]
        first = split_families(cases)
        second = split_families(cases)
        self.assertEqual(first, second)
        counts = {name: sum(first[group_key(case)] == name for case in cases)
                  for name in ("train", "dev", "holdout")}
        self.assertTrue(68 <= counts["train"] <= 72, counts)
        self.assertTrue(13 <= counts["dev"] <= 17, counts)
        self.assertTrue(13 <= counts["holdout"] <= 17, counts)


if __name__ == "__main__":
    unittest.main()
