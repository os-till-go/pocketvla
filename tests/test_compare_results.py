"""Protect pairing and exact-test semantics used in the published report."""
import unittest

from tools.compare_results import compare, mcnemar_exact, paired_outcomes, wilson


def run(outcomes, ids=None, split="ID", digest="fixed"):
    return {"split": split, "dataset_sha256_at_export": digest,
            "episodes": len(outcomes), "results": [
                {"episode": i, "success": v}
                for i, v in zip(ids if ids is not None else range(len(outcomes)), outcomes)]}


class ComparisonTests(unittest.TestCase):
    def test_exact_binomial(self):
        self.assertEqual(mcnemar_exact(0, 0), 1)
        self.assertEqual(mcnemar_exact(6, 0), 0.03125)
        self.assertEqual(mcnemar_exact(0, 6), 0.03125)

    def test_pair_by_id_not_row_order(self):
        a = run([False, True])
        b = run([False, True], ids=[1, 0])
        result = compare(a, b, 100)
        self.assertEqual((result['gained'], result['lost'], result['delta_pp']), (1, 1, 0))
        self.assertEqual(compare(a, a, 100)['bootstrap_95_pp'], [0, 0])

    def test_reject_invalid_pairing(self):
        a = run([True, False])
        for b in (run([True, False], ids=[0, 0]), run([True, False], ids=[0, 2]),
                  run([True, False], split="OOD"), run([True, False], digest="other")):
            with self.assertRaises(ValueError):
                paired_outcomes(a, b)

    def test_wilson_boundary(self):
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0)
        self.assertTrue(0 < hi < 1)
        with self.assertRaises(ValueError):
            wilson(0, 0)


if __name__ == "__main__":
    unittest.main()
