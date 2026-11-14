import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_checks", Path(__file__).resolve().parents[2] / "scripts/build_checks.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    counts = {"processes": 2, "production": 2, "technosphere": 1, "biosphere": 3}
    return {
        "dataset": "synthetic",
        "release": "test",
        "revision": "test-revision",
        "input_sha256": {"input": "a" * 64},
        "scope": {
            k: "synthetic"
            for k in (
                "background",
                "method_source",
                "functional_unit",
                "allocation",
                "boundary",
                "geography",
                "representation",
            )
        },
        "counts": {s: counts.copy() for s in module.STAGES},
        "changes": [],
        "methods": [
            {
                "id": "climate",
                "emitted": ["methane", "co2-biogenic"],
                "required": ["methane", "co2-biogenic"],
                "factors": {"methane": 27.0, "co2-biogenic": 0.0},
                "excluded": {},
                "policy_evidence_url": "https://example.org/synthetic-policy",
            }
        ],
        "probes": {"synthetic-product": 1.123456789012345},
        "packages": [],
        "artifacts": {"output": "b" * 64},
    }


class BuildChecks(unittest.TestCase):
    def test_explicit_zero_is_covered(self):
        self.assertEqual(module.validate(example()), [])

    def test_silent_exchange_loss_fails(self):
        data = example()
        data["counts"]["served"]["biosphere"] -= 1
        self.assertTrue(any("unexplained" in e for e in module.validate(data)))
        data["changes"] = [
            {
                "id": "drop-1",
                "transition": "imported->served",
                "kind": "biosphere",
                "delta": -1,
                "reason": "synthetic exclusion",
                "evidence_url": "https://example.org/test",
            }
        ]
        self.assertEqual(module.validate(data), [])

    def test_missing_methane_is_not_zero(self):
        data = example()
        del data["methods"][0]["factors"]["methane"]
        self.assertTrue(any("lacks characterization" in e for e in module.validate(data)))

    def test_exclusions_need_evidence(self):
        data = example()
        del data["methods"][0]["factors"]["methane"]
        data["methods"][0]["excluded"]["methane"] = {"reason": "test"}
        self.assertTrue(any("exclusion needs" in e for e in module.validate(data)))

    def test_summary_stability_and_artifact_distinction(self):
        a = example()
        b = copy.deepcopy(a)
        b["timestamp"] = "later"
        b["project"] = "another-location"
        b["revision"] = "new-importer"
        b["artifacts"]["output"] = "c" * 64
        b["probes"]["synthetic-product"] += 1e-15
        self.assertEqual(module.summary_identity(a), module.summary_identity(b))
        b["probes"]["synthetic-product"] += 0.01
        self.assertNotEqual(module.summary_identity(a), module.summary_identity(b))

    def test_nonfinite_and_boolean_counts_fail(self):
        data = example()
        data["probes"]["synthetic-product"] = float("nan")
        self.assertTrue(module.validate(data))
        data = example()
        data["counts"]["source"]["processes"] = True
        self.assertTrue(module.validate(data))

    def test_reordered_flow_lists_have_same_identity(self):
        a = example()
        b = copy.deepcopy(a)
        b["methods"][0]["emitted"].reverse()
        self.assertEqual(module.summary_identity(a), module.summary_identity(b))


if __name__ == "__main__":
    unittest.main()
