import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("community_records", ROOT / "scripts/community.py")
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)


class ContributionTests(unittest.TestCase):
    def setUp(self):
        self.record = records.read(ROOT / "community/templates/contribution.json")
        self.record.update(
            id="C001",
            title="Test",
            status="accepted",
            evidence=["https://example.org/source"],
            required_checks=["roundtrip"],
            required_reviewers=["partner"],
            approvals=[{"reviewer": "partner", "evidence_url": "https://example.org/approval"}],
        )
        self.record["baseline"] = dict(
            dataset="test",
            release="1",
            import_repository="https://example.org/repo",
            import_revision="abc",
            input_hashes={"source": "hash"},
        )
        self.record["layer"]["implementation_revision"] = "def"
        self.record["layer"].update(
            kind="synthetic", implementation_url="https://example.org/layer"
        )
        self.record["approvals"][0].update(
            baseline=copy.deepcopy(self.record["baseline"]), layer_revision="def"
        )
        self.record["runs"] = [
            dict(
                baseline=copy.deepcopy(self.record["baseline"]),
                layer_revision="def",
                evidence_url="https://example.org/run",
                checks=[
                    dict(
                        id="roundtrip",
                        status="passed",
                        evidence_url="https://example.org/check",
                    )
                ],
            )
        ]

    def test_accepted_requires_passing_checks_and_approvals(self):
        self.assertFalse(records.validate(self.record))
        self.record["runs"][0]["checks"][0]["status"] = "skipped"
        self.assertTrue(records.validate(self.record))

    def test_old_run_cannot_validate_new_layer(self):
        self.record["layer"]["implementation_revision"] = "changed"
        self.assertTrue(records.validate(self.record))

    def test_missing_partner_review_is_not_consensus(self):
        self.record["approvals"] = []
        self.assertTrue(records.validate(self.record))

    def test_unresolved_objection_blocks_acceptance(self):
        self.record["objections"] = [{"reason": "boundary"}]
        self.assertTrue(records.validate(self.record))

    def test_approval_for_old_layer_is_not_consensus(self):
        self.record["approvals"][0]["layer_revision"] = "old-layer"
        self.assertTrue(records.validate(self.record))

    def test_empty_implementation_cannot_be_accepted(self):
        self.record["layer"]["implementation_revision"] = ""
        self.record["runs"][0]["layer_revision"] = ""
        self.record["approvals"][0]["layer_revision"] = ""
        self.assertTrue(records.validate(self.record))

    def test_every_status_needs_a_nonempty_string_id(self):
        # The id is the card title and the record link on the site, whatever the status.
        for status in sorted(records.STATES):
            for bad in ("", "   ", None, 7):
                record = copy.deepcopy(self.record)
                record.update(status=status, id=bad)
                self.assertIn(
                    "id must be a non-empty string", records.validate(record), (status, bad)
                )

    def test_id_must_be_usable_in_a_file_name_and_a_link(self):
        for bad in ("C#2", "C 3", "C%2F4?x", "a/b", "../C001", ".hidden", "C001 "):
            record = dict(copy.deepcopy(self.record), id=bad)
            self.assertTrue(any("id may contain only" in e for e in records.validate(record)), bad)
        for good in ("C001", "C006-rice-methane", "c_1.2"):
            record = dict(copy.deepcopy(self.record), id=good)
            self.assertFalse(records.validate(record), good)

    def test_title_must_be_a_nonempty_string(self):
        for bad in ("", "   ", None):
            record = dict(copy.deepcopy(self.record), status="proposed", title=bad)
            self.assertIn("title must be a non-empty string", records.validate(record), bad)

    def test_check_rejects_an_empty_id_and_an_id_that_is_not_the_file_name(self):
        proposed = records.read(ROOT / "community/contributions/C001.json")
        cases = [
            ({"C001.json": "C001"}, 0),
            ({"C001.json": ""}, 1),
            ({"C001.json": "C002"}, 1),
        ]
        for files, expected in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "community/contributions").mkdir(parents=True)
                (root / "community/recommendations").mkdir(parents=True)
                (root / "community/recommendations/tracker.json").write_text("[]")
                for name, identifier in files.items():
                    record = dict(copy.deepcopy(proposed), id=identifier)
                    (root / "community/contributions" / name).write_text(json.dumps(record))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(records.check(root), expected, files)

    def test_comparison_checks_scope_metrics_units_and_finiteness(self):
        baseline = records.read(ROOT / "community/templates/run-summary.json")
        with tempfile.TemporaryDirectory() as directory:
            before, after = (
                Path(directory) / "baseline.json",
                Path(directory) / "candidate.json",
            )
            before.write_text(json.dumps(baseline))
            candidate = copy.deepcopy(baseline)
            candidate["metrics"]["processes"]["value"] = 2
            after.write_text(json.dumps(candidate))
            self.assertEqual(records.compare(before, after)["metrics"]["processes"]["delta"], 1)
            for change in [
                lambda x: x["scope"].update(method="different"),
                lambda x: x["metrics"].pop("processes"),
                lambda x: x["metrics"]["processes"].update(unit="kg"),
                lambda x: x["metrics"]["processes"].update(value=float("nan")),
            ]:
                candidate = copy.deepcopy(baseline)
                change(candidate)
                after.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    records.compare(before, after)
