"""Track contributed layers and compare isolated builds; no importer dependencies."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

STATES = {"proposed", "testing", "reviewing", "accepted", "rejected", "superseded"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(record):
    errors = []
    required = {
        "id",
        "title",
        "status",
        "baseline",
        "layer",
        "evidence",
        "required_checks",
        "runs",
        "required_reviewers",
        "approvals",
        "objections",
        "upstream_project",
    }
    if required - record.keys():
        return ["missing fields: " + ", ".join(sorted(required - record.keys()))]
    # The id titles the card and names the record file the site links to, in every state.
    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append("id must be a non-empty string")
    elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", record["id"]):
        errors.append("id may contain only letters, digits, '.', '-' and '_'")
    if not isinstance(record["title"], str) or not record["title"].strip():
        errors.append("title must be a non-empty string")
    if record["status"] not in STATES:
        errors.append("invalid status")
    for key in (
        "evidence",
        "required_checks",
        "runs",
        "required_reviewers",
        "approvals",
        "objections",
    ):
        if not isinstance(record[key], list):
            errors.append(f"{key} must be a list")
    for key, fields in [
        (
            "baseline",
            {
                "dataset",
                "release",
                "import_repository",
                "import_revision",
                "input_hashes",
            },
        ),
        (
            "layer",
            {"kind", "implementation_url", "implementation_revision", "payloads"},
        ),
    ]:
        if not isinstance(record[key], dict) or fields - record[key].keys():
            errors.append(f"{key} needs {sorted(fields)}")
    if errors:
        return errors
    if record["status"] == "accepted":
        baseline = record["baseline"]
        if not all(
            record["layer"].get(k)
            for k in ("kind", "implementation_url", "implementation_revision")
        ):
            errors.append("accepted contributions need a named, pinned implementation")
        if any(not isinstance(r, str) or not r.strip() for r in record["required_reviewers"]):
            errors.append("required reviewers must have nonempty identities")
        if not all(
            baseline.get(k)
            for k in (
                "dataset",
                "release",
                "import_repository",
                "import_revision",
                "input_hashes",
            )
        ):
            errors.append("accepted contributions need a pinned baseline and input hashes")
        if not record["required_checks"] or not record["evidence"]:
            errors.append("acceptance needs required checks and source evidence")
        # A failed or skipped required check cannot be hidden by another passed run.
        checks = {}
        for run in record["runs"]:
            if (
                not isinstance(run, dict)
                or run.get("baseline") != baseline
                or run.get("layer_revision") != record["layer"]["implementation_revision"]
            ):
                errors.append(
                    "accepted run must identify the exact baseline and candidate revision"
                )
                continue
            if not run.get("evidence_url"):
                errors.append("accepted run needs public validation evidence")
            for check in run.get("checks", []):
                checks.setdefault(check.get("id"), []).append(check)
        for name in record["required_checks"]:
            results = checks.get(name, [])
            if not results or any(
                c.get("status") != "passed" or not c.get("evidence_url") for c in results
            ):
                errors.append(f"required check {name} has missing, failed or skipped evidence")
        approved = {
            a.get("reviewer")
            for a in record["approvals"]
            if isinstance(a, dict)
            and a.get("evidence_url")
            and a.get("baseline") == baseline
            and a.get("layer_revision") == record["layer"]["implementation_revision"]
        }
        if not record["required_reviewers"] or not set(record["required_reviewers"]) <= approved:
            errors.append(
                "acceptance needs named reviewers and every approval tied to the exact baseline and layer revision"
            )
        if any(
            not isinstance(o, dict) or not o.get("disposition") or not o.get("evidence_url")
            for o in record["objections"]
        ):
            errors.append("objections need documented disposition and evidence")
    return errors


def check(root):
    errors, records = [], {}
    for path in sorted((root / "community/contributions").glob("*.json")):
        record = read(path)
        errors += [f"{path.name}: {e}" for e in validate(record)]
        if record.get("id") != path.stem:
            errors.append(f"{path.name}: id must equal the file name without .json")
        if record.get("id") in records:
            errors.append(f"duplicate contribution {record.get('id')}")
        records[record.get("id")] = record
    for packet in read(root / "community/recommendations/tracker.json"):
        record = records.get(packet.get("contribution_id"))
        state = packet.get("status")
        if state not in {"draft", "ready", "sent", "acknowledged", "resolved"}:
            errors.append("invalid recommendation state")
        if not record:
            errors.append("recommendation references unknown contribution")
        if state != "draft" and (
            not record or record["status"] != "accepted" or not packet.get("recipient")
        ):
            errors.append(
                "ready recommendation needs accepted contribution and confirmed recipient"
            )
        if state in {"sent", "acknowledged", "resolved"} and not packet.get("delivery_url"):
            errors.append("sent recommendation needs delivery evidence")
        if state in {"acknowledged", "resolved"} and not packet.get("response_url"):
            errors.append("provider response needs evidence")
    print(
        "\n".join(errors)
        if errors
        else f"Validated {len(records)} contribution records and recommendation references."
    )
    return int(bool(errors))


def compare(baseline_path, candidate_path):
    baseline, candidate = read(baseline_path), read(candidate_path)
    if not baseline.get("baseline_id") or baseline.get("baseline_id") != candidate.get(
        "baseline_id"
    ):
        raise ValueError("baseline identities differ or are missing")
    scope = {
        "background",
        "method",
        "functional_unit",
        "allocation",
        "boundary",
        "geography",
    }
    if not isinstance(baseline.get("scope"), dict) or scope - baseline["scope"].keys():
        raise ValueError("comparison scope is incomplete")
    if baseline["scope"] != candidate.get("scope"):
        raise ValueError("comparison scopes differ")
    if (
        not baseline.get("metrics")
        or baseline["metrics"].keys() != candidate.get("metrics", {}).keys()
    ):
        raise ValueError("metric sets differ or are empty; omitted metrics cannot hide regressions")
    metrics = {}
    for key, before in baseline["metrics"].items():
        after = candidate["metrics"][key]
        if not before.get("unit") or before["unit"] != after.get("unit"):
            raise ValueError(f"metric {key}: incompatible units")
        values = [before.get("value"), after.get("value")]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError(f"metric {key}: values must be finite numbers")
        delta = values[1] - values[0]
        if not math.isfinite(delta):
            raise ValueError(f"metric {key}: delta overflow")
        metrics[key] = {
            "baseline": values[0],
            "candidate": values[1],
            "delta": delta,
            "relative_delta": delta / abs(values[0]) if values[0] else None,
            "unit": before["unit"],
        }
    return {
        "baseline_id": baseline["baseline_id"],
        "scope": baseline["scope"],
        "input_sha256": {
            "baseline": digest(baseline_path),
            "candidate": digest(candidate_path),
        },
        "metrics": metrics,
        "checks": {
            "baseline": baseline.get("checks", []),
            "candidate": candidate.get("checks", []),
        },
        "interpretation": "Comparison only; scientific validation and partner review are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    command = sub.add_parser("compare")
    command.add_argument("baseline", type=Path)
    command.add_argument("candidate", type=Path)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        return check(Path(__file__).resolve().parents[1])
    # Refuse overwrite: old evidence must remain available for review.
    report = compare(args.baseline, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
