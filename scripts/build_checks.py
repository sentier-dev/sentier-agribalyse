"""Validate a normalized build audit without loading or publishing source inventories."""

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

KINDS = {"processes", "production", "technosphere", "biosphere"}
STAGES = ("source", "imported", "served")


def integer(value):
    return type(value) is int and value >= 0


def canonical(value):
    """Stable summary encoding: floats rounded to 12 significant digits, not file hashes."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite summary value")
        return {"float12": format(0.0 if value == 0 else value, ".12g")}
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def validate(data):
    errors = []
    for key in (
        "dataset",
        "release",
        "revision",
        "input_sha256",
        "scope",
        "counts",
        "changes",
        "methods",
        "probes",
        "packages",
        "artifacts",
    ):
        if key not in data:
            errors.append(f"missing {key}")
    if errors:
        return errors
    if not all(isinstance(data[k], str) and data[k] for k in ("dataset", "release", "revision")):
        errors.append("dataset, release and revision must be nonempty strings")
    hashes = data["input_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        errors.append("input_sha256 must contain pinned input hashes")
    else:
        for name, value in hashes.items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                errors.append(f"input {name}: invalid sha256")
    scope_keys = {
        "background",
        "method_source",
        "functional_unit",
        "allocation",
        "boundary",
        "geography",
        "representation",
    }
    if not isinstance(data["scope"], dict) or any(not data["scope"].get(k) for k in scope_keys):
        errors.append(
            "scope must identify background, method source, unit, allocation, boundary, geography and representation"
        )
    counts = data["counts"]
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(stage), dict)
        or set(counts[stage]) != KINDS
        or any(not integer(v) for v in counts[stage].values())
        for stage in STAGES
    ):
        return [*errors, "counts require all four nonnegative integer counts at each stage"]
    changes = data["changes"]
    if not isinstance(changes, list):
        return [*errors, "changes must be a list"]
    deltas, ids = {}, set()
    for change in changes:
        if not isinstance(change, dict):
            errors.append("invalid change record")
            continue
        transition, kind, delta = change.get("transition"), change.get("kind"), change.get("delta")
        if (
            transition not in {"source->imported", "imported->served"}
            or kind not in KINDS
            or type(delta) is not int
            or delta == 0
        ):
            errors.append("changes need a valid transition, kind and nonzero integer delta")
            continue
        if (
            not change.get("id")
            or change["id"] in ids
            or not change.get("reason")
            or not change.get("evidence_url")
        ):
            errors.append("changes need unique ids, reasons and evidence")
        ids.add(change.get("id"))
        deltas[(transition, kind)] = deltas.get((transition, kind), 0) + delta
    for before, after in pairwise(STAGES):
        for kind in sorted(KINDS):
            actual = counts[after][kind] - counts[before][kind]
            if actual != deltas.get((f"{before}->{after}", kind), 0):
                errors.append(f"unexplained {before}->{after} {kind} delta {actual}")
    methods = data["methods"]
    if not isinstance(methods, list) or not methods:
        return [*errors, "methods must contain a characterization audit"]
    method_ids = set()
    for method in methods:
        if not isinstance(method, dict) or not method.get("id") or method["id"] in method_ids:
            errors.append("methods need unique ids")
            continue
        method_ids.add(method["id"])
        emitted, required, factors, excluded = (
            method.get(k) for k in ("emitted", "required", "factors", "excluded")
        )
        if (
            not isinstance(emitted, list)
            or not isinstance(required, list)
            or not isinstance(factors, dict)
            or not isinstance(excluded, dict)
        ):
            errors.append("methods need emitted and required key lists, factors and exclusions")
            continue
        if any(not isinstance(k, str) or not k for k in emitted + required):
            errors.append("flow keys must be nonempty strings")
            continue
        if len(set(emitted)) != len(emitted) or len(set(required)) != len(required):
            errors.append("flow key lists must be unique")
        if not method.get("policy_evidence_url"):
            errors.append("required-flow policy needs source evidence")
        if not set(required) <= set(emitted):
            errors.append("required flow keys must belong to the emitted inventory")
        for key, value in factors.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                errors.append(f"{method['id']} {key}: non-finite or invalid factor")
        for key, entry in excluded.items():
            if (
                not isinstance(entry, dict)
                or not entry.get("reason")
                or not entry.get("evidence_url")
            ):
                errors.append(f"{method['id']} {key}: exclusion needs reason and evidence")
        for key in required:
            if key not in factors and key not in excluded:
                errors.append(f"{method['id']}: required emitted flow {key} lacks characterization")
        if set(factors) & set(excluded):
            errors.append("a flow cannot be both characterized and excluded")
    if not isinstance(data["packages"], list) or any(
        not isinstance(p, dict) or not p.get("id") or not p.get("revision")
        for p in data["packages"]
    ):
        errors.append("packages must list applied ids and revisions")
    if not isinstance(data["artifacts"], dict) or not data["artifacts"]:
        errors.append("artifacts must identify the actual output file hashes")
    else:
        for key, value in data["artifacts"].items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                errors.append(f"artifact {key}: invalid sha256")
    try:
        canonical(data["probes"])
    except ValueError as exc:
        errors.append(str(exc))
    if not isinstance(data["probes"], dict) or not data["probes"]:
        errors.append("probes must contain named reference results")
    return errors


def summary_identity(data):
    # Explicit allowlist excludes timestamps, importer versions and serving paths.
    # Exact artifact hashes are separate: this hash describes the measured summary.
    content = {
        k: data[k]
        for k in (
            "dataset",
            "release",
            "scope",
            "counts",
            "changes",
            "methods",
            "probes",
            "packages",
        )
    }
    content["changes"] = sorted(content["changes"], key=lambda c: c["id"])
    content["packages"] = sorted(content["packages"], key=lambda p: p["id"])
    content["methods"] = sorted(content["methods"], key=lambda m: m["id"])
    content["methods"] = [
        {**m, "emitted": sorted(m["emitted"]), "required": sorted(m["required"])}
        for m in content["methods"]
    ]
    encoded = json.dumps(
        canonical(content), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    data = json.loads(args.audit.read_text())
    errors = validate(data)
    print(
        json.dumps(
            {
                "status": "failed" if errors else "passed",
                "errors": errors,
                "summary_sha256": None if errors else summary_identity(data),
            },
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
