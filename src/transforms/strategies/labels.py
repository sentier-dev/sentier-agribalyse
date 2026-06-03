"""Label-normalisation strategies — lifted from ``bw2io.strategies``.

Pure dict transforms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizeSimaproLabelsToBrightwayStandard:
    """Rename ``context`` → ``categories`` and ``identifier`` → ``code`` on
    unlinked exchanges. Mirrors
    ``bw2io.strategies.normalize_simapro_labels_to_brightway_standard``.

    Some randonneur transformations use the more standard (non-Brightway)
    labels; the SimaPro/EF datapackages produced by ``RandonneurPackagesExporter``
    end up with ``context`` keys that the linker expects to see as
    ``categories`` afterwards.
    """

    # bw2io.importers.base.apply_strategy reads ``strategy.__name__`` for its
    # progress log; instances don't inherit ``__name__`` from the class, so
    # expose it explicitly. Without this, ``sp.apply_strategy(...)`` raises
    # AttributeError before ever invoking ``__call__``.
    __name__ = "normalize_simapro_labels_to_brightway_standard"

    name: str = "normalize_simapro_labels_to_brightway_standard"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            for exc in ds.get("exchanges", []):
                if "input" in exc:
                    continue
                if "context" in exc and "categories" not in exc:
                    exc["categories"] = tuple(exc["context"])
                if "identifier" in exc and "code" not in exc:
                    exc["code"] = exc["identifier"]
        return data
