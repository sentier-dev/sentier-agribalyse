"""``RegionalSuffixApplier`` — postprocess linked exchanges with regional codes.

The :class:`matching.biosphere.BiosphereMatcher` applies the
``@<region>`` suffix on the synthetic code it emits, but the matcher
only runs for exchanges whose ``_best_candidate`` resolves a new
outcome. Many AGB exchanges are already linked by earlier pipeline
steps — :class:`transforms.biosphere_prelinker.BiospherePrelinker`
matches ~5.5M exchanges by ``code`` alone, before the matcher visits
them. For those the matcher's outcome may stay equal to the prior
link, and the regional dimension would be silently lost.

This applier closes that gap by sweeping every linked biosphere
exchange after all matching has completed and rewriting the
``input`` tuple to its ``<base_code>@<region>`` synthetic form when
the source name carries a recognised regional suffix.

The pass is idempotent: codes that already carry the ``@`` separator
are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from matching.regional_suffix import RegionalSuffixParser


@dataclass(frozen=True)
class RegionalSuffixApplierStats:
    n_visited: int = 0
    n_rewritten: int = 0
    n_already_synthetic: int = 0
    n_unsupported_db: int = 0
    n_no_input: int = 0
    n_skipped_non_water: int = 0


@dataclass(frozen=True)
class RegionalSuffixApplier:
    """Apply the ``@<region>`` synthetic code suffix to every regional
    biosphere flow already linked in ``sp.data``.

    Mirrors :class:`matching.biosphere.BiosphereMatcher`'s allow-list
    so the suffix lands only on the databases the augmented catalog
    knows how to synthesise rows for.
    """

    REGIONAL_DB_ALLOWLIST: ClassVar[frozenset[str]] = frozenset(
        {"ecoinvent-3.9.1-biosphere", "biosphere3"}
    )

    SEPARATOR: ClassVar[str] = "@"

    # Restrict the synthetic suffix to water-class flow names. Other
    # regionally-suffixed flows in the AGB inventory (``BOD5, FR``,
    # ``COD, CN``, ``Nitrogen oxides, DE``) ALSO carry regional
    # suffixes, but SimaPro EF v3.1 (adapted) only ships per-region
    # CFs for the **Water use** method. Moving those non-water flows
    # onto synthetic ``@<region>`` matrix rows strips them of the
    # inherited CFs they need on Particulate Matter / Acidification /
    # Eutrophication and silently zeroes their contributions — a 2-8x
    # regression on those methods. Keeping the suffix water-only
    # preserves the regional scope of this fix.
    CANONICAL_NAME_ALLOWLIST_PREFIXES: ClassVar[tuple[str, ...]] = ("water",)

    # Field where
    # :class:`transforms.regional_source_name_snapshotter.RegionalSourceNameSnapshotter`
    # records the pre-flowmap source name. The biosphere flowmap rewrites
    # ``exc["name"]`` to the canonical bio3 spelling (dropping the
    # country suffix); we need the original spelling here.
    SNAPSHOT_FIELD: ClassVar[str] = "_regional_source_name"

    parser: RegionalSuffixParser = field(default_factory=RegionalSuffixParser)

    def apply(self, sp_data: list[dict]) -> RegionalSuffixApplierStats:
        n_visited = 0
        n_rewritten = 0
        n_already = 0
        n_unsupported_db = 0
        n_no_input = 0
        n_skipped_non_water = 0
        for ds in sp_data:
            for exc in ds.get("exchanges", ()):
                if exc.get("type") != "biosphere":
                    continue
                n_visited += 1
                input_ = exc.get("input")
                if not (isinstance(input_, tuple) and len(input_) == 2):
                    n_no_input += 1
                    continue
                db, code = input_
                if not isinstance(code, str):
                    continue
                if self.SEPARATOR in code:
                    n_already += 1
                    continue
                # The flowmap rewrites ``exc.name`` (``Water, IN`` →
                # ``Water``) before this step runs. The pre-flowmap
                # snapshot field carries the original AGB-side spelling
                # with regional suffix intact. Fall back to ``simapro
                # name`` (set by
                # ``RemoveBiosphereLocationPrefixIfFlowInSameLocation``)
                # and finally to ``name`` itself for safety.
                name = (
                    exc.get(self.SNAPSHOT_FIELD) or exc.get("simapro name") or exc.get("name") or ""
                ).strip()
                if not name:
                    continue
                _, region = self.parser.parse(name)
                if not region:
                    continue
                if db not in self.REGIONAL_DB_ALLOWLIST:
                    n_unsupported_db += 1
                    continue
                # Restrict to water-class names. The bio3 canonical
                # ``name`` (post-flowmap) is the authoritative type
                # signal — ``Nitrogen oxides`` with a regional suffix
                # is NOT a water flow and must keep its base UUID so
                # the inherited PM / acidification / eutrophication
                # CFs still apply.
                canonical = (exc.get("name") or "").strip().lower()
                if not canonical.startswith(self.CANONICAL_NAME_ALLOWLIST_PREFIXES):
                    n_skipped_non_water += 1
                    continue
                exc["input"] = (db, f"{code}{self.SEPARATOR}{region}")
                n_rewritten += 1
        return RegionalSuffixApplierStats(
            n_visited=n_visited,
            n_rewritten=n_rewritten,
            n_already_synthetic=n_already,
            n_unsupported_db=n_unsupported_db,
            n_no_input=n_no_input,
            n_skipped_non_water=n_skipped_non_water,
        )
