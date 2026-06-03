"""``dds-compare-cfs`` — print per-method CF distribution stats side-by-side.

Compares ``cache/simapro-EF31-adapted-cfs.parquet`` (SimaPro's adapted EF 3.1
method export) against one of:

* ``registry/method_cfs/*.parquet`` — the built EF v3.1 CF registry that
  the scoring pipeline actually uses. **This is the default.** For every
  ecoinvent biosphere flow the registry has a CF for, ``FlowLevelCfJoiner``
  looks up the SimaPro CF for the same substance/compartment, then the
  per-method stats are computed over the joined ``(sp_cf, ef_cf)`` pairs.
  Both sides of every stat are now at the same granularity, so deltas
  measure real CF disagreement rather than data-shape mismatch. The
  joined per-flow frame is also written to
  ``cache/cf_per_flow_joined.parquet`` for future drill-down.
* ``source/EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet`` — the raw JRC
  EF v3.1 reference (opt-in via ``--source raw``). Computes distribution
  stats over the full SimaPro and JRC substance × compartment exports
  independently. Useful for build-pipeline debugging and to inspect the
  EF v3.1 universe without the registry's flow-matching applied.

Output is one block per LCIA method on stdout (count/min/max/mean/
median/std/sum on each side) plus ``dashboard/cf_stats.csv`` for the
CF-stats dashboard tab.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

import pandas as pd

from cli._base import BaseCli
from core.logging import Logging
from ef.cf_flow_join import (
    BiosphereCatalogLoader,
    ContextNormaliser,
    FlowLevelCfJoiner,
    JoinedFlowFrame,
    JoinedFlowParquetWriter,
    SimaProCfIndex,
)
from ef.simapro_cf_table import SimaProEFCfTable
from reporting import CfStatsEmitter


class MethodNameNormalizer:
    """Normalize SimaPro and EF31 LCIA method names so equivalents collide.

    Examples:
        ``"Climate change - Biogenic"`` and ``"Climate change-Biogenic"``
        both → ``"climate change biogenic"``.

        ``"Ionising radiation, human health"`` → ``"ionising radiation"``
        (matches SimaPro's ``"Ionising radiation"``).

        ``"Climate change - Land use and LU change"`` →
        ``"climate change land use and land use change"`` to match the EF31
        ``"Climate change-Land use and land use change"``.
    """

    _EF_PREFIX: ClassVar[str] = "ef-"
    _SEPARATOR_RE: ClassVar[re.Pattern[str]] = re.compile(r"[-_,]")
    _WHITESPACE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\s+")
    _HUMAN_HEALTH_SUFFIX: ClassVar[str] = " human health"
    # Expand common abbreviations that appear in SimaPro names but not EF31:
    # "lu" as a standalone word stands for "land use".
    _LU_ABBREV_RE: ClassVar[re.Pattern[str]] = re.compile(r"\blu\b")

    @classmethod
    def normalize(cls, s: str) -> str:
        out = s.lower().strip()
        if out.startswith(cls._EF_PREFIX):
            out = out[len(cls._EF_PREFIX) :]
        out = cls._SEPARATOR_RE.sub(" ", out)
        out = cls._WHITESPACE_RE.sub(" ", out).strip()
        if out.endswith(cls._HUMAN_HEALTH_SUFFIX):
            out = out[: -len(cls._HUMAN_HEALTH_SUFFIX)].rstrip()
        out = cls._LU_ABBREV_RE.sub("land use", out)
        out = cls._WHITESPACE_RE.sub(" ", out).strip()
        return out


@dataclass(frozen=True)
class MethodAliasResolver:
    """Map SimaPro LCIA method names to the built registry's method names.

    Wraps :data:`SimaProEFCfTable.METHOD_TO_OUR_KEY` — the project's single
    source of truth for SimaPro ↔ registry method-name aliasing. Also folds
    SimaPro's ``- inorganics`` / ``- organics`` sub-methods into their root
    parent (the registry only stores the parent).
    """

    _SUBMETHOD_SUFFIXES: ClassVar[tuple[str, ...]] = (" - inorganics", " - organics")

    def strip_submethod(self, simapro_method: str) -> str:
        for suffix in self._SUBMETHOD_SUFFIXES:
            if simapro_method.endswith(suffix):
                return simapro_method[: -len(suffix)]
        return simapro_method

    def simapro_to_registry(self, simapro_method: str) -> str | None:
        """Return the registry category name, or ``None`` if no alias exists."""
        root = self.strip_submethod(simapro_method)
        key = SimaProEFCfTable.METHOD_TO_OUR_KEY.get(root)
        return key[0] if key is not None else None


class CfStatsComputer:
    """Compute the 7-stat summary for a pandas CF series."""

    STAT_KEYS: ClassVar[tuple[str, ...]] = (
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "sum",
    )

    @classmethod
    def compute(cls, series: pd.Series) -> dict[str, float]:
        return {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "sum": float(series.sum()),
        }


@dataclass(frozen=True)
class SimaproCfLoader:
    """Read the SimaPro adapted-CFs parquet, return ``(method_raw, cf)``."""

    path: Path

    _METHOD_COL: ClassVar[str] = "simapro_method"
    _CF_COL: ClassVar[str] = "cf"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"SimaPro CF parquet not found: {self.path}")
        df = pd.read_parquet(self.path, columns=[self._METHOD_COL, self._CF_COL])
        return df.rename(columns={self._METHOD_COL: "method_raw", self._CF_COL: "cf"})


@dataclass(frozen=True)
class Ef31CfLoader:
    """Read the JRC EF v3.1 CFs parquet, return ``(method_raw, cf)``."""

    path: Path

    _METHOD_COL: ClassVar[str] = "LCIAMethod_name"
    _CF_COL: ClassVar[str] = "CF EF3.1"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"EF v3.1 CF parquet not found: {self.path}")
        df = pd.read_parquet(self.path, columns=[self._METHOD_COL, self._CF_COL])
        return df.rename(columns={self._METHOD_COL: "method_raw", self._CF_COL: "cf"})


@dataclass(frozen=True)
class MethodCfsRegistryLoader:
    """Read the built EF v3.1 CF registry under ``registry/method_cfs/``.

    The directory contains one parquet per method plus ``_index.json``
    mapping each method's 4-tuple key (database, EF version, category,
    indicator) to its parquet slug. ``load()`` concatenates every per-method
    parquet's ``amount`` column into a single ``(method_raw, cf)`` DataFrame
    where ``method_raw`` is the registry's category name (the 3rd element of
    the key tuple, e.g. ``"energy resources: non-renewable"``).
    """

    path: Path

    _INDEX_NAME: ClassVar[str] = "_index.json"
    _AMOUNT_COL: ClassVar[str] = "amount"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Method CFs registry directory not found: {self.path}")
        index_path = self.path / self._INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(f"Registry index file not found: {index_path}")
        index = json.loads(index_path.read_text())
        frames: list[pd.DataFrame] = []
        for entry in index["methods"]:
            method_name = entry["key"][2]
            df = pd.read_parquet(self.path / entry["slug"], columns=[self._AMOUNT_COL])
            df = df.assign(method_raw=method_name).rename(columns={self._AMOUNT_COL: "cf"})
            frames.append(df[["method_raw", "cf"]])
        if not frames:
            return pd.DataFrame(
                {
                    "method_raw": pd.Series(dtype="object"),
                    "cf": pd.Series(dtype="float64"),
                }
            )
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class MethodRow:
    """One row of the comparison table.

    ``per_flow_diff_stats`` is populated when the comparator can compute
    a per-(method, flow) join (currently only the joined-registry path).
    When present, :class:`reporting.CfStatsEmitter` uses these stats for
    the ``diff_<stat>`` columns so the displayed %diff reflects per-flow
    agreement rather than the inherently-noisy "diff of summary stats"
    (which blows up when one side's stat is near zero — e.g. Water use
    with 2 matched flows reading ``diff_mean = -100%`` when the actual
    per-flow disagreement averages ``-6%``). ``None`` falls back to the
    historical formula ``(ef_<stat> - sp_<stat>) / max(|sp|, |ef|)``.
    """

    display_name: str
    norm_key: str
    simapro_stats: dict[str, float] | None
    ef31_stats: dict[str, float] | None
    per_flow_diff_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class CfComparator:
    """Build comparison rows from the two loaders, applying name filters.

    When ``alias_resolver`` is ``None`` (default), both sides are normalized
    with :class:`MethodNameNormalizer` — appropriate when the right-hand side
    is the raw EF31 parquet that uses similar method names to SimaPro.

    When ``alias_resolver`` is set, the SimaPro side is mapped to the
    registry's category names via :data:`SimaProEFCfTable.METHOD_TO_OUR_KEY`
    (and sub-method suffixes are folded into the root). The right-hand side
    is the built registry, whose category names are already canonical, so
    they're only lowercased/stripped.
    """

    simapro_loader: SimaproCfLoader
    ef31_loader: Ef31CfLoader
    method_filters: tuple[str, ...]
    alias_resolver: MethodAliasResolver | None = None

    def build_rows(self) -> list[MethodRow]:
        sp_df = self.simapro_loader.load()
        ef_df = self.ef31_loader.load()

        sp_by_norm = self._group_by_norm(sp_df, self._sp_key, self._sp_display)
        ef_by_norm = self._group_by_norm(ef_df, self._target_key, self._identity)

        rows: list[MethodRow] = []
        for norm_key in sorted(set(sp_by_norm) | set(ef_by_norm)):
            sp_block = sp_by_norm.get(norm_key)
            ef_block = ef_by_norm.get(norm_key)

            display_name = sp_block["raw_name"] if sp_block is not None else ef_block["raw_name"]
            if not self._matches_filters(display_name):
                continue

            rows.append(
                MethodRow(
                    display_name=display_name,
                    norm_key=norm_key,
                    simapro_stats=(
                        CfStatsComputer.compute(sp_block["cf"]) if sp_block is not None else None
                    ),
                    ef31_stats=(
                        CfStatsComputer.compute(ef_block["cf"]) if ef_block is not None else None
                    ),
                )
            )
        return rows

    def _sp_key(self, name: str) -> str | None:
        if self.alias_resolver is not None:
            return self.alias_resolver.simapro_to_registry(name)
        return MethodNameNormalizer.normalize(name)

    def _sp_display(self, name: str) -> str:
        if self.alias_resolver is not None:
            return self.alias_resolver.strip_submethod(name)
        return name

    def _target_key(self, name: str) -> str:
        if self.alias_resolver is not None:
            return name.lower().strip()
        return MethodNameNormalizer.normalize(name)

    @staticmethod
    def _identity(name: str) -> str:
        return name

    @staticmethod
    def _group_by_norm(
        df: pd.DataFrame,
        key_fn: Callable[[str], str | None],
        display_fn: Callable[[str], str],
    ) -> dict[str, dict]:
        """Return ``{norm_key: {"raw_name": <display>, "cf": <Series>}}``.

        Rows for which ``key_fn`` returns ``None`` (no alias) are dropped.
        ``display_fn`` maps a raw method name to its display form (e.g.
        folds ``- inorganics`` / ``- organics`` into the root).
        """
        if df.empty:
            return {}
        df = df.assign(norm_key=df["method_raw"].map(key_fn))
        df = df[df["norm_key"].notna()]
        out: dict[str, dict] = {}
        for norm_key, group in df.groupby("norm_key", sort=False):
            out[norm_key] = {
                "raw_name": display_fn(group["method_raw"].iloc[0]),
                "cf": group["cf"],
            }
        return out

    def _matches_filters(self, display_name: str) -> bool:
        if not self.method_filters:
            return True
        lowered = display_name.lower()
        return any(f.lower() in lowered for f in self.method_filters)


@dataclass(frozen=True)
class JoinedFlowCfComparator:
    """Build :class:`MethodRow` list by joining registry CFs against SimaPro
    at the ecoinvent biosphere flow level.

    Per registry method:

    1. Load ``registry/method_cfs/<slug>/cfs.parquet`` (cols: database,
       code, amount = ef_cf).
    2. Apply :class:`FlowLevelCfJoiner` to attach an ``sp_cf`` to every
       row via the biosphere catalog + context normaliser + SP index.
    3. Compute :class:`CfStatsComputer.compute` on the ``sp_cf`` and
       ``ef_cf`` columns directly. pandas' default NaN-skipping means
       SP-side stats automatically ignore flows that didn't match in SP.

    Returns the rows plus the list of joined frames so the CLI can also
    write the per-flow parquet sidecar.
    """

    registry_dir: Path
    joiner: FlowLevelCfJoiner
    method_filters: tuple[str, ...]

    _INDEX_NAME: ClassVar[str] = "_index.json"
    _GLOBAL_CFS_FILE: ClassVar[str] = "cfs.parquet"
    _PARQUET_COLS: ClassVar[tuple[str, ...]] = ("database", "code", "amount")

    def build_rows(self) -> tuple[list[MethodRow], list[JoinedFlowFrame]]:
        index_path = self.registry_dir / self._INDEX_NAME
        if not index_path.exists():
            raise FileNotFoundError(f"Registry index not found: {index_path}")
        index = json.loads(index_path.read_text())
        rows: list[MethodRow] = []
        frames: list[JoinedFlowFrame] = []
        for entry in index["methods"]:
            method_key = tuple(entry["key"])
            display_name = self._registry_key_to_sp_name(method_key)
            if not self._matches_filters(display_name):
                continue
            # Read only cfs.parquet (global CFs, one row per ecoinvent flow).
            # The slug directory may also hold ``regional_cfs.parquet`` —
            # ~3300 country-specific water-use CFs for the same biosphere
            # codes. Including those would create 3300 "matches" for a
            # single SP global CF, inflating the apparent disagreement.
            # Regional CFs are scored separately by RegionalCorrectionBuilder
            # and have no SimaPro counterpart, so they're out of scope here.
            ef_cfs = pd.read_parquet(
                self.registry_dir / entry["slug"] / self._GLOBAL_CFS_FILE,
                columns=list(self._PARQUET_COLS),
            )
            jf = self.joiner.join_method(method_key, ef_cfs)
            frames.append(jf)
            sp_stats, ef_stats, per_flow_diff_stats = self._intersected_stats(jf)
            rows.append(
                MethodRow(
                    display_name=display_name,
                    norm_key=display_name.lower(),
                    simapro_stats=sp_stats,
                    ef31_stats=ef_stats,
                    per_flow_diff_stats=per_flow_diff_stats,
                )
            )
        return rows, frames

    @staticmethod
    def _intersected_stats(
        jf: JoinedFlowFrame,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float] | None]:
        """Return ``(sp_stats, ef_stats, per_flow_diff_stats)`` for one
        method's joined frame.

        Distribution stats (``min, max, mean, median, std, sum``) are
        computed over the **intersection** — the subset of flows where
        SimaPro carries a CF — for both sides. Without this, SP-side
        stats would be over the matched subset while EF-side stats would
        be over the full registry, reintroducing a data-shape mismatch
        at a finer granularity.

        ``per_flow_diff_stats`` aggregates the per-(method, flow)
        symmetric relative diff ``(ef_cf - sp_cf) / max(|sp_cf|, |ef_cf|)``
        over the matched flows. This is the honest "how much do SP and
        EF disagree on the same flow" metric: the historical
        ``(ef_<stat> - sp_<stat>) / max(...)`` formula goes to ±100%
        whenever one side's summary stat lands near zero (because
        positive and negative CFs cancel), producing misleading
        customer-facing numbers (e.g. Water use with 2 matched flows
        reading ``diff_mean = -100%`` when the actual per-flow
        disagreement averages ``-6%``).

        Only the ``count`` stat keeps the asymmetric form: ``sp_count``
        is the size of the intersection while ``ef_count`` is the total
        number of flows in this method's registry. This keeps the
        dashboard cardinality column meaningful — it reads as "flows SP
        covers / flows the registry has".
        """
        df = jf.df
        if df.empty:
            return (None, None, None)
        matched = df[df["sp_cf"].notna()]
        total_count = len(df)
        if matched.empty:
            ef = df["ef_cf"]
            ef_stats = {
                "count": total_count,
                "min": float(ef.min()),
                "max": float(ef.max()),
                "mean": float(ef.mean()),
                "median": float(ef.median()),
                "std": float(ef.std()),
                "sum": float(ef.sum()),
            }
            return (None, ef_stats, None)
        sp = matched["sp_cf"]
        ef_match = matched["ef_cf"]
        sp_stats = {
            "count": int(sp.count()),
            "min": float(sp.min()),
            "max": float(sp.max()),
            "mean": float(sp.mean()),
            "median": float(sp.median()),
            "std": float(sp.std()),
            "sum": float(sp.sum()),
        }
        ef_stats = {
            "count": total_count,
            "min": float(ef_match.min()),
            "max": float(ef_match.max()),
            "mean": float(ef_match.mean()),
            "median": float(ef_match.median()),
            "std": float(ef_match.std()),
            "sum": float(ef_match.sum()),
        }
        # Per-flow symmetric relative diff.
        denom = pd.concat([sp.abs(), ef_match.abs()], axis=1).max(axis=1)
        per_flow = pd.Series(0.0, index=sp.index)
        nonzero = denom > 0
        per_flow.loc[nonzero] = (ef_match[nonzero] - sp[nonzero]) / denom[nonzero]
        per_flow_diff_stats = {
            "count": int(per_flow.count()),
            "min": float(per_flow.min()),
            "max": float(per_flow.max()),
            "mean": float(per_flow.mean()),
            "median": float(per_flow.median()),
            "std": float(per_flow.std()),
            "sum": float(per_flow.sum()),
        }
        return (sp_stats, ef_stats, per_flow_diff_stats)

    def _matches_filters(self, display_name: str) -> bool:
        if not self.method_filters:
            return True
        lowered = display_name.lower()
        return any(f.lower() in lowered for f in self.method_filters)

    @staticmethod
    def _registry_key_to_sp_name(method_key: tuple) -> str:
        cat, ind = method_key[2], method_key[3]
        for sp_name, our_key in SimaProEFCfTable.METHOD_TO_OUR_KEY.items():
            if our_key == (cat, ind):
                return sp_name
        return cat


@dataclass(frozen=True)
class ConsoleReporter:
    """Format ``MethodRow`` lists as side-by-side stat blocks on stdout."""

    decimals: int

    _COL_LABEL_WIDTH: ClassVar[int] = 42
    _COL_VALUE_WIDTH: ClassVar[int] = 14
    _MISSING: ClassVar[str] = "—"

    def render(self, rows: list[MethodRow], stream: TextIO | None = None) -> None:
        stream = stream if stream is not None else sys.stdout
        for row in rows:
            stream.write(self._format_block(row))
            stream.write("\n")
        stream.write(self._format_footer(rows))
        stream.write("\n")

    def _format_block(self, row: MethodRow) -> str:
        header = (
            f"{row.display_name:<{self._COL_LABEL_WIDTH}}"
            f"{'SimaPro':>{self._COL_VALUE_WIDTH}}"
            f"{'EF31':>{self._COL_VALUE_WIDTH}}\n"
        )
        lines = [header]
        for key in CfStatsComputer.STAT_KEYS:
            label = f"  {key}"
            sp_cell = self._format_cell(key, row.simapro_stats)
            ef_cell = self._format_cell(key, row.ef31_stats)
            lines.append(
                f"{label:<{self._COL_LABEL_WIDTH}}"
                f"{sp_cell:>{self._COL_VALUE_WIDTH}}"
                f"{ef_cell:>{self._COL_VALUE_WIDTH}}\n"
            )
        return "".join(lines)

    def _format_cell(self, key: str, stats: dict[str, float] | None) -> str:
        if stats is None:
            return self._MISSING
        value = stats[key]
        if key == "count":
            return f"{int(value):d}"
        return f"{value:.{self.decimals}e}"

    def _format_footer(self, rows: list[MethodRow]) -> str:
        only_sp = sum(1 for r in rows if r.simapro_stats is not None and r.ef31_stats is None)
        only_ef = sum(1 for r in rows if r.simapro_stats is None and r.ef31_stats is not None)
        both = sum(1 for r in rows if r.simapro_stats is not None and r.ef31_stats is not None)
        return f"methods only in SimaPro: {only_sp}; only in EF31: {only_ef}; aligned: {both}"


@dataclass(frozen=True)
class CompareConfig:
    """Per-run knobs (not paths — those come from ``Settings.paths``)."""

    SOURCE_REGISTRY: ClassVar[str] = "registry"
    SOURCE_RAW: ClassVar[str] = "raw"
    VALID_SOURCES: ClassVar[tuple[str, ...]] = (SOURCE_REGISTRY, SOURCE_RAW)

    decimals: int = 4
    method_filters: tuple[str, ...] = ()
    source: str = SOURCE_REGISTRY


class CompareCfsCli(BaseCli):
    """``dds-compare-cfs`` entry point."""

    PROG: ClassVar[str] = "cli.compare_cfs"
    DESCRIPTION: ClassVar[str] = (
        "Print per-LCIA-method CF distribution stats side-by-side (SimaPro adapted vs EF v3.1)."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog=cls.PROG, description=cls.DESCRIPTION)
        p.add_argument(
            "--source",
            choices=CompareConfig.VALID_SOURCES,
            default=CompareConfig.SOURCE_REGISTRY,
            help=(
                "EF31 comparison target. 'registry' (default) compares against "
                "the built registry/method_cfs/*.parquet (what the scoring "
                "pipeline uses) via a per-flow join: for every ecoinvent "
                "biosphere flow in the registry, the SimaPro CF for the same "
                "substance/compartment is looked up and the per-method stats "
                "are computed over the joined (sp_cf, ef_cf) pairs. 'raw' "
                "compares against the raw JRC EF v3.1 parquet "
                "(source/EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet) "
                "via distribution stats over the full substance x compartment "
                "exports independently -- useful for build-pipeline debugging."
            ),
        )
        p.add_argument(
            "--method",
            action="append",
            default=[],
            metavar="NAME",
            help=(
                "Case-insensitive substring filter on the SimaPro method "
                "name. Repeatable; matches if ANY filter is a substring."
            ),
        )
        p.add_argument(
            "--decimals",
            type=int,
            default=4,
            help="Float precision in the printed table (default: 4).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        config = CompareConfig(
            decimals=args.decimals,
            method_filters=tuple(args.method),
            source=args.source,
        )
        if config.source == CompareConfig.SOURCE_REGISTRY:
            rows = self._run_joined_registry(config)
        else:
            rows = self._run_raw_jrc(config)
        if config.method_filters and not rows:
            raise ValueError(f"No methods matched filters: {list(config.method_filters)}")
        ConsoleReporter(decimals=config.decimals).render(rows)
        csv_path = CfStatsEmitter(
            out_path=self.settings.paths.dashboard_cf_stats_csv,
        ).write(rows)
        Logging.get(self.PROG).info(
            "compare.dashboard.emitted",
            path=str(csv_path),
            n_rows=len(rows),
        )
        Logging.get(self.PROG).info("compare.done", source=config.source, n_methods=len(rows))

    def _run_joined_registry(self, config: CompareConfig) -> list[MethodRow]:
        paths = self.settings.paths
        catalog = BiosphereCatalogLoader(paths.registry_biosphere_catalog).load()
        normaliser = ContextNormaliser.from_path(paths.registry_context_normalisation)
        # SimaProCfIndex needs the full SP schema (name, compartment, sub,
        # cas, cf), not the (method_raw, cf) shape SimaproCfLoader exposes.
        sp_full = pd.read_parquet(paths.simapro_ef31_cache)
        sp_index = SimaProCfIndex.from_dataframe(sp_full)
        joiner = FlowLevelCfJoiner(
            catalog=catalog,
            normaliser=normaliser,
            sp_index=sp_index,
        )
        comparator = JoinedFlowCfComparator(
            registry_dir=paths.registry_method_cfs_dir,
            joiner=joiner,
            method_filters=config.method_filters,
        )
        rows, frames = comparator.build_rows()
        joined_path = JoinedFlowParquetWriter(
            out_path=paths.cache_cf_per_flow_joined_parquet,
        ).write(frames)
        Logging.get(self.PROG).info(
            "compare.joined_parquet.written",
            path=str(joined_path),
            n_methods=len(frames),
            n_rows=sum(len(jf.df) for jf in frames),
        )
        return rows

    def _run_raw_jrc(self, config: CompareConfig) -> list[MethodRow]:
        comparator = CfComparator(
            simapro_loader=SimaproCfLoader(self.settings.paths.simapro_ef31_cache),
            ef31_loader=Ef31CfLoader(self.settings.paths.ef_cf_parquet),
            method_filters=config.method_filters,
            alias_resolver=None,
        )
        return comparator.build_rows()


def main() -> int:
    return CompareCfsCli().run()


if __name__ == "__main__":
    sys.exit(main())
