"""``BacktestPipeline`` — score every mapped product against ADEME's reference."""

from __future__ import annotations

import multiprocessing as _mp
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pandas as pd

from config import Settings
from core.logging import Logging
from readers import ParquetReader
from reporting.backtest_dashboard_csv import BacktestPass1Emitter
from reporting.near_zero_floor import NearZeroFloor
from scoring.native_scorer import NativeLciaScorer, NativeScoreWorker, NativeWorkerPayload
from scoring.product_catalog import ProductCatalog
from scoring.scorer import ScoringResult
from scoring.scoring_package import ScoringPackage
from scoring.scoring_package_locator import ScoringPackageLocator


@dataclass(frozen=True)
class BacktestOptions:
    solver: str = "pardiso"
    n_workers: int = 1
    n_products: int | None = None
    """If set, score only the first N reference rows (deterministic order). For quick spot-checks."""

    match_mode: str = "all"
    """How to map ADEME reference rows → AGB DB activities.

    * ``"all"`` — current default: CIQUAL code → exact LCI Name → substring fallback.
    * ``"exact_name"`` — only keep rows whose ``LCI Name`` exactly matches an
      AGB DB process/product name (the SimaPro-style restored name). Used for
      the curated "32 products with exact-name match to SimaPro" subset.
    """


@dataclass
class BacktestPipeline:
    settings: Settings
    options: BacktestOptions = field(default_factory=BacktestOptions)

    INFO_COLS: ClassVar[tuple[str, ...]] = (
        "Code AGB",
        "Code CIQUAL",
        "Groupe d'aliment",
        "Sous-groupe d'aliment",
        "Nom du Produit",
        "LCI Name",
        "Storage",
    )

    # ADEME's French storage taxonomy → substring expected in the AGB
    # English product name. Used by Pass 1.5 in ``_map_products`` to
    # disambiguate same-Ciqual product candidates (e.g. Confit de canard
    # 8110 has both a canned ``Ambient (long)`` and a chilled
    # ``Preserved duck`` Pack twin, both tagged ``[Ciqual code: 8110]``).
    # "Glacé" in ADEME's taxonomy means preserved shelf-stable (olives in
    # brine, tapenade, oil-confit) — not "frozen"; the corresponding AGB
    # English label is ``Ambient (long)``. ``Congelé`` is the dedicated
    # frozen code.
    ADEME_STORAGE_TO_AGB: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "Ambiant (long)": "Ambient (long)",
            "Glacé": "Ambient (long)",
            "Ambiant (moyen)": "Ambient (medium)",
            "Ambiant (court)": "Ambient (short)",
            "Congelé": "Frozen",
        }
    )

    NEAR_ZERO_FACTOR: ClassVar[float] = 0.01
    """``|x| < factor * median(|reference|)`` → both computed and reference rounded to 0."""

    DASHBOARD_PASS1_CSV: ClassVar[str] = "backtest_pass1.csv"

    METHOD_TO_ADEME: ClassVar[Mapping[tuple[str, ...], tuple[int, str]]] = MappingProxyType(
        {
            ("ecoinvent-3.9.1", "EF v3.1", "climate change", "global warming potential (GWP100)"): (
                13,
                "Changement climatique",
            ),
            ("ecoinvent-3.9.1", "EF v3.1", "ozone depletion", "ozone depletion potential (ODP)"): (
                14,
                "Appauvrissement de la couche d'ozone",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "ionising radiation: human health",
                "human exposure efficiency relative to u235",
            ): (15, "Rayonnements ionisants"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "photochemical oxidant formation: human health",
                "tropospheric ozone concentration increase",
            ): (16, "Formation photochimique d'ozone"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "particulate matter formation",
                "impact on human health",
            ): (
                17,
                "Particules fines",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "human toxicity: non-carcinogenic",
                "comparative toxic unit for human (CTUh)",
            ): (18, "Effets tox. non-cancérogènes"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "human toxicity: carcinogenic",
                "comparative toxic unit for human (CTUh)",
            ): (19, "Effets tox. cancérogènes"),
            ("ecoinvent-3.9.1", "EF v3.1", "acidification", "accumulated exceedance (AE)"): (
                20,
                "Acidification terrestre et eaux douces",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "eutrophication: freshwater",
                "fraction of nutrients reaching freshwater end compartment (P)",
            ): (21, "Eutrophisation eaux douces"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "eutrophication: marine",
                "fraction of nutrients reaching marine end compartment (N)",
            ): (22, "Eutrophisation marine"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "eutrophication: terrestrial",
                "accumulated exceedance (AE)",
            ): (
                23,
                "Eutrophisation terrestre",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "ecotoxicity: freshwater",
                "comparative toxic unit for ecosystems (CTUe)",
            ): (24, "Écotoxicité eaux douces"),
            ("ecoinvent-3.9.1", "EF v3.1", "land use", "soil quality index"): (
                25,
                "Utilisation du sol",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "water use",
                "user deprivation potential (deprivation-weighted water consumption)",
            ): (26, "Épuisement des ressources eau"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "energy resources: non-renewable",
                "abiotic depletion potential (ADP): fossil fuels",
            ): (27, "Épuisement des ressources énergétiques"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "material resources: metals/minerals",
                "abiotic depletion potential (ADP): elements (ultimate reserves)",
            ): (28, "Épuisement des ressources minéraux"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "climate change: biogenic",
                "global warming potential (GWP100)",
            ): (29, "CC - émissions biogéniques"),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "climate change: fossil",
                "global warming potential (GWP100)",
            ): (
                30,
                "CC - émissions fossiles",
            ),
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "climate change: land use and land use change",
                "global warming potential (GWP100)",
            ): (31, "CC - changement d'affectation des sols"),
        }
    )

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def run(self) -> dict[str, Any]:
        s = self.settings
        s.paths.ensure_runtime_dirs()
        out_dir = s.paths.dashboard_backtest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        pkg = self._load_scoring_package()
        catalog = ProductCatalog.load(self.settings.paths.registry_product_catalog)

        df = self._load_ademe_reference()
        if self.options.n_products is not None:
            df = df.head(self.options.n_products).copy()
            self._log.info("backtest.subset", n_products=len(df))
        row_to_key = self._map_products(df, catalog)

        if self.options.match_mode == "exact_name":
            # Drop unmapped rows so scores/diff/summary parquets only carry the
            # exact-name-matched subset.
            mapped_idx = [idx for idx in df.index if row_to_key.get(idx) is not None]
            df = df.loc[mapped_idx].reset_index(drop=True)
            row_to_key = self._map_products(df, catalog)
            self._log.info("backtest.exact_name.kept", n_products=len(df))

        methods = self._registered_methods(pkg)
        if not methods:
            self._log.warning("backtest.no_methods", note="run link_all first")
            return {"warning": "no methods registered"}

        scores = self._compute_scores(row_to_key, methods, pkg, catalog)
        artifacts = self._compare_and_save(df, row_to_key, scores, methods, out_dir)
        return {"out_dir": str(out_dir), "artifacts": artifacts}

    # ------------------------------------------------------------------

    @staticmethod
    def _split_chunks(items: list, n: int) -> list[list]:
        """Round-robin split into ``n`` non-empty chunks for parallel dispatch."""
        if n <= 0:
            n = 1
        buckets: list[list] = [[] for _ in range(n)]
        for i, item in enumerate(items):
            buckets[i % n].append(item)
        return [b for b in buckets if b]

    def _load_scoring_package(self) -> ScoringPackage:
        """Load the ScoringPackage recorded by the last link run (via run_report.json)."""
        return ScoringPackageLocator(settings=self.settings).load()

    EMBALLAGE_CORRIGE_TOKEN: ClassVar[str] = "emballage corrig"

    def _load_ademe_reference(self) -> pd.DataFrame:
        path = self.settings.paths.ademe_reference_synthese_raw
        df_raw = ParquetReader().read(path)
        data = df_raw.iloc[3:]
        # Synthese columns 0-5 are the unambiguous identifiers
        # (Code AGB, Code CIQUAL, group, subgroup, French name, English
        # LCI name); column 8 is the storage taxonomy (``Glacé`` /
        # ``Ambiant (long)`` / ``Congelé`` / …) used by Pass 1.5 in
        # ``_map_products`` to disambiguate twin AGB product variants
        # tagged with the same Ciqual code. Columns 6-7 (subgroup code,
        # seasonality) are not load-bearing for the matcher.
        info_indices = [str(i) for i in (0, 1, 2, 3, 4, 5, 8)]
        df = pd.DataFrame(data[info_indices].values, columns=list(self.INFO_COLS))
        df.reset_index(drop=True, inplace=True)
        for method_tuple, (col_idx, _) in self.METHOD_TO_ADEME.items():
            short = method_tuple[2]
            df[f"ademe_{short}"] = pd.to_numeric(data.iloc[:, col_idx].values, errors="coerce")
        for col in self.INFO_COLS:
            df[col] = df[col].astype(str)
        # Drop ADEME's "(emballage corrigé)" twins. These are rows where
        # ADEME re-ran scoring on a corrected packaging chain (smaller
        # aluminium can or PVC/glass replacement) — our matrix only has
        # the original AGB activity, so the corrected reference cannot
        # be matched against any computed score. Keeping them
        # double-rows the underlying Code AGB against the legacy
        # ``version "logiciel"`` twin, inflates the outlier list with
        # uncorrectable diffs (+600 % ecotox / -47 % e_fw on Sardine
        # 26034 + Hareng family), and obscures real CF / inventory
        # bugs in the per-product diff. Filter at load so neither the
        # comparison nor the summary parquet sees them.
        name_lower = df["Nom du Produit"].astype(str).str.lower()
        keep_mask = ~name_lower.str.contains(self.EMBALLAGE_CORRIGE_TOKEN, na=False)
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            self._log.info("backtest.load.dropped_emballage_corrige_twins", n=n_dropped)
        return df.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def _normalise_ciqual(code: str) -> str:
        """Collapse equivalent CIQUAL code spellings to one key.

        Two divergences exist between AGB-side embedded codes (in the
        product name as ``[Ciqual code: 0001]``) and ADEME's reference
        column ``Code CIQUAL``:

        * Leading zeros: AGB has ``"0001"``, ADEME has ``"1"``. Strip
          leading zeros via ``int()`` round-trip on bare numerics.
        * Compound codes: AGB has ``[Ciqual code: 9341_1]``, ADEME has
          ``"9341_1"``. The previous regex captured only ``\\d+``,
          dropping the ``_1`` suffix and silently mismatching. Compound
          codes pass through this normaliser unchanged (no leading-zero
          handling — preserve the suffix intact).

        Without this, pass-1 CIQUAL matching misses ~62 products and
        the substring fallback routes them to the wrong lifecycle stage
        (typically at-processing instead of at-consumer), causing
        large per-product backtest deviations that look like CF or
        flow-mapping bugs but are actually mis-matched references.
        """
        s = (code or "").strip()
        if not s:
            return ""
        # Compound code (digits + underscore segments) — keep as-is.
        if "_" in s:
            return s
        # Bare numeric — strip leading zeros via int round-trip.
        if s.isdigit():
            return str(int(s))
        return s

    def _map_products(
        self, df: pd.DataFrame, catalog: ProductCatalog
    ) -> dict[int, tuple[str, str] | None]:
        products_df = catalog.scoreable_entries()

        ciqual_index: dict[str, list[dict]] = {}
        name_index: dict[str, list[dict]] = {}
        for row in products_df.itertuples(index=False):
            entry = {
                "key": (str(row.database), str(row.code)),
                "type": str(row.type),
                "name": str(row.name),
            }
            # Capture compound CIQUAL codes too (e.g. ``9341_1``) — the
            # ``\d+(?:_\d+)*`` form keeps trailing ``_n`` segments that
            # the old ``\d+`` form silently dropped.
            m = re.search(r"\[Ciqual code:\s*(\d+(?:_\d+)*)\]", entry["name"], re.IGNORECASE)
            if m:
                key = self._normalise_ciqual(m.group(1))
                ciqual_index.setdefault(key, []).append(entry)
            name_index.setdefault(entry["name"], []).append(entry)

        row_to_key: dict[int, tuple[str, str] | None] = {}

        if self.options.match_mode == "exact_name":
            for idx in df.index:
                lci_name = df.at[idx, "LCI Name"]
                if not lci_name or lci_name.lower() == "nan":
                    continue
                exact = name_index.get(lci_name, [])
                if len(exact) == 1:
                    row_to_key[idx] = exact[0]["key"]
                elif len(exact) > 1:
                    product_only = [e for e in exact if e["type"] == "product"]
                    if len(product_only) == 1:
                        row_to_key[idx] = product_only[0]["key"]
            for idx in df.index:
                row_to_key.setdefault(idx, None)
            self._log.info(
                "backtest.match.exact_name",
                n_total=len(df),
                n_mapped=sum(1 for v in row_to_key.values() if v is not None),
            )
            return row_to_key

        # Pass 1: CIQUAL — prefer 'product' nodes (memory: AGB consumer refs are products).
        # Try ``Code AGB`` first (unambiguous, carries compound suffixes like
        # ``9341_1``), then fall back to ``Code CIQUAL`` (bare parent integer
        # ``9341``). Without the Code-AGB-first probe, compound-suffix
        # products would all collide on the parent CIQUAL key and the matcher
        # could not disambiguate ``9341_1`` from ``9341_2``.
        #
        # Pass 1.5: when multiple product candidates share a Ciqual code,
        # disambiguate by parsing the AGB English storage label from the
        # candidate names and matching it to ADEME's ``Storage`` column via
        # ``ADEME_STORAGE_TO_AGB``. Without this, the matcher fell through
        # to LCI Name fallback in Pass 2 and silently picked the wrong twin
        # (Confit de canard 8110 → chilled "Preserved duck" instead of the
        # canned ``Ambient (long)`` variant that ADEME's reference scores).
        for idx in df.index:
            for col in ("Code AGB", "Code CIQUAL"):
                code = str(df.at[idx, col]).strip()
                if not code or code.lower() == "nan":
                    continue
                matches = ciqual_index.get(self._normalise_ciqual(code), [])
                if not matches:
                    continue
                if len(matches) == 1:
                    row_to_key[idx] = matches[0]["key"]
                    break
                product_only = [e for e in matches if e["type"] == "product"]
                if len(product_only) == 1:
                    row_to_key[idx] = product_only[0]["key"]
                    break
                # Pass 1.5: storage-disambiguation across same-Ciqual products.
                storage = str(df.at[idx, "Storage"]).strip() if "Storage" in df.columns else ""
                agb_storage = self.ADEME_STORAGE_TO_AGB.get(storage)
                if agb_storage and product_only:
                    storage_matches = [e for e in product_only if agb_storage in e["name"]]
                    if len(storage_matches) == 1:
                        row_to_key[idx] = storage_matches[0]["key"]
                        break

        # Pass 2: LCI Name exact / substring fallback.
        all_entries = list(products_df.itertuples(index=False))
        for idx in df.index:
            if idx in row_to_key:
                continue
            lci_name = df.at[idx, "LCI Name"]
            if not lci_name or lci_name.lower() == "nan":
                continue
            exact = name_index.get(lci_name, [])
            if len(exact) == 1:
                row_to_key[idx] = exact[0]["key"]
                continue
            if len(exact) > 1:
                product_only = [e for e in exact if e["type"] == "product"]
                if len(product_only) == 1:
                    row_to_key[idx] = product_only[0]["key"]
                    continue
            # Fall through to substring only when exact match is genuinely ambiguous
            cands = [e for e in all_entries if lci_name.lower() in e.name.lower()]
            if cands:
                best = min(cands, key=lambda e: len(e.name))
                row_to_key[idx] = (str(best.database), str(best.code))

        for idx in df.index:
            row_to_key.setdefault(idx, None)
        return row_to_key

    def _registered_methods(self, pkg: ScoringPackage) -> list[tuple[str, ...]]:
        return [m for m in self.METHOD_TO_ADEME if m in pkg.methods]

    def _compute_scores(
        self,
        row_to_key: dict[int, tuple[str, str] | None],
        methods: list[tuple[str, ...]],
        pkg: ScoringPackage,
        catalog: ProductCatalog,
    ) -> dict[tuple[str, str], dict[tuple[str, ...], float]]:
        unique_keys = list({k for k in row_to_key.values() if k is not None})
        if not unique_keys:
            return {}

        products: list[tuple[tuple[str, str], int]] = []
        for key in unique_keys:
            pid = catalog.product_id_for(key[0], key[1])
            if pid is not None:
                products.append((key, pid))

        use_pardiso = self.options.solver == "pardiso"
        if self.options.n_workers > 1 and len(products) > 1:
            raw = self._score_parallel_native(products, methods, pkg, use_pardiso)
        else:
            scorer = NativeLciaScorer(package=pkg, use_pardiso=use_pardiso)
            raw = scorer.score(products, methods)

        out: dict[tuple[str, str], dict[tuple[str, ...], float]] = {}
        for key, res in raw.items():
            if res.scores is not None:
                out[key] = dict(res.scores)

        n_with_scores = sum(1 for v in out.values() if v)
        if unique_keys and n_with_scores == 0:
            raise RuntimeError(
                f"All {len(raw)} scoring results are empty — nothing to write to "
                "the dashboard. Check worker logs for solver errors or empty method lists."
            )
        return out

    def _score_parallel_native(
        self,
        products: list[tuple[tuple[str, str], int]],
        methods: list[tuple[str, ...]],
        pkg: ScoringPackage,
        use_pardiso: bool,
    ) -> dict[tuple[str, str], ScoringResult]:
        if not products:
            return {}
        s = self.settings
        chunks = self._split_chunks(products, self.options.n_workers)
        payloads = [
            NativeWorkerPayload(
                store_root=s.paths.scoring_packages_root,
                content_hash=pkg.content_hash,
                products=tuple(chunk),
                methods=tuple(methods),
                use_pardiso=use_pardiso,
            )
            for chunk in chunks
        ]
        ctx = _mp.get_context("spawn")
        merged: dict[tuple[str, str], ScoringResult] = {}
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            for chunk_results in pool.map(NativeScoreWorker(), payloads):
                merged.update(chunk_results)
        self._log.info(
            "backtest.score_parallel_native.done",
            n_workers=len(chunks),
            n_products=len(products),
            n_scored=sum(1 for r in merged.values() if r.scores is not None),
        )
        return merged

    def _compare_and_save(
        self,
        df: pd.DataFrame,
        row_to_key: dict[int, tuple[str, str] | None],
        scores: dict[tuple[str, str], dict[tuple[str, ...], float]],
        methods: list[tuple[str, ...]],
        out_dir: Path,
    ) -> list[str]:
        scores_rows = []
        for idx in df.index:
            row = {col: df.at[idx, col] for col in self.INFO_COLS}
            row["mapped"] = row_to_key.get(idx) is not None
            product_key = row_to_key.get(idx)
            row["product_db"] = product_key[0] if product_key else None
            row["product_code"] = product_key[1] if product_key else None
            ps = scores.get(product_key, {}) if product_key else {}
            for method_tuple in methods:
                short = method_tuple[2]
                row[f"computed_{short}"] = ps.get(method_tuple)
                row[f"reference_{short}"] = df.at[idx, f"ademe_{short}"]
            scores_rows.append(row)
        if scores_rows:
            scores_df = pd.DataFrame(scores_rows)
        else:
            all_cols = (
                list(self.INFO_COLS)
                + ["mapped", "product_db", "product_code"]
                + [f"computed_{m[2]}" for m in methods]
                + [f"reference_{m[2]}" for m in methods]
            )
            scores_df = pd.DataFrame(columns=all_cols)

        diff_abs: dict[str, pd.Series] = {col: scores_df[col] for col in self.INFO_COLS}
        diff_pct: dict[str, pd.Series] = {col: scores_df[col] for col in self.INFO_COLS}
        for method_tuple in methods:
            short = method_tuple[2]
            computed = pd.to_numeric(scores_df[f"computed_{short}"], errors="coerce")
            reference = pd.to_numeric(scores_df[f"reference_{short}"], errors="coerce")
            diff_abs[short] = computed - reference
            diff_pct[short] = ((computed - reference) / reference * 100).round(4)

        floor = NearZeroFloor.compute(
            scores_df, [m[2] for m in methods], factor=self.NEAR_ZERO_FACTOR
        )
        floored_counts = floor.apply(scores_df, diff_abs, diff_pct)
        if any(floored_counts.values()):
            self._log.info(
                "backtest.near_zero_floor.applied",
                factor=self.NEAR_ZERO_FACTOR,
                counts={k: v for k, v in floored_counts.items() if v > 0},
            )

        diff_abs_df = pd.DataFrame(diff_abs)
        diff_pct_df = pd.DataFrame(diff_pct)

        summary_rows = []
        for method_tuple in methods:
            short = method_tuple[2]
            computed = pd.to_numeric(scores_df[f"computed_{short}"], errors="coerce")
            reference = pd.to_numeric(scores_df[f"reference_{short}"], errors="coerce")
            valid = computed.notna() & reference.notna() & (reference != 0)
            n = int(valid.sum())
            if n == 0:
                summary_rows.append(
                    {
                        "method": short,
                        "n_compared": 0,
                        "n_near_zero_floored": floored_counts.get(short, 0),
                    }
                )
                continue
            pct = (computed[valid] - reference[valid]) / reference[valid] * 100
            summary_rows.append(
                {
                    "method": short,
                    "n_compared": n,
                    "mean_diff_pct": round(float(pct.mean()), 4),
                    "median_diff_pct": round(float(pct.median()), 4),
                    "std_diff_pct": round(float(pct.std()), 4),
                    "within_1pct": int((pct.abs() <= 1.0).sum()),
                    "within_5pct": int((pct.abs() <= 5.0).sum()),
                    "outliers_gt5pct": int((pct.abs() > 5.0).sum()),
                    "max_abs_diff_pct": round(float(pct.abs().max()), 4),
                    "n_near_zero_floored": floored_counts.get(short, 0),
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        artifacts: list[str] = []
        for fname, df_, sort_by in (
            ("scores.parquet", scores_df, ["Code AGB"]),
            ("diff_abs.parquet", diff_abs_df, ["Code AGB"]),
            ("diff_pct.parquet", diff_pct_df, ["Code AGB"]),
            ("summary.parquet", summary_df, ["method"]),
        ):
            path = out_dir / fname
            df_.sort_values(by=sort_by).to_parquet(path, index=False, compression=None)
            artifacts.append(str(path.name))

        dashboard_csv = out_dir.parent / self.DASHBOARD_PASS1_CSV
        BacktestPass1Emitter(out_path=dashboard_csv).write(
            scores_df.sort_values(by=["Code AGB"]),
            diff_pct_df.sort_values(by=["Code AGB"]),
        )
        artifacts.append(str(dashboard_csv.name))
        return artifacts
