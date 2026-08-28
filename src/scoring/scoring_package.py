"""``ScoringPackage`` — content-addressable, parquet/numpy-native handoff.

Phase 4 of the SQLite refactor. Replaces the ``bw_processing`` zip
format as the on-disk artifact that scoring workers consume. The zip
format works fine, but it carries baggage: a JSON manifest, a fixed
file layout, and a ``ZipFileSystem`` indirection that costs ~50 ms per
worker on cold load. With 6 workers that's 300 ms of pure loading
overhead per backtest, every backtest.

This format is a directory keyed by content hash::

    cache/scoring_packages/<hash>/
        technosphere.csr.npz   # row, col, data, shape — scipy CSR
        biosphere.csr.npz
        characterization/
            <method-tuple-slug>.csr.npz
        ids.parquet            # row_id_to_idx, col_id_to_idx, ...

The hash is the SHA-256 of the input ``ExchangeFrame`` parquet bytes.
That gives us idempotency: two pipelines that produce the same frame
share one cached package directory. Scoring loads the .npz files
directly into ``scipy.sparse`` matrices — no zip seek, no JSON parse,
no bw2calc indirection."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import sparse as sp

from scoring.aware_consumption_correction import AwareConsumptionCorrectionBuilder
from scoring.exchange_frame import ExchangeFrame
from scoring.matrix_builder import (
    BiosphereBuilder,
    BuiltMatrix,
    CharacterizationBuilder,
    TechnosphereBuilder,
)
from scoring.method_slug import MethodSlug
from scoring.regional_correction import RegionalCorrectionBuilder


@dataclass(frozen=True)
class ScoringPackage:
    """Materialised matrices + id maps. In-memory representation.

    ``corrections`` holds an optional per-method ``(1, n_activities)``
    sparse row whose ``correction @ supply`` is added to the
    characterisation score. Methods without a regional split simply
    omit a key (no correction). See ``RegionalCorrectionBuilder``.
    """

    technosphere: BuiltMatrix
    biosphere: BuiltMatrix
    methods: dict[tuple[str, ...], sp.csr_matrix]
    corrections: dict[tuple[str, ...], sp.csr_matrix] = field(default_factory=dict)
    content_hash: str = ""

    @property
    def n_products(self) -> int:
        return self.technosphere.shape[0]

    @property
    def n_activities(self) -> int:
        return self.technosphere.shape[1]

    @property
    def n_biosphere_flows(self) -> int:
        return self.biosphere.shape[0]


@dataclass(frozen=True)
class ScoringPackageBuilder:
    """One-shot builder. Takes an ExchangeFrame + per-method CF tables,
    produces a ``ScoringPackage`` in memory.

    Optional ``regional_cfs`` + ``col_id_to_location`` enable per-activity
    correction rows (currently used for the water-use AWARE regional
    split). Methods absent from ``regional_cfs`` simply don't get a
    correction; the standard ``Q @ B @ supply`` path stays intact.

    Optional ``biosphere_catalog`` + ``aware_regional_cf_by_location``
    enable the AWARE net-consumption correction for the water-use
    method (see :class:`AwareConsumptionCorrectionBuilder`). When both
    are supplied, the per-activity correction is summed into the
    water-use correction row.
    """

    # The method tuple the AWARE net-consumption correction applies to.
    # Hardcoded here because it's the only method whose inventory is
    # consumption-asymmetric (resource vs return) rather than per-flow
    # additive. Other methods don't need a parallel mechanism.
    WATER_USE_METHOD: ClassVar[tuple[str, str, str, str]] = (
        "ecoinvent-3.9.1",
        "EF v3.1",
        "water use",
        "user deprivation potential (deprivation-weighted water consumption)",
    )

    technosphere_builder: TechnosphereBuilder = field(default_factory=TechnosphereBuilder)
    biosphere_builder: BiosphereBuilder = field(default_factory=BiosphereBuilder)
    cf_builder: CharacterizationBuilder = field(default_factory=CharacterizationBuilder)
    correction_builder: RegionalCorrectionBuilder = field(default_factory=RegionalCorrectionBuilder)
    aware_consumption_builder: AwareConsumptionCorrectionBuilder = field(
        default_factory=AwareConsumptionCorrectionBuilder
    )

    def build(
        self,
        frame: ExchangeFrame,
        method_cfs: dict[tuple[str, ...], pd.DataFrame],
        regional_cfs: dict[tuple[str, ...], pd.DataFrame] | None = None,
        col_id_to_location: dict[int, str] | None = None,
        biosphere_catalog: pd.DataFrame | None = None,
        aware_regional_cf_by_location: dict[str, float] | None = None,
    ) -> ScoringPackage:
        a = self.technosphere_builder.build(frame)
        # Reuse A's column ordering so ``B @ supply`` is well-defined
        # (``supply = A^{-1} demand`` is indexed by A's columns).
        b = self.biosphere_builder.build(frame, col_id_to_idx=a.col_id_to_idx)
        methods = {method: self.cf_builder.build(cf_df, b) for method, cf_df in method_cfs.items()}

        corrections: dict[tuple[str, ...], sp.csr_matrix] = {}
        if regional_cfs and col_id_to_location:
            for method, regional_df in regional_cfs.items():
                if regional_df.empty:
                    continue
                global_cf_df = method_cfs.get(method, pd.DataFrame(columns=["flow_id", "cf"]))
                row = self.correction_builder.build(
                    global_cf_df=global_cf_df,
                    regional_cf_df=regional_df,
                    biosphere=b,
                    technosphere=a,
                    col_id_to_location=col_id_to_location,
                )
                if row.nnz > 0:
                    corrections[method] = row

        # AWARE net-consumption correction (water-use only). Sums into
        # the existing water-use correction row (or creates one if the
        # regional pass didn't emit anything).
        if biosphere_catalog is not None and aware_regional_cf_by_location and col_id_to_location:
            aware_row = self.aware_consumption_builder.build(
                biosphere=b,
                technosphere=a,
                biosphere_catalog=biosphere_catalog,
                regional_cf_by_location=aware_regional_cf_by_location,
                col_id_to_location=col_id_to_location,
            )
            if aware_row.nnz > 0:
                key = self.WATER_USE_METHOD
                if key in corrections:
                    corrections[key] = (corrections[key] + aware_row).tocsr()
                else:
                    corrections[key] = aware_row

        return ScoringPackage(
            technosphere=a,
            biosphere=b,
            methods=methods,
            corrections=corrections,
            content_hash=self._hash_inputs(
                frame,
                method_cfs,
                regional_cfs,
                col_id_to_location,
                biosphere_catalog,
                aware_regional_cf_by_location,
            ),
        )

    @classmethod
    def _hash_inputs(
        cls,
        frame: ExchangeFrame,
        method_cfs: dict[tuple[str, ...], pd.DataFrame],
        regional_cfs: dict[tuple[str, ...], pd.DataFrame] | None = None,
        col_id_to_location: dict[int, str] | None = None,
        biosphere_catalog: pd.DataFrame | None = None,
        aware_regional_cf_by_location: dict[str, float] | None = None,
    ) -> str:
        """SHA-256 of the frame parquet bytes + the method CFs + regional CFs +
        column-location lookup.

        Including method CFs is critical: the on-disk package has one
        CSR per method, so the cache key MUST invalidate when CFs
        change. Otherwise ``ScoringPackageStore.write`` short-circuits
        on an existing directory and silently preserves stale CFs —
        the failure mode that swallowed the §A.1 Barite filter and
        §C.1 water augmenter on 2026-05-04. The same invalidation
        rule applies to ``regional_cfs`` AND ``col_id_to_location``:
        the regional correction row is a function of both, so a change
        to *either* input must produce a new content hash or the
        ``corrections`` directory inside the cache stays stale.
        """
        h = hashlib.sha256()
        h.update(cls._frame_bytes(frame))
        for method in sorted(method_cfs.keys()):
            h.update(b"\x00")
            h.update("\x1e".join(method).encode("utf-8"))
            h.update(b"\x00")
            h.update(cls._cf_bytes(method_cfs[method]))
        if regional_cfs:
            for method in sorted(regional_cfs.keys()):
                h.update(b"\x01")
                h.update("\x1e".join(method).encode("utf-8"))
                h.update(b"\x01")
                h.update(cls._regional_cf_bytes(regional_cfs[method]))
        if col_id_to_location:
            h.update(b"\x02")
            # Hash deterministically — sort by col_id so two callers
            # that produced the same mapping in different insertion
            # orders converge on the same hash.
            for col_id in sorted(col_id_to_location.keys()):
                h.update(str(int(col_id)).encode("utf-8"))
                h.update(b"=")
                h.update(str(col_id_to_location[col_id]).encode("utf-8"))
                h.update(b"\x1f")
        if aware_regional_cf_by_location:
            h.update(b"\x03")
            for loc in sorted(aware_regional_cf_by_location.keys()):
                h.update(str(loc).encode("utf-8"))
                h.update(b"=")
                h.update(format(float(aware_regional_cf_by_location[loc]), ".10g").encode("utf-8"))
                h.update(b"\x1f")
        if biosphere_catalog is not None and not biosphere_catalog.empty:
            # We only need the catalog's water-row schema to be stable;
            # hashing the (database, code) of every row keeps the hash
            # responsive to any catalog change without re-parquet-ing.
            h.update(b"\x04")
            cols = [c for c in ("database", "code") if c in biosphere_catalog.columns]
            if cols:
                snap = biosphere_catalog[cols].astype(str).sort_values(cols).reset_index(drop=True)
                buf = io.BytesIO()
                snap.to_parquet(buf, engine="pyarrow", compression=None)
                h.update(buf.getvalue())
        # AWARE consumption-builder tuning constants — class variables
        # the per-activity correction depends on. We MUST hash them so a
        # threshold edit invalidates the cached package.
        h.update(b"\x05")
        cfg = AwareConsumptionCorrectionBuilder
        h.update(format(cfg.ASYMMETRY_THRESHOLD, ".10g").encode("utf-8"))
        h.update(b"|")
        h.update(format(cfg.MIN_NET_M3, ".10g").encode("utf-8"))
        h.update(b"|")
        h.update(format(cfg.FALLBACK_CF, ".10g").encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _regional_cf_bytes(df: pd.DataFrame) -> bytes:
        """Deterministic bytes for a per-method regional CF table."""
        if df.empty:
            return b""
        cols = [c for c in ("flow_id", "location", "cf") if c in df.columns]
        sorted_df = df[cols].sort_values(cols).reset_index(drop=True)
        buf = io.BytesIO()
        sorted_df.to_parquet(buf, engine="pyarrow", compression=None)
        return buf.getvalue()

    @staticmethod
    def _frame_bytes(frame: ExchangeFrame) -> bytes:
        df = frame.df.sort_values(by=["output_id", "input_id", "edge_type"]).reset_index(drop=True)
        # We use parquet bytes (not CSV) because parquet's binary
        # representation is deterministic for fixed schema + row order
        # and avoids float/string formatting drift. ``BytesIO`` keeps
        # the operation in-memory — no temp file.
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression=None)
        return buf.getvalue()

    @staticmethod
    def _cf_bytes(cf_df: pd.DataFrame) -> bytes:
        """Deterministic byte representation of one method's CF table.

        Sorts on (flow_id, cf) so input row order doesn't shift the
        hash — only the CF *content* matters for cache invalidation.
        """
        if cf_df.empty:
            return b""
        sorted_df = cf_df[["flow_id", "cf"]].sort_values(["flow_id", "cf"]).reset_index(drop=True)
        buf = io.BytesIO()
        sorted_df.to_parquet(buf, engine="pyarrow", compression=None)
        return buf.getvalue()


@dataclass(frozen=True)
class ScoringPackageStore:
    """Directory-backed cache for ``ScoringPackage``.

    ``write(pkg)`` lays out the per-package directory, ``read(hash)``
    rehydrates an in-memory ScoringPackage. The store is content-
    addressable: the same package can be written by multiple producers
    and they all collide on the same directory deterministically.

    Crash safety: writes go to ``<dir>.partial/`` first and are renamed
    into place atomically once every file has landed. Readers never see
    a half-finished directory."""

    root: Path

    #: What-if parameter overrides fork a new content-hashed package per
    #: distinct value; without eviction, iterating what-ifs accumulates
    #: multi-GB package dirs without bound. Keep the most-recently-used N.
    MAX_PACKAGES: ClassVar[int] = 5

    def path_for(self, content_hash: str) -> Path:
        return self.root / content_hash

    def write(self, package: ScoringPackage) -> Path:
        target = self.path_for(package.content_hash)
        if target.exists():
            # Refresh mtime so the LRU eviction treats a re-used package
            # (e.g. clearing overrides back to baseline) as recent.
            target.touch()
            return target
        partial = self.root / f"{package.content_hash}.partial"
        partial.mkdir(parents=True, exist_ok=True)

        self._write_csr(partial / "technosphere.csr.npz", package.technosphere.matrix)
        self._write_csr(partial / "biosphere.csr.npz", package.biosphere.matrix)
        methods_dir = partial / "characterization"
        methods_dir.mkdir(exist_ok=True)
        for method, q in package.methods.items():
            self._write_csr(methods_dir / f"{self._slug(method)}.csr.npz", q)

        if package.corrections:
            corrections_dir = partial / "corrections"
            corrections_dir.mkdir(exist_ok=True)
            for method, row in package.corrections.items():
                self._write_csr(corrections_dir / f"{self._slug(method)}.csr.npz", row)

        ids = {
            "technosphere_row_id_to_idx": package.technosphere.row_id_to_idx,
            "technosphere_col_id_to_idx": package.technosphere.col_id_to_idx,
            "biosphere_row_id_to_idx": package.biosphere.row_id_to_idx,
            "method_keys": [list(m) for m in package.methods],
            "correction_keys": [list(m) for m in package.corrections],
        }
        (partial / "ids.json").write_text(json.dumps(ids, sort_keys=True, default=str))

        partial.rename(target)
        self._evict_lru(keep=target)
        return target

    def _evict_lru(self, keep: Path) -> None:
        """Prune package dirs beyond ``MAX_PACKAGES``, oldest-mtime first.

        ``keep`` (the just-written package) is always preserved. Leftover
        ``*.partial`` dirs from crashed writes are removed unconditionally.
        """
        import shutil

        if not self.root.exists():
            return
        for stale_partial in self.root.glob("*.partial"):
            shutil.rmtree(stale_partial, ignore_errors=True)
        packages = sorted(
            (p for p in self.root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for victim in packages[self.MAX_PACKAGES :]:
            if victim == keep:
                continue
            shutil.rmtree(victim, ignore_errors=True)

    def read(self, content_hash: str) -> ScoringPackage:
        path = self.path_for(content_hash)
        if not path.exists():
            raise FileNotFoundError(f"ScoringPackage not in store: {path}")
        ids = json.loads((path / "ids.json").read_text())
        a = self._load_built(
            path / "technosphere.csr.npz",
            ids["technosphere_row_id_to_idx"],
            ids["technosphere_col_id_to_idx"],
        )
        b = self._load_built(
            path / "biosphere.csr.npz",
            ids["biosphere_row_id_to_idx"],
            ids["technosphere_col_id_to_idx"],
        )
        methods: dict[tuple[str, ...], sp.csr_matrix] = {}
        method_dir = path / "characterization"
        for method_key in ids["method_keys"]:
            tup = tuple(method_key)
            methods[tup] = self._read_csr(method_dir / f"{self._slug(tup)}.csr.npz")
        corrections: dict[tuple[str, ...], sp.csr_matrix] = {}
        # ``correction_keys`` is forward-compatible: older packages
        # written before the regional split land here without the key
        # and we simply hand back an empty corrections dict.
        for method_key in ids.get("correction_keys", []):
            tup = tuple(method_key)
            corrections[tup] = self._read_csr(path / "corrections" / f"{self._slug(tup)}.csr.npz")
        return ScoringPackage(
            technosphere=a,
            biosphere=b,
            methods=methods,
            corrections=corrections,
            content_hash=content_hash,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _write_csr(path: Path, csr: sp.csr_matrix) -> None:
        np.savez(
            path,
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            shape=np.asarray(csr.shape, dtype="int64"),
        )

    @staticmethod
    def _read_csr(path: Path) -> sp.csr_matrix:
        with np.load(path) as f:
            return sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))

    @classmethod
    def _load_built(
        cls,
        path: Path,
        row_id_to_idx: dict,
        col_id_to_idx: dict,
    ) -> BuiltMatrix:
        return BuiltMatrix(
            matrix=cls._read_csr(path),
            row_id_to_idx={int(k): int(v) for k, v in row_id_to_idx.items()},
            col_id_to_idx={int(k): int(v) for k, v in col_id_to_idx.items()},
        )

    @staticmethod
    def _slug(method: tuple[str, ...]) -> str:
        # Delegate to the shared encoder so the on-disk slug is identical
        # to the one ``MethodCfRegistryBuilder`` writes.
        return MethodSlug.encode(method)
