"""``CorrectionEmbedder`` — fold per-activity corrections into the biosphere.

The bundle adds AWARE regional + consumption corrections as a per-method
row of shape ``(1, n_activities)`` applied to ``supply`` (see
``native_scorer``). A standard ``Q @ B @ supply`` cannot express that —
a per-flow CF is location-blind. We embed each correction as a synthetic
biosphere flow whose B-row equals the correction vector and whose CF is
``1.0`` in only that method's characterization. Then stock bw2calc
reproduces ``correction @ supply`` exactly.

Synthetic flow ids are allocated as small positive integers with an
explicit collision check against every existing id (real ids are 63-bit
SHA-256 hashes, so a small integer collision is astronomically unlikely
but we assert rather than assume).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse as sp

from scoring.scoring_package import ScoringPackage


@dataclass(frozen=True)
class EmbeddedInventory:
    """Augmented matrices ready for datapackage emission.

    ``biosphere`` is the original B stacked with one extra row per
    corrected method. ``method_cfs`` maps each method tuple to a
    ``{flow_id: cf}`` dict (real CFs from Q plus the synthetic CF=1.0).
    """

    technosphere: sp.csr_matrix
    technosphere_row_id_to_idx: dict[int, int]
    technosphere_col_id_to_idx: dict[int, int]
    biosphere: sp.csr_matrix
    biosphere_row_id_to_idx: dict[int, int]
    method_cfs: dict[tuple[str, ...], dict[int, float]]
    synthetic_flow_ids: dict[tuple[str, ...], int] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectionEmbedder:
    """Stateless. ``embed(package)`` returns an :class:`EmbeddedInventory`."""

    def embed(self, package: ScoringPackage) -> EmbeddedInventory:
        b = package.biosphere.matrix.tocsr()
        bio_row_map = dict(package.biosphere.row_id_to_idx)
        n_cols = b.shape[1]

        # CF dicts seeded from each method's Q vector (1, n_flows).
        idx_to_flow = {v: k for k, v in bio_row_map.items()}
        method_cfs: dict[tuple[str, ...], dict[int, float]] = {}
        for method, q in package.methods.items():
            q_coo = q.tocoo()
            cfs: dict[int, float] = {}
            for col, val in zip(q_coo.col, q_coo.data, strict=True):
                cfs[int(idx_to_flow[int(col)])] = float(val)
            method_cfs[method] = cfs

        used_ids = (
            set(bio_row_map)
            | set(package.technosphere.row_id_to_idx)
            | set(package.technosphere.col_id_to_idx)
        )
        synthetic_flow_ids: dict[tuple[str, ...], int] = {}
        new_rows: list[np.ndarray] = []
        next_id = 1
        for method in sorted(package.corrections, key=lambda m: "\x1e".join(m)):
            correction = package.corrections[method].tocsr()
            if correction.nnz == 0:
                continue
            while next_id in used_ids:
                next_id += 1
            syn_id = next_id
            used_ids.add(syn_id)
            next_id += 1
            synthetic_flow_ids[method] = syn_id
            bio_row_map[syn_id] = b.shape[0] + len(new_rows)
            new_rows.append(correction.toarray().reshape(1, n_cols))
            method_cfs.setdefault(method, {})[syn_id] = 1.0

        if new_rows:
            b = sp.vstack([b, sp.csr_matrix(np.vstack(new_rows))]).tocsr()

        return EmbeddedInventory(
            technosphere=package.technosphere.matrix.tocsr(),
            technosphere_row_id_to_idx=dict(package.technosphere.row_id_to_idx),
            technosphere_col_id_to_idx=dict(package.technosphere.col_id_to_idx),
            biosphere=b,
            biosphere_row_id_to_idx=bio_row_map,
            method_cfs=method_cfs,
            synthetic_flow_ids=synthetic_flow_ids,
        )
