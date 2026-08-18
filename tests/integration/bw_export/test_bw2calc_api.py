"""Pin the exact bw_processing + bw2calc datapackage API.

This test locks down, by experiment, the contract that all bw-export code
depends on:

* Matrix names: ``technosphere_matrix``, ``biosphere_matrix``,
  ``characterization_matrix``.
* ``indices_array`` uses ``bw_processing.INDICES_DTYPE`` (``row``/``col``).
* Technosphere/biosphere values are stored PRE-SIGNED with ``flip_array``
  all-False, so the assembled matrix equals A exactly (production positive
  on the diagonal, consumption negative).
* The characterization matrix is a diagonal keyed on the biosphere flow id,
  i.e. indices pairs ``(flow_id, flow_id)``.
* ``bw2calc.LCA`` demand is keyed by the PRODUCT (row) id, not the
  activity (column) id.
* In-memory ``create_datapackage()`` objects are passed directly to
  ``data_objs`` (no serialization/zip/dir round-trip needed).

Known correct answer for this system: inventory score for product 101 = 20.0.

    A = [[1, -0.5], [0, 1]]   rows = (101, 102), cols = (201, 202)
    B = [[2, 3]]              row = 301, cols = (201, 202)
    CF(301) = 10.0
    demand = 1 unit of product 101
    supply = A^-1 @ e0 = [1, 0]
    inventory = B @ supply = [2]
    score = 10 * 2 = 20.0
"""

import numpy as np
import pytest

bc = pytest.importorskip("bw2calc", reason="needs the [bw] extra")
bwp = pytest.importorskip("bw_processing", reason="needs the [bw] extra")


def _indices(pairs):
    """Build a structured indices array with ``bw_processing.INDICES_DTYPE``."""
    arr = np.empty(len(pairs), dtype=bwp.INDICES_DTYPE)
    for i, (row, col) in enumerate(pairs):
        arr[i] = (row, col)
    return arr


def test_bw2calc_reproduces_known_score():
    inventory = bwp.create_datapackage()
    inventory.add_persistent_vector(
        matrix="technosphere_matrix",
        name="technosphere",
        data_array=np.array([1.0, -0.5, 1.0]),
        indices_array=_indices([(101, 201), (101, 202), (102, 202)]),
        flip_array=np.array([False, False, False]),
    )
    inventory.add_persistent_vector(
        matrix="biosphere_matrix",
        name="biosphere",
        data_array=np.array([2.0, 3.0]),
        indices_array=_indices([(301, 201), (301, 202)]),
        flip_array=np.array([False, False]),
    )

    method = bwp.create_datapackage()
    method.add_persistent_vector(
        matrix="characterization_matrix",
        name="characterization",
        data_array=np.array([10.0]),
        indices_array=_indices([(301, 301)]),
    )

    # Demand is keyed by the PRODUCT (row) id 101, NOT the activity id 201.
    lca = bc.LCA({101: 1.0}, data_objs=[inventory, method])
    lca.lci()
    lca.lcia()

    assert abs(lca.score - 20.0) < 1e-9
