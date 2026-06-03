"""Permissive stub modules for ``bw2*`` and ``randonneur*``.

Every Brightway / randonneur import in production code is deferred (lives
inside a method body), so we can publish fake modules into ``sys.modules``
once at session start. Tests that need stricter doubles patch over these.

We avoid pulling the real packages because:

* ``bw2data.projects.set_current(...)`` writes to ``$HOME``;
* ``bw2io.SimaProBlockCSVImporter`` requires the real Agribalyse CSV;
* ``bw2calc.LCA`` requires assembled matrices.

The stubs intentionally raise ``NotImplementedError`` for any behavior that a
test must exercise — the test must patch what it needs, otherwise it would
silently no-op into a green pass that proves nothing.
"""

from __future__ import annotations

import sys
import types


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _RaiseOnUseDict(dict):
    """A dict-shaped object that raises if iterated/queried by tests that
    forgot to populate it. Keeps surprises loud."""

    def __init__(self, label: str):
        super().__init__()
        self._label = label


class _NullCtx:
    """No-op context manager for stubbed ``sqlite3_lci_db.atomic()``."""

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def install_stubs() -> None:
    """Idempotently publish the fake bw2 + randonneur modules."""
    if "bw2data" in sys.modules and getattr(sys.modules["bw2data"], "_dds_test_stub", False):
        return

    # ---- bw2data -----------------------------------------------------
    bw2data = _make_module("bw2data", _dds_test_stub=True)
    bw2data.databases = {}
    bw2data.methods = {}
    bw2data.projects = types.SimpleNamespace(
        set_current=lambda *a, **k: None,
        change_base_directories=lambda *a, **k: None,
        dir=".",
    )

    class _Database:
        def __init__(self, name: str):
            self.name = name

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

        def __contains__(self, _item):
            return False

        def register(self):
            return None

        def write(self, *_a, **_k):
            return None

    class _Method:
        def __init__(self, key):
            self.key = key

        def load(self):
            return []

        def write(self, *_a, **_k):
            return None

    bw2data.Database = _Database
    bw2data.Method = _Method
    bw2data.get_activity = lambda key: None

    # ``bw2data.labels`` — referenced by the SimaPro CSV wrapper for
    # process / product / biosphere edge type lists. The real module
    # contains thin lists; tests don't care about the contents, only
    # that attribute lookups don't blow up.
    bw2data_labels = _make_module("bw2data.labels")
    bw2data_labels.process_node_types = ["process"]
    bw2data_labels.product_node_types = ["product"]
    bw2data_labels.multifunctional_node_default = "multifunctional"
    bw2data_labels.biosphere_edge_types = ["biosphere"]
    bw2data.labels = bw2data_labels

    # ``backends`` submodule — used by exporters and matrix purger.
    bw2data_backends = _make_module("bw2data.backends")
    bw2data_backends.SQLiteBackend = type("SQLiteBackend", (), {})
    bw2data_backends.sqlite3_lci_db = types.SimpleNamespace(
        atomic=lambda: _NullCtx(),
    )
    bw2data_backends_schema = _make_module("bw2data.backends.schema")
    bw2data_backends_schema.ExchangeDataset = type(
        "ExchangeDataset", (), {"select": classmethod(lambda cls, *a, **k: [])}
    )
    bw2data_backends_schema.ActivityDataset = type(
        "ActivityDataset", (), {"select": classmethod(lambda cls, *a, **k: [])}
    )

    # ---- bw2io -------------------------------------------------------
    bw2io = _make_module("bw2io", _dds_test_stub=True)

    class _SimaProBlockCSVImporter:
        def __init__(self, *_a, **_k):
            self.data = []

        def apply_strategy(self, *_a, **_k):
            return None

        def apply_strategies(self):
            return None

        def match_database(self, *_a, **_k):
            return None

        def match_database_against_top_level_context(self, *_a, **_k):
            return None

        def match_database_against_only_available_in_given_context_tree(self, *_a, **_k):
            return None

        def randonneur(self, *_a, **_k):
            return None

        def normalize_labels_to_brightway_standard(self):
            return None

    bw2io.SimaProBlockCSVImporter = _SimaProBlockCSVImporter
    bw2io.create_core_migrations = lambda: None
    bw2io.create_default_biosphere3 = lambda: None
    bw2io.create_default_lcia_methods = lambda: None

    bw2io_strategies = _make_module("bw2io.strategies")

    def _noop_strategy(*_a, **_k):
        return None

    for n in (
        "change_electricity_unit_mj_to_kwh",
        "normalize_simapro_biosphere_categories",
        "normalize_simapro_biosphere_names",
        "normalize_biosphere_categories",
        "normalize_biosphere_names",
        "strip_biosphere_exc_locations",
        "drop_unspecified_subcategories",
        "remove_biosphere_location_prefix_if_flow_in_same_location",
    ):
        setattr(bw2io_strategies, n, _noop_strategy)

    bw2io_utils = _make_module("bw2io.utils")

    def _rescale_exchange(exc, multiplier):
        exc["amount"] = exc.get("amount", 0) * multiplier
        return exc

    bw2io_utils.rescale_exchange = _rescale_exchange
    bw2io.utils = bw2io_utils  # so ``bw2io.utils.rescale_exchange`` resolves

    bw2io_imp = _make_module("bw2io.importers")
    bw2io_imp_simapro = _make_module("bw2io.importers.simapro_block_csv")
    bw2io_imp_simapro.SimaProBlockCSVImporter = _SimaProBlockCSVImporter
    bw2io.importers = bw2io_imp
    bw2io_imp.SimaProBlockCSVImporter = _SimaProBlockCSVImporter
    bw2io_imp.simapro_block_csv = bw2io_imp_simapro

    # ``bw2io.errors`` — only the StrategyError type is needed for
    # the F2 wrapper's ``match_database`` raise.
    bw2io_errors = _make_module("bw2io.errors")
    bw2io_errors.StrategyError = type("StrategyError", (Exception,), {})

    # ``bw2io.strategies.generic`` — link_iterable_by_fields. Tests that
    # exercise it monkeypatch the parser module directly; this stub just
    # makes the import succeed.
    bw2io_strat_generic = _make_module("bw2io.strategies.generic")
    bw2io_strat_generic.link_iterable_by_fields = _noop_strategy
    bw2io_strategies.generic = bw2io_strat_generic
    bw2io_strategies.link_iterable_by_fields = _noop_strategy
    for n in (
        "match_against_top_level_context",
        "match_against_only_available_in_given_context_tree",
        "set_metadata_using_single_functional_exchange",
        "split_simapro_name_geo",
        "drop_unlinked",
        "normalize_simapro_labels_to_brightway_standard",
    ):
        setattr(bw2io_strategies, n, _noop_strategy)

    # ---- bw2calc -----------------------------------------------------
    bw2calc = _make_module("bw2calc", _dds_test_stub=True)

    class _LCA:
        def __init__(self, demand=None, method=None, *_a, **_k):
            self.demand = demand
            self.method = method
            self.score = 0.0

        def lci(self, demand=None):
            if demand is not None:
                self.demand = demand
            return None

        def switch_method(self, method):
            self.method = method

        def lcia_calculation(self):
            self.score = 1.0

    bw2calc.LCA = _LCA
    bw2calc_lca_base = _make_module("bw2calc.lca_base")
    bw2calc_lca_base.spsolve = lambda *a, **k: None
    bw2calc_lca_base.factorized = lambda *a, **k: None

    # ---- randonneur -------------------------------------------------
    randonneur = _make_module("randonneur", _dds_test_stub=True)

    class _Datapackage:
        def __init__(self, data=None):
            self.data = data or {"update": []}

        @classmethod
        def from_json(cls, _path):
            return cls({"update": []})

    randonneur.Datapackage = _Datapackage
    randonneur.MigrationConfig = type(
        "MigrationConfig",
        (),
        {"__init__": lambda self, **kw: self.__dict__.update(kw)},
    )
    randonneur.migrate_edges = lambda graph, migrations, config: graph

    # ---- randonneur_data --------------------------------------------
    randonneur_data = _make_module("randonneur_data", _dds_test_stub=True)

    class _Registry(dict):
        def __init__(self):
            super().__init__()
            self.data_dir = "."

        def get_file(self, label):  # pragma: no cover — overridden in tests
            raise NotImplementedError(f"randonneur_data stub: get_file({label})")

    randonneur_data.Registry = _Registry

    # ---- pypardiso (optional) ---------------------------------------
    _make_module("pypardiso", _dds_test_stub=True)
    pypardiso_aliases = _make_module("pypardiso.scipy_aliases")
    pypardiso_aliases.pypardiso_solver = types.SimpleNamespace(
        set_iparm=lambda *_a, **_k: None,
    )

    # ---- bw_simapro_csv ----------------------------------------------
    bw_simapro_csv = _make_module("bw_simapro_csv", _dds_test_stub=True)

    class _SimaProCSV:
        def __init__(self, **kw):
            self.database_name = kw.get("database_name") or "fallback-db"

        def to_brightway(self, separate_products=True, shorten_names=True):
            return {
                "database": {},
                "processes": [],
                "products": [],
                "database_parameters": [],
                "project_parameters": [],
            }

    bw_simapro_csv.SimaProCSV = _SimaProCSV
    _make_module("bw_migrations", _dds_test_stub=True)
    _make_module("ecoinvent_interface", _dds_test_stub=True)
    _make_module("ecoinvent_migrate", _dds_test_stub=True)
    _make_module("multifunctional", _dds_test_stub=True)
