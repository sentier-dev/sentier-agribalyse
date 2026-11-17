import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from config import Settings
from ef.cf_registry import MethodCfRegistryLoader
from pipelines.link_all import LinkAllPipeline
from scoring import CharacterizationAuditor
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix

SRC = Path(__file__).resolve().parents[2] / "src"


def run_audit(biosphere, method_cfs, catalog):
    return CharacterizationAuditor().audit(biosphere, method_cfs, catalog)


def fixture():
    catalog = pd.DataFrame(
        [
            {"database": "bio", "code": "ch4", "name": "Methane", "categories": ("air",)},
            {
                "database": "bio",
                "code": "co2",
                "name": "Carbon dioxide, biogenic",
                "categories": ("air",),
            },
            {"database": "bio", "code": "nh3", "name": "Ammonia", "categories": ("air",)},
        ]
    )
    ids = [ExchangeFrameBuilder.flow_id_for(("bio", c)) for c in catalog.code]
    b = BuiltMatrix(
        matrix=sparse.csr_matrix([[1.0], [1.0], [2.0]]),
        row_id_to_idx=dict(zip(ids, range(3), strict=True)),
        col_id_to_idx={1: 0},
    )
    cfs = {("test", "climate change"): pd.DataFrame({"flow_id": ids[:2], "cf": [27.0, 0.0]})}
    return b, cfs, catalog


def test_explicit_zero_is_characterized_and_ammonia_not_required():
    b, cfs, catalog = fixture()
    audit = run_audit(b, cfs, catalog)
    assert audit["status"] == "passed"
    assert audit["methods"][0]["explicit_zero_flows"] == 1
    assert audit["methods"][0]["uncharacterized_flows"] == 1


def test_unqualified_methane_missing_cf_fails_without_guessing():
    b, cfs, catalog = fixture()
    cfs[("test", "climate change")] = cfs[("test", "climate change")].iloc[1:]
    audit = run_audit(b, cfs, catalog)
    assert audit["status"] == "failed"
    assert audit["methods"][0]["missing_core_gases"][0]["name"] == "Methane"


def test_zero_inventory_row_not_required():
    b, cfs, catalog = fixture()
    b.matrix[0, 0] = 0
    cfs[("test", "climate change")] = cfs[("test", "climate change")].iloc[1:]
    assert run_audit(b, cfs, catalog)["status"] == "passed"


def test_unlabelled_used_flow_cannot_be_certified():
    b, cfs, catalog = fixture()
    assert run_audit(b, cfs, catalog.iloc[1:])["status"] == "failed"


def test_nonfinite_and_duplicate_factors_fail():
    b, cfs, catalog = fixture()
    key = ("test", "climate change")
    cfs[key].loc[0, "cf"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        run_audit(b, cfs, catalog)
    b, cfs, catalog = fixture()
    cfs[key] = pd.concat([cfs[key], cfs[key].iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        run_audit(b, cfs, catalog)


def many_flows_fixture(n=400):
    catalog = pd.DataFrame(
        [("bio", f"s{i}", f"Substance {i}", ("air",)) for i in range(n)],
        columns=["database", "code", "name", "categories"],
    )
    ids = [ExchangeFrameBuilder.flow_id_for(("bio", code)) for code in catalog.code]
    biosphere = BuiltMatrix(
        matrix=sparse.csr_matrix(np.ones((n, 1))),
        row_id_to_idx=dict(zip(ids, range(n), strict=True)),
        col_id_to_idx={1: 0},
    )
    cfs = {("test", "acidification"): pd.DataFrame({"flow_id": ids[:1], "cf": [1.0]})}
    return biosphere, cfs, catalog


def test_receipt_records_counts_and_does_not_grow_with_the_inventory():
    """Most flows have no factor in most methods; the run report must not list them."""
    sizes = {}
    for n in (40, 400):
        biosphere, cfs, catalog = many_flows_fixture(n)
        audit = run_audit(biosphere, cfs, catalog)
        method = audit["methods"][0]
        assert audit["emitted_flows"] == n
        assert method["characterized_flows"] == 1
        assert method["uncharacterized_flows"] == n - 1
        assert "missing_flow_ids" not in method
        sizes[n] = len(json.dumps(audit))
    assert sizes[400] - sizes[40] < 10  # a few more digits, not 360 more ids


def test_unlabelled_flows_are_counted_and_only_a_bounded_sample_is_listed():
    biosphere, cfs, catalog = many_flows_fixture(400)
    audit = run_audit(biosphere, cfs, catalog.iloc[:0])
    assert audit["unlabelled_emitted_flows"] == 400
    sample = audit["unlabelled_emitted_flow_id_sample"]
    assert len(sample) == CharacterizationAuditor.SAMPLE_SIZE
    assert set(sample) <= set(biosphere.row_id_to_idx)


CLIMATE = ("test", "climate change")
EF_AIR = ("Emissions", "Emissions to air")


def zero_by_definition_fixture():
    """Fossil and non-fossil CO2 are emitted. Only fossil CO2 has a factor on its own flow.
    The method's table lists biogenic CO2 under its own flow ids (``ef``), all zero."""
    catalog = pd.DataFrame(
        [
            ("bio", "fossil", "Carbon dioxide, fossil", ("air", "urban air close to ground")),
            ("bio", "nonfossil", "Carbon dioxide, non-fossil", ("air",)),
            ("ef", "b1", "carbon dioxide (biogenic)", (*EF_AIR, "Emissions to air, unspecified")),
            ("ef", "b2", "carbon dioxide (biogenic)", (*EF_AIR, "Emissions to air, indoor")),
            ("ef", "f1", "carbon dioxide (fossil)", (*EF_AIR, "Emissions to air, unspecified")),
        ],
        columns=["database", "code", "name", "categories"],
    )
    ids = {
        code: ExchangeFrameBuilder.flow_id_for((database, code))
        for database, code in zip(catalog.database, catalog.code, strict=True)
    }
    biosphere = BuiltMatrix(
        matrix=sparse.csr_matrix([[1.0], [1.0]]),
        row_id_to_idx={ids["fossil"]: 0, ids["nonfossil"]: 1},
        col_id_to_idx={1: 0},
    )
    factors = {"fossil": 1.0, "b1": 0.0, "b2": 0.0, "f1": 1.0}
    return biosphere, catalog, ids, factors


def climate_cfs(ids, factors):
    return {
        CLIMATE: pd.DataFrame(
            {"flow_id": [ids[code] for code in factors], "cf": list(factors.values())}
        )
    }


def test_method_defined_zero_passes_and_is_reported_apart_from_missing():
    biosphere, catalog, ids, factors = zero_by_definition_fixture()
    audit = run_audit(biosphere, climate_cfs(ids, factors), catalog)
    assert audit["status"] == "passed"
    method = audit["methods"][0]
    assert [g["name"] for g in method["zero_by_method_definition"]] == [
        "Carbon dioxide, non-fossil"
    ]
    assert method["missing_core_gases"] == []


def test_zero_is_not_assumed_without_the_method_tables_own_zero_rows():
    biosphere, catalog, ids, factors = zero_by_definition_fixture()
    for change in (
        lambda f: (f.pop("b1"), f.pop("b2")),  # the method table does not list the species
        lambda f: f.pop("b2"),  # it lists the species incompletely
        lambda f: f.update(b2=1.0),  # it gives the species a factor
    ):
        changed = dict(factors)
        change(changed)
        audit = run_audit(biosphere, climate_cfs(ids, changed), catalog)
        assert audit["status"] == "failed", changed
        method = audit["methods"][0]
        assert [g["name"] for g in method["missing_core_gases"]] == ["Carbon dioxide, non-fossil"]
        assert method["zero_by_method_definition"] == []


def test_a_gas_the_policy_does_not_name_is_never_excused():
    biosphere, catalog, ids, _ = zero_by_definition_fixture()
    # Fossil CO2 loses its factor; even an all-zero method row for it must not excuse that.
    changed = {"b1": 0.0, "b2": 0.0, "f1": 0.0}
    audit = run_audit(biosphere, climate_cfs(ids, changed), catalog)
    assert audit["status"] == "failed"
    assert [g["name"] for g in audit["methods"][0]["missing_core_gases"]] == [
        "Carbon dioxide, fossil"
    ]


def emitted_fixture(rows):
    """Every catalog row is emitted once. ``rows``: (database, code, name, categories)."""
    catalog = pd.DataFrame(rows, columns=["database", "code", "name", "categories"])
    ids = [
        ExchangeFrameBuilder.flow_id_for((database, code))
        for database, code in zip(catalog.database, catalog.code, strict=True)
    ]
    biosphere = BuiltMatrix(
        matrix=sparse.csr_matrix(np.ones((len(ids), 1))),
        row_id_to_idx=dict(zip(ids, range(len(ids)), strict=True)),
        col_id_to_idx={1: 0},
    )
    return biosphere, catalog, ids


def missing_names(audit):
    return sorted(g["name"] for g in audit["methods"][0]["missing_core_gases"])


def test_land_use_change_co2_and_methane_are_core_gases():
    """ecoinvent 3.9.1 and biosphere3 name these flows differently from fossil CO2/CH4."""
    names = [
        "Carbon dioxide, from soil or biomass stock",
        "Methane, from soil or biomass stock",
        "Carbon dioxide, land transformation",
        "Methane, land transformation",
    ]
    biosphere, catalog, ids = emitted_fixture(
        [("bio", f"luc{i}", name, ("air",)) for i, name in enumerate(names)]
        + [("bio", "nh3", "Ammonia", ("air",))]
    )
    without = {CLIMATE: pd.DataFrame({"flow_id": ids[-1:], "cf": [0.0]})}
    audit = run_audit(biosphere, without, catalog)
    assert audit["status"] == "failed"
    assert missing_names(audit) == sorted(names)
    with_factors = {CLIMATE: pd.DataFrame({"flow_id": ids[:4], "cf": [1.0, 29.8, 1.0, 29.8]})}
    assert run_audit(biosphere, with_factors, catalog)["status"] == "passed"


def test_core_gases_linked_to_the_methods_own_flows_are_required_too():
    """The matcher may link an emission to an ``ef`` flow; those carry the EF names."""
    names = [
        "carbon dioxide (fossil)",
        "Carbon dioxide (land use change)",
        "methane (fossil)",
        "methane (biogenic)",
        "methane (land use change)",
        "nitrous oxide",
    ]
    air = (*EF_AIR, "Emissions to air, unspecified")
    biosphere, catalog, ids = emitted_fixture(
        [("ef", f"g{i}", name, air) for i, name in enumerate(names)]
        # Not emissions to air: never required, whatever the name.
        + [
            ("ef", "soil", "nitrous oxide", ("Emissions", "Emissions to soil", "x")),
            ("ef", "uptake", "carbon dioxide (fossil)", ("Resources", "Resources from air", "x")),
        ]
    )
    unrelated = {CLIMATE: pd.DataFrame({"flow_id": ids[-1:], "cf": [0.0]})}
    audit = run_audit(biosphere, unrelated, catalog)
    assert audit["status"] == "failed"
    assert missing_names(audit) == sorted(names)


def test_compartment_is_read_whatever_shape_the_catalog_gives_it():
    shapes = [("air",), ["air", "urban air close to ground"], np.array(["air"], dtype=object)]
    shapes += [(" Air ",), "air"]
    biosphere, catalog, _ = emitted_fixture(
        [("bio", f"m{i}", "Methane, fossil", shape) for i, shape in enumerate(shapes)]
        + [("bio", "none", "Methane, fossil", None), ("bio", "nan", "Methane, fossil", np.nan)]
    )
    unrelated = {CLIMATE: pd.DataFrame({"flow_id": [-1], "cf": [1.0]})}
    audit = run_audit(biosphere, unrelated, catalog)
    # Five rows say "air" in one shape or another; a row without a compartment is not air.
    assert len(audit["methods"][0]["missing_core_gases"]) == len(shapes)


def test_a_zero_factor_on_a_core_gas_needs_the_methods_definition_too():
    """A row that says 0 for fossil CO2, methane or N2O scores it zero as surely as no row."""
    biosphere, catalog, ids = emitted_fixture(
        [
            ("bio", "co2f", "Carbon dioxide, fossil", ("air",)),
            ("bio", "ch4", "Methane, fossil", ("air",)),
            ("bio", "n2o", "Dinitrogen monoxide", ("air",)),
            ("bio", "co2b", "Carbon dioxide, biogenic", ("air",)),
            ("ef", "b1", "carbon dioxide (biogenic)", (*EF_AIR, "Emissions to air, indoor")),
        ]
    )
    zeros = {CLIMATE: pd.DataFrame({"flow_id": ids, "cf": [0.0, 0.0, -0.0, 0.0, 0.0]})}
    audit = run_audit(biosphere, zeros, catalog)
    assert audit["status"] == "failed"
    method = audit["methods"][0]
    assert sorted(g["name"] for g in method["zero_factor_core_gases"]) == [
        "Carbon dioxide, fossil",
        "Dinitrogen monoxide",
        "Methane, fossil",
    ]
    assert sorted(g["name"] for g in method["zero_by_method_definition"]) == [
        "Carbon dioxide, biogenic",
        "carbon dioxide (biogenic)",
    ]
    assert method["missing_core_gases"] == []

    factors = {CLIMATE: pd.DataFrame({"flow_id": ids, "cf": [1.0, 29.8, 273.0, 0.0, 0.0]})}
    audit = run_audit(biosphere, factors, catalog)
    assert audit["status"] == "passed"
    assert audit["methods"][0]["zero_factor_core_gases"] == []


def test_without_a_headline_climate_method_nothing_was_audited():
    """A pass must mean the core gases were checked, not that there was nothing to check."""
    biosphere, cfs, catalog = fixture()
    no_methane_factor = cfs[CLIMATE].iloc[1:]
    for method_cfs in (
        {},
        {("test", "acidification"): no_methane_factor},
        {("test", "climate change: fossil"): no_methane_factor},  # a sub-indicator
    ):
        audit = run_audit(biosphere, method_cfs, catalog)
        assert audit["status"] == "not_audited", method_cfs.keys()
        assert audit["not_audited_reason"] == "the registry holds no headline climate-change method"
        assert audit["climate_methods_audited"] == 0
        assert audit["failed_methods"] == []
    audit = run_audit(biosphere, cfs, catalog)
    assert (audit["status"], audit["climate_methods_audited"]) == ("passed", 1)


def test_a_pass_needs_an_emitted_core_gas_that_was_checked():
    """A climate method that recognised no emitted core gas certified nothing.

    Here fossil CO2 is emitted without a factor, but its compartment arrives as one
    string the audit does not read as air. Before, that was a pass with an empty
    ``missing_core_gases``; now the receipt says 0 core-gas flows were checked and
    the status is ``not_audited``, with the reason."""
    unread = "air/urban air close to ground"
    biosphere, catalog, ids = emitted_fixture(
        [
            ("bio", "co2", "Carbon dioxide, fossil", unread),
            ("bio", "nh3", "Ammonia", ("air",)),
        ]
    )
    no_co2_factor = {CLIMATE: pd.DataFrame({"flow_id": ids[1:], "cf": [0.0]})}
    audit = run_audit(biosphere, no_co2_factor, catalog)
    assert audit["status"] == "not_audited"
    assert "no emitted flow was recognised as a core gas" in audit["not_audited_reason"]
    assert audit["methods"][0]["core_gas_flows_checked"] == 0
    assert audit["failed_methods"] == []

    # The same flow with a compartment the audit reads: checked, and it fails.
    catalog.at[0, "categories"] = ("air", "urban air close to ground")
    audit = run_audit(biosphere, no_co2_factor, catalog)
    assert audit["status"] == "failed"
    assert audit["not_audited_reason"] is None
    assert audit["methods"][0]["core_gas_flows_checked"] == 1
    assert missing_names(audit) == ["Carbon dioxide, fossil"]

    # With its factor it passes, and the receipt says what was checked.
    factors = {CLIMATE: pd.DataFrame({"flow_id": ids, "cf": [1.0, 0.0]})}
    audit = run_audit(biosphere, factors, catalog)
    assert (audit["status"], audit["not_audited_reason"]) == ("passed", None)
    assert audit["methods"][0]["core_gas_flows_checked"] == 1


def test_only_climate_methods_report_the_core_gases_they_checked():
    biosphere, cfs, catalog = fixture()
    cfs[("test", "acidification")] = cfs[CLIMATE]
    audit = run_audit(biosphere, cfs, catalog)
    by_method = {tuple(m["method"]): m for m in audit["methods"]}
    # Methane and biogenic CO2 are emitted core gases; ammonia is not one.
    assert by_method[CLIMATE]["core_gas_flows_checked"] == 2
    assert "core_gas_flows_checked" not in by_method[("test", "acidification")]


def test_headline_method_is_recognised_whatever_its_case_and_spacing():
    biosphere, cfs, catalog = fixture()
    spelled = {("EF v3.1", " Climate Change ", "GWP100"): cfs[CLIMATE].iloc[1:]}
    audit = run_audit(biosphere, spelled, catalog)
    assert audit["status"] == "failed"
    assert missing_names(audit) == ["Methane"]


def test_a_catalog_with_two_rows_for_one_flow_is_refused():
    """The label of a flow must not depend on which of two rows came last."""
    biosphere, cfs, catalog = fixture()
    twice = pd.concat([catalog, catalog.iloc[:1].assign(name="Carbon dioxide, biogenic")])
    for ordered in (twice, twice.iloc[::-1]):
        with pytest.raises(ValueError, match="Duplicate biosphere catalog rows"):
            run_audit(biosphere, cfs, ordered)


def test_receipt_is_written_whatever_integer_type_the_flow_ids_have(tmp_path):
    biosphere, cfs, catalog = fixture()
    numpy_ids = BuiltMatrix(
        matrix=biosphere.matrix,
        row_id_to_idx={np.int64(key): row for key, row in biosphere.row_id_to_idx.items()},
        col_id_to_idx=biosphere.col_id_to_idx,
    )
    auditor = CharacterizationAuditor()
    audit = auditor.audit(numpy_ids, cfs, catalog.iloc[1:])  # one emitted flow is unlabelled
    assert audit["unlabelled_emitted_flows"] == 1
    receipt = json.loads(auditor.write_receipt(audit, "abc", tmp_path / "audit.json").read_text())
    assert receipt["unlabelled_emitted_flow_id_sample"] == [
        int(ExchangeFrameBuilder.flow_id_for(("bio", "ch4")))
    ]
    assert list(tmp_path.iterdir()) == [tmp_path / "audit.json"]


def test_ef31_source_table_gives_every_policy_species_a_zero_factor_to_air():
    """The policy names species; the public JRC table in ``source/`` is its evidence."""
    table = pd.read_parquet(Settings().paths.ef_cf_parquet)
    species = set(CharacterizationAuditor.ZERO_BY_METHOD_DEFINITION.values())
    rows = table[
        (table["LCIAMethod_name"] == "Climate change")
        & table["FLOW_name"].str.casefold().isin(species)
        & (table["FLOW_class1"] == "Emissions to air")
    ]
    assert set(rows["FLOW_name"].str.casefold()) == species
    assert rows["CF EF3.1"].eq(0).all()


class TestRealRegistry:
    """Against the built, licensed registry (``registry/``). Skipped where it is absent."""

    @staticmethod
    def _load():
        paths = Settings().paths
        if not (
            paths.registry_method_cfs_index.exists() and paths.registry_biosphere_catalog.exists()
        ):
            pytest.skip("Built registry not present (licensed; see BOOTSTRAP.md)")
        catalog = pd.read_parquet(paths.registry_biosphere_catalog)
        method_cfs = {
            key: LinkAllPipeline._project_method_cfs(frame)
            for key, frame in MethodCfRegistryLoader(paths.registry_method_cfs_dir)
            .load_all()
            .items()
        }
        # Treat every flow of the inventory's biosphere database and of the method's own
        # flow list as emitted: a superset of what a link run can emit from those two, so
        # a pass here holds for any such run. (The matcher's last resort, biosphere3, has
        # no factors in this registry; a core gas linked there fails the audit by design.)
        settings = Settings()
        emitted = catalog[
            catalog["database"].isin([settings.biosphere_db_name, settings.ef_db_name])
        ]
        ids = [
            ExchangeFrameBuilder.flow_id_for((database, code))
            for database, code in zip(emitted["database"], emitted["code"], strict=True)
        ]
        biosphere = BuiltMatrix(
            matrix=sparse.csr_matrix(np.ones((len(ids), 1))),
            row_id_to_idx={flow_id: row for row, flow_id in enumerate(ids)},
            col_id_to_idx={1: 0},
        )
        return biosphere, method_cfs, catalog

    @staticmethod
    def _headline(audit):
        (method,) = [m for m in audit["methods"] if "climate change" in m["method"]]
        return method

    def test_registry_passes_and_biogenic_co2_is_the_only_method_defined_zero(self):
        biosphere, method_cfs, catalog = self._load()
        audit = run_audit(biosphere, method_cfs, catalog)
        assert audit["status"] == "passed"
        headline = self._headline(audit)
        assert headline["missing_core_gases"] == []
        assert headline["zero_factor_core_gases"] == []
        # The inventory flow (no row of its own) and the method's own flows (zero rows).
        assert {g["name"] for g in headline["zero_by_method_definition"]} == {
            "Carbon dioxide, non-fossil",
            "carbon dioxide (biogenic)",
        }

    def test_registry_without_its_fossil_co2_factors_fails(self):
        biosphere, method_cfs, catalog = self._load()
        fossil = {
            ExchangeFrameBuilder.flow_id_for((row.database, row.code))
            for row in catalog.itertuples(index=False)
            if row.name == "Carbon dioxide, fossil"
        }
        key = next(k for k in method_cfs if "climate change" in k)
        method_cfs[key] = method_cfs[key][~method_cfs[key]["flow_id"].isin(fossil)]
        audit = run_audit(biosphere, method_cfs, catalog)
        assert audit["status"] == "failed"
        assert {g["name"] for g in self._headline(audit)["missing_core_gases"]} == {
            "Carbon dioxide, fossil"
        }

    def test_registry_without_the_methods_zero_rows_fails(self):
        biosphere, method_cfs, catalog = self._load()
        biogenic = {
            ExchangeFrameBuilder.flow_id_for((row.database, row.code))
            for row in catalog.itertuples(index=False)
            if row.name == "carbon dioxide (biogenic)"
        }
        key = next(k for k in method_cfs if "climate change" in k)
        method_cfs[key] = method_cfs[key][~method_cfs[key]["flow_id"].isin(biogenic)]
        audit = run_audit(biosphere, method_cfs, catalog)
        assert audit["status"] == "failed"
        # The method's own flows lost their rows, and with them the inventory flow its excuse.
        assert {g["name"] for g in self._headline(audit)["missing_core_gases"]} == {
            "carbon dioxide (biogenic)",
            "Carbon dioxide, non-fossil",
        }


def test_audit_follows_the_source_conventions():
    """``src/`` convention for the audit: a class, not a module-level function, and
    imported at the top of ``link_all``, not inside the function that runs it. Only
    the audit's own import is checked; other imports in ``link_all`` are not this
    test's business."""
    audit_module = ast.parse((SRC / "scoring/characterization_audit.py").read_text())
    assert [n.name for n in audit_module.body if isinstance(n, ast.FunctionDef)] == []
    assert "CharacterizationAuditor" in [
        n.name for n in audit_module.body if isinstance(n, ast.ClassDef)
    ]

    def names_the_audit(node):
        module = getattr(node, "module", None) or ""
        return module.endswith("characterization_audit") or any(
            alias.name in ("CharacterizationAuditor", "characterization_audit")
            or alias.name.endswith(".characterization_audit")
            for alias in node.names
        )

    link_all = ast.parse((SRC / "pipelines/link_all.py").read_text())
    top_level = [
        node
        for node in link_all.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and names_the_audit(node)
    ]
    nested = [
        ast.unparse(node)
        for function in ast.walk(link_all)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and names_the_audit(node)
    ]
    assert top_level, "link_all must import the audit at the top of the file"
    assert nested == []
