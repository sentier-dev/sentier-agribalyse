"""Unit tests for ``StableCodeAssigner``.

``bw_simapro_csv`` draws a ``uuid4`` for the code of every product node, of
every process without a ``Process identifier`` and of every allocated
(read-only) process. The load-bearing case: two draws of the same data end
with the same codes, whatever the order of the blocks.

The reader is stubbed in the unit suite, so ``_Reader`` below writes the
same shape ``lci_to_brightway`` does, with a fresh ``uuid4`` wherever the
library draws one.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pytest

from transforms import sp_csv_parser
from transforms.sp_csv_parser import SimaProCsvParser
from transforms.stable_codes import StableCodeAssigner

DB = "agb"
HEX32 = re.compile(r"^[0-9a-f]{32}$")


class _Sub:
    def __init__(self, parsed):
        self.parsed = parsed


class Process:
    """Stand-in for ``bw_simapro_csv.blocks.Process`` (duck-typed by name)."""

    def __init__(self, identifier, name, products):
        self.parsed = {"metadata": {"Process identifier": identifier}}
        self.blocks = {
            "Input parameters": _Sub(
                [{"name": "SP_X", "original_name": "x", "amount": 1.0, "comment": ""}]
            )
        }
        self.name = name
        self.products = products


class _Reader:
    """Writes what ``lci_to_brightway(separate_products=True)`` writes."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.database_name = DB

    def to_brightway(self, separate_products=True, shorten_names=True):
        processes: list[dict] = []
        products: list[dict] = []
        for block in self.blocks:
            identifier = block.parsed["metadata"]["Process identifier"]
            code = identifier or uuid4().hex
            run = uuid4().hex
            functional = []
            for name in block.products:
                product_code = uuid4().hex
                products.append(
                    {
                        "type": "product",
                        "database": DB,
                        "code": product_code,
                        "name": name,
                        "unit": "kg",
                        "reference process": (DB, code),
                    }
                )
                functional.append(
                    {
                        "type": "production",
                        "functional": True,
                        "name": name,
                        "unit": "kg",
                        "amount": 1.0,
                        "input": (DB, product_code),
                    }
                )
            other = {"type": "biosphere", "name": "Ammonia", "unit": "kg", "amount": 0.5}
            parent = {
                "database": DB,
                "code": code,
                "name": block.name,
                "type": "multifunctional" if len(functional) > 1 else "process",
                "exchanges": [dict(other), *functional],
            }
            processes.append(parent)
            if len(functional) < 2:
                continue
            parent["mf_allocation_run_uuid"] = run
            for exc in functional:
                child_code = uuid4().hex
                exc["mf_allocated_process_code"] = child_code
                processes.append(
                    {
                        "database": DB,
                        "code": child_code,
                        "name": f"{exc['name']} (read-only process)",
                        "type": "readonly_process",
                        "mf_parent_key": (DB, code),
                        "mf_allocation_run_uuid": run,
                        "exchanges": [
                            {k: v for k, v in exc.items() if not k.startswith("mf_")},
                            dict(other),
                        ],
                    }
                )
        return {"database": {"name": DB}, "processes": processes, "products": products}


def _blocks() -> list[Process]:
    return [
        Process("REAL1", "Wheat", ["Wheat grain"]),
        Process(None, "Fertiliser", ["Fertiliser"]),
        Process("REAL3", "Mill", ["Oil", "Meal"]),
        Process("", "Dairy", ["Cream", "Skimmed milk"]),
    ]


def _assigned(blocks: list[Process]) -> dict:
    reader = _Reader(blocks)
    bw_data = reader.to_brightway()
    StableCodeAssigner().apply(reader, bw_data)
    return bw_data


def _codes_by_name(bw_data: dict) -> dict[tuple[str, str], str]:
    return {
        (ds["type"], ds["name"]): ds["code"] for ds in (*bw_data["processes"], *bw_data["products"])
    }


class TestSameCodesFromEveryDraw:
    def test_the_reader_stand_in_draws_new_codes_each_time(self):
        """Guard of the guard: without the assigner two draws differ."""
        reader = _Reader(_blocks())
        assert reader.to_brightway() != reader.to_brightway()

    def test_two_draws_end_with_the_same_data(self):
        assert _assigned(_blocks()) == _assigned(_blocks())

    def test_codes_do_not_depend_on_where_a_block_sits(self):
        full = _codes_by_name(_assigned(_blocks()))
        blocks = _blocks()
        # The first block removed (as the ecoinvent strip does), the rest reversed.
        fewer = _codes_by_name(_assigned(blocks[:0:-1]))
        assert fewer
        assert set(fewer) < set(full)
        assert {key: full[key] for key in fewer} == fewer

    def test_a_process_identifier_is_kept(self):
        codes = _codes_by_name(_assigned(_blocks()))
        assert codes[("process", "Wheat")] == "REAL1"
        assert codes[("multifunctional", "Mill")] == "REAL3"

    def test_derived_codes_keep_the_shape_of_the_drawn_ones(self):
        codes = _codes_by_name(_assigned(_blocks()))
        derived = [code for code in codes.values() if not code.startswith("REAL")]
        assert len(derived) == 2 + 6 + 4  # processes, products, allocated processes
        assert all(HEX32.match(code) for code in derived)

    def test_stats_count_each_kind(self):
        reader = _Reader(_blocks())
        stats = StableCodeAssigner().apply(reader, reader.to_brightway())
        assert stats == {
            "processes": 2,
            "products": 6,
            "allocated_processes": 4,
            "told_apart_by_order": 0,
        }


class TestReferencesFollowTheirNode:
    @pytest.fixture
    def bw_data(self) -> dict:
        return _assigned(_blocks())

    @pytest.fixture
    def keys(self, bw_data) -> set[tuple[str, str]]:
        datasets = (*bw_data["processes"], *bw_data["products"])
        keys = {(ds["database"], ds["code"]) for ds in datasets}
        assert len(keys) == len(datasets)
        return keys

    def test_exchange_inputs(self, bw_data, keys):
        inputs = [
            exc["input"] for ds in bw_data["processes"] for exc in ds["exchanges"] if "input" in exc
        ]
        assert len(inputs) == 6 + 4  # one per product on its parent, one per allocated process
        assert all(isinstance(key, tuple) and key in keys for key in inputs)

    def test_reference_process_of_products(self, bw_data, keys):
        references = [ds["reference process"] for ds in bw_data["products"]]
        assert all(isinstance(key, tuple) and key in keys for key in references)
        by_name = {ds["name"]: ds["reference process"][1] for ds in bw_data["products"]}
        codes = _codes_by_name(bw_data)
        assert by_name["Fertiliser"] == codes[("process", "Fertiliser")]
        assert by_name["Cream"] == codes[("multifunctional", "Dairy")]

    def test_parent_links_and_allocated_codes(self, bw_data, keys):
        children = [ds for ds in bw_data["processes"] if "mf_parent_key" in ds]
        assert len(children) == 4
        assert all(ds["mf_parent_key"] in keys for ds in children)
        allocated = {
            exc["mf_allocated_process_code"]
            for ds in bw_data["processes"]
            for exc in ds["exchanges"]
            if "mf_allocated_process_code" in exc
        }
        assert allocated == {ds["code"] for ds in children}

    def test_an_allocated_process_keeps_the_product_of_its_parent(self, bw_data):
        by_code = {ds["code"]: ds for ds in bw_data["processes"]}
        for parent in bw_data["processes"]:
            for exc in parent["exchanges"]:
                if "mf_allocated_process_code" in exc:
                    child = by_code[exc["mf_allocated_process_code"]]
                    assert child["mf_parent_key"] == (DB, parent["code"])
                    assert child["exchanges"][0]["input"] == exc["input"]

    def test_the_allocation_run_is_derived_and_shared_with_the_children(self, bw_data):
        runs = {
            ds.get("mf_parent_key", (DB, ds["code"]))[1]: set()
            for ds in bw_data["processes"]
            if "mf_allocation_run_uuid" in ds
        }
        for ds in bw_data["processes"]:
            if "mf_allocation_run_uuid" in ds:
                runs[ds.get("mf_parent_key", (DB, ds["code"]))[1]].add(ds["mf_allocation_run_uuid"])
        assert len(runs) == 2
        assert all(len(values) == 1 for values in runs.values())
        assert len({next(iter(values)) for values in runs.values()}) == 2


class TestWhatACodeIsDerivedFrom:
    """Each part of a derivation matters: change it and the code changes."""

    @staticmethod
    def _process_code(blocks: list[Process]) -> str:
        return next(
            ds["code"] for ds in _assigned(blocks)["processes"] if "mf_parent_key" not in ds
        )

    def test_a_process_without_identifier_by_its_name(self):
        assert self._process_code([Process(None, "A", ["P"])]) != self._process_code(
            [Process(None, "B", ["P"])]
        )

    def test_a_process_without_identifier_by_its_products(self):
        assert self._process_code([Process(None, "A", ["P"])]) != self._process_code(
            [Process(None, "A", ["Q"])]
        )

    def test_a_process_without_identifier_by_the_units_of_its_products(self):
        reader = _Reader([Process(None, "A", ["P"])])
        in_kg = reader.to_brightway()
        in_litres = reader.to_brightway()
        for exc in in_litres["processes"][0]["exchanges"]:
            if exc.get("functional"):
                exc["unit"] = "l"
        StableCodeAssigner().apply(reader, in_kg)
        StableCodeAssigner().apply(reader, in_litres)
        assert in_kg["processes"][0]["code"] != in_litres["processes"][0]["code"]

    def test_an_allocated_process_by_its_parent(self):
        def child_codes(identifier: str) -> set[str]:
            data = _assigned([Process(identifier, "Mill", ["Oil", "Meal"])])
            return {ds["code"] for ds in data["processes"] if "mf_parent_key" in ds}

        assert child_codes("REAL3").isdisjoint(child_codes("REAL4"))


class TestEdgeCases:
    def test_nodes_with_the_same_content_are_told_apart_by_order(self):
        twins = [Process(None, "Twin", ["Same product"]), Process(None, "Twin", ["Same product"])]
        reader = _Reader(twins)
        bw_data = reader.to_brightway()
        stats = StableCodeAssigner().apply(reader, bw_data)
        codes = [ds["code"] for ds in (*bw_data["processes"], *bw_data["products"])]
        assert len(set(codes)) == 4
        assert stats["told_apart_by_order"] == 1  # the processes; each product hangs on its own
        assert bw_data == _assigned(twins)

    def test_a_product_named_twice_on_one_process(self):
        bw_data = _assigned([Process("REAL1", "Twice", ["Same", "Same"])])
        assert len({ds["code"] for ds in bw_data["products"]}) == 2
        assert bw_data == _assigned([Process("REAL1", "Twice", ["Same", "Same"])])

    def test_quoted_empty_identifier_counts_as_none(self):
        codes = _codes_by_name(_assigned([Process('""', "Quoted", ["Product"])]))
        assert HEX32.match(codes[("process", "Quoted")])

    def test_a_code_no_block_explains_raises(self):
        reader = _Reader([Process("REAL1", "Wheat", ["Wheat grain"])])
        bw_data = reader.to_brightway()
        bw_data["processes"][0]["code"] = "NOT-IN-THE-FILE"
        with pytest.raises(ValueError, match="cannot tell which codes"):
            StableCodeAssigner().apply(reader, bw_data)

    def test_a_derived_code_equal_to_an_identifier_raises(self):
        reader = _Reader([Process(None, "Fertiliser", ["Fertiliser"])])
        taken = _codes_by_name(_assigned(reader.blocks))[("process", "Fertiliser")]
        clash = _Reader([*reader.blocks, Process(taken, "Other", ["Other product"])])
        with pytest.raises(ValueError, match="more than one dataset"):
            StableCodeAssigner().apply(clash, clash.to_brightway())

    def test_a_parser_without_blocks_is_left_alone(self):
        class _NoBlocks:
            pass

        bw_data = {"processes": [{"name": "P1", "type": "process", "exchanges": []}]}
        stats = StableCodeAssigner().apply(_NoBlocks(), bw_data)
        assert bw_data == {"processes": [{"name": "P1", "type": "process", "exchanges": []}]}
        assert set(stats.values()) == {0}

    def test_processes_without_a_recognised_process_block_raise(self):
        """A reader that has blocks, none of them recognised as a Process,
        and still returns processes: ``bw_simapro_csv`` changed its block
        class. The codes would stay random under the stable scheme."""

        class ProcessBlock(Process):  # another class name than ``Process``
            pass

        reader = _Reader([ProcessBlock("REAL1", "Wheat", ["Wheat grain"])])
        with pytest.raises(ValueError, match="none of its blocks is recognised"):
            StableCodeAssigner().apply(reader, reader.to_brightway())

    def test_a_reader_with_no_process_and_no_process_block_is_left_alone(self):
        reader = _Reader([])
        bw_data = reader.to_brightway()
        stats = StableCodeAssigner().apply(reader, bw_data)
        assert set(stats.values()) == {0}


class TestParserAssignsBeforeAnythingReadsACode:
    @pytest.fixture
    def parse(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(sp_csv_parser, "SimaProCSV", lambda **kw: _Reader(_blocks()))
        csv = tmp_path / "agb.csv"
        csv.write_bytes(b"fake")
        return SimaProCsvParser(csv_path=csv, database_name=DB).parse

    def test_two_parses_give_the_same_data_and_parameters(self, parse):
        first, second = parse(), parse()
        assert first.data == second.data
        assert first.parameters == second.parameters

    def test_parameter_rows_are_keyed_by_the_stable_codes(self, parse):
        parsed = parse()
        codes = {ds["code"] for ds in parsed.data if ds["type"] != "product"}
        assert {row["process_code"] for row in parsed.parameters} == codes
        parents = {row["mf_parent_code"] for row in parsed.parameters if "mf_parent_code" in row}
        assert parents == {ds["code"] for ds in parsed.data if ds["type"] == "multifunctional"}

    def test_the_result_names_the_code_scheme(self, parse):
        assert parse().code_scheme == StableCodeAssigner.SCHEME
