"""Integration tests: the codes of a parse through the real ``bw_simapro_csv``.

The reader draws a ``uuid4`` for every product node, for every process
without a ``Process identifier`` and — in ``multifunctional``'s allocation
step — for every read-only child of a multi-output process. Parsing a
small synthetic export twice through ``SimaProCsvParser`` pins what
``StableCodeAssigner`` promises:

1. two parses of one file give the same datasets and parameter rows;
2. a ``Process identifier`` stays the code of its process;
3. the codes do not depend on where a block sits in the file, nor on
   which other blocks the file holds (the ecoinvent-free export is the
   full one minus 2,264 blocks), unless two nodes share the whole
   derivation: those are told apart by their order;
4. every reference still points at a dataset.

The export is synthetic (invented names and amounts): one process with an
identifier, one without, a multi-output process of each kind, and a waste
treatment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import pytest

from transforms import sp_csv_parser
from transforms.sp_csv_parser import ParsedSimaProCsv, SimaProCsvParser

HEX32 = re.compile(r"^[0-9a-f]{32}$")


class SyntheticExport:
    """A SimaPro CSV export assembled from named process blocks."""

    DATABASE: ClassVar[str] = "synthetic"

    HEADER: ClassVar[str] = "\n".join(
        [
            "{SimaPro 9.6.0.1}",
            "{processes}",
            "{Date: 08/11/2024}",
            "{Time: 17:56:59}",
            "{Project: Synthetic - unit}",
            "{CSV Format version: 9.0.0}",
            "{CSV separator: Semicolon}",
            "{Decimal separator: ,}",
            "{Date separator: /}",
            "{Short date format: dd/MM/yyyy}",
            "{Export platform IDs: No}",
            "{Skip empty fields: No}",
            "{Convert expressions to constants: No}",
            "{Selection: Selection (5)}",
            "{Related objects (system descriptions, substances, units, etc.): Yes}",
            "{Include sub product stages and processes: No}",
            "{Skip unused parameters: Yes}",
            "{Open library: 'Synthetic - unit'}",
            "",
            "",
        ]
    )

    _METADATA: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Type", ""),
        ("Process name", ""),
        ("Status", ""),
        ("Time period", "Unspecified"),
        ("Geography", "Unspecified"),
        ("Technology", "Unspecified"),
        ("Representativeness", "Unspecified"),
        ("Multiple output allocation", "Unspecified"),
        ("Substitution allocation", "Unspecified"),
        ("Cut off rules", "Unspecified"),
        ("Capital goods", "Unspecified"),
        ("Boundary with nature", "Unspecified"),
        ("Infrastructure", "No"),
        ("Date", "10/08/2023"),
        ("Record", ""),
        ("Generator", ""),
        ("External documents", ""),
        ("Literature references", ""),
        ("Collection method", ""),
        ("Data treatment", ""),
        ("Verification", ""),
        ("Comment", ""),
        ("Allocation rules", ""),
        ("System description", ""),
    )

    @classmethod
    def block(
        cls,
        identifier: str,
        *,
        products: tuple[str, ...] = (),
        waste: tuple[str, ...] = (),
        materials: tuple[str, ...] = (),
        air: tuple[str, ...] = (),
        parameters: tuple[str, ...] = (),
    ) -> str:
        lines = ["Process", "", "Category type", "material" if products else "waste treatment", ""]
        lines += ["Process identifier", identifier, ""]
        for label, value in cls._METADATA:
            lines += [label, value, ""]
        if products:
            lines += ["Products", *products, ""]
        if waste:
            lines += ["Waste treatment", *waste, ""]
        lines += ["Avoided products", "", "Resources", ""]
        lines += ["Materials/fuels", *materials, "", "Electricity/heat", ""]
        lines += ["Emissions to air", *air, "", "Emissions to water", "", "Emissions to soil", ""]
        lines += ["Final waste flows", "", "Non material emissions", "", "Social issues", ""]
        lines += ["Economic issues", "", "Waste to treatment", ""]
        lines += ["Input parameters", *parameters, "", "Calculated parameters", "", "End", "", ""]
        return "\n".join(lines) + "\n"

    @classmethod
    def blocks(cls) -> dict[str, str]:
        return {
            "wheat": cls.block(
                "TEST000000000000000001",
                products=("Wheat grain, at farm {FR} U;kg;1;100;not defined;Synthetic\\Crops;",),
                materials=("Fertiliser, at plant {FR} U;kg;dose*2;Undefined;0;0;0;",),
                air=("Ammonia;;kg;0,002;Undefined;0;0;0;",),
                parameters=("dose;0,05;Undefined;1;0;0;No;fertiliser dose",),
            ),
            "fertiliser": cls.block(
                "",
                products=("Fertiliser, at plant {FR} U;kg;1;100;not defined;Synthetic\\Inputs;",),
                air=("Carbon dioxide, fossil;;kg;1,5;Undefined;0;0;0;",),
            ),
            "mill": cls.block(
                "TEST000000000000000003",
                products=(
                    "Rapeseed oil, at mill {FR} U;kg;0,4;70;not defined;Synthetic\\Oils;",
                    "Rapeseed meal, at mill {FR} U;kg;0,6;30;not defined;Synthetic\\Feed;",
                ),
                materials=("Wheat grain, at farm {FR} U;kg;share*3;Undefined;0;0;0;",),
                air=("Carbon dioxide, fossil;;kg;0,25;Undefined;0;0;0;",),
                parameters=("share;0,5;Undefined;1;0;0;No;share of grain",),
            ),
            "dairy": cls.block(
                "",
                products=(
                    "Cream, at dairy {FR} U;kg;0,1;60;not defined;Synthetic\\Dairy;",
                    "Skimmed milk, at dairy {FR} U;kg;0,9;40;not defined;Synthetic\\Dairy;",
                ),
                materials=("Wheat grain, at farm {FR} U;kg;0,2;Undefined;0;0;0;",),
            ),
            "landfill": cls.block(
                "TEST000000000000000005",
                waste=("Waste plastic, to landfill {FR} U;kg;1;All waste types;Synthetic\\Waste;",),
                air=("Methane, biogenic;;kg;0,01;Undefined;0;0;0;",),
            ),
        }

    @classmethod
    def write(cls, path: Path, names: tuple[str, ...]) -> Path:
        blocks = cls.blocks()
        path.write_text(cls.HEADER + "".join(blocks[name] for name in names), encoding="latin-1")
        return path


ALL_BLOCKS = ("wheat", "fertiliser", "mill", "dairy", "landfill")


class Codes:
    """Views over one parse, keyed by what a dataset is (never by its code)."""

    def __init__(self, parsed: ParsedSimaProCsv):
        self.parsed = parsed

    @property
    def by_name(self) -> dict[tuple[str, str], str]:
        return {(ds["type"], ds["name"]): ds["code"] for ds in self.parsed.data}

    @property
    def keys(self) -> set[tuple[str, str]]:
        return {(ds["database"], ds["code"]) for ds in self.parsed.data}

    @property
    def parameters(self) -> set[tuple[str, str, str | None]]:
        return {
            (row["name"], row["process_code"], row.get("mf_parent_code"))
            for row in self.parsed.parameters
        }


@pytest.fixture(scope="module")
def parse(real_bw_simapro_csv, tmp_path_factory):
    """``parse(names)``: write the blocks to a file and parse it with the real reader."""
    folder = tmp_path_factory.mktemp("synthetic_export")

    def run(names: tuple[str, ...] = ALL_BLOCKS, file_name: str = "export.csv") -> Codes:
        path = SyntheticExport.write(folder / file_name, names)
        parser = SimaProCsvParser(csv_path=path, database_name=SyntheticExport.DATABASE)
        return Codes(parser.parse())

    # ``sp_csv_parser`` bound the stubbed reader when the suite imported it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sp_csv_parser, "SimaProCSV", real_bw_simapro_csv.SimaProCSV)
        yield run


@pytest.fixture(scope="module")
def full(parse) -> Codes:
    return parse()


@pytest.mark.integration
class TestTwoParsesOfOneFile:
    def test_the_export_holds_every_kind_of_dataset(self, full):
        kinds = [ds["type"] for ds in full.parsed.data]
        assert {kind: kinds.count(kind) for kind in set(kinds)} == {
            "process": 3,
            "multifunctional": 2,
            "readonly_process": 4,
            "product": 7,
        }
        assert len(full.keys) == len(full.parsed.data)

    def test_same_datasets_and_parameter_rows(self, parse, full):
        again = parse()
        assert again.parsed.data == full.parsed.data
        assert again.parsed.parameters == full.parsed.parameters
        assert full.parsed.parameters

    def test_a_process_identifier_stays_the_code(self, full):
        codes = full.by_name
        assert codes[("process", "Wheat grain, at farm {FR} U")] == "TEST000000000000000001"
        assert codes[("process", "Waste plastic, to landfill {FR} U")] == "TEST000000000000000005"
        with_identifier = [code for code in codes.values() if code.startswith("TEST")]
        assert len(with_identifier) == 3
        derived = [code for code in codes.values() if not code.startswith("TEST")]
        assert len(derived) == 2 + 7 + 4  # processes, products, allocated processes
        assert all(HEX32.match(code) for code in derived)


@pytest.mark.integration
class TestCodesDoNotDependOnTheRestOfTheFile:
    def test_blocks_in_another_order(self, parse, full):
        reordered = parse(ALL_BLOCKS[::-1], "reordered.csv")
        assert reordered.by_name == full.by_name
        assert reordered.parameters == full.parameters

    def test_blocks_removed(self, parse, full):
        """What the ecoinvent strip does: blocks leave, the others keep their codes."""
        fewer = parse(("dairy", "mill"), "stripped.csv")
        assert len(fewer.by_name) == 2 + 4 + 4
        assert {key: full.by_name[key] for key in fewer.by_name} == fewer.by_name
        assert fewer.parameters < full.parameters


@pytest.mark.integration
class TestReferencesResolve:
    def test_inputs_parents_and_reference_processes(self, full):
        keys = full.keys
        for ds in full.parsed.data:
            for label in ("reference process", "mf_parent_key"):
                assert ds.get(label, next(iter(keys))) in keys
            for exc in ds.get("exchanges", []):
                assert exc.get("input", next(iter(keys))) in keys
                child = exc.get("mf_allocated_process_code")
                assert child is None or (ds["database"], child) in keys

    def test_an_allocated_process_produces_the_product_of_its_parent(self, full):
        by_code = {ds["code"]: ds for ds in full.parsed.data}
        allocated = 0
        for parent in full.parsed.data:
            for exc in parent.get("exchanges", []):
                if "mf_allocated_process_code" not in exc:
                    continue
                child = by_code[exc["mf_allocated_process_code"]]
                assert child["type"] == "readonly_process"
                assert child["mf_parent_key"] == (parent["database"], parent["code"])
                functional = [e for e in child["exchanges"] if e.get("functional")]
                assert [e["input"] for e in functional] == [exc["input"]]
                assert by_code[exc["input"][1]]["reference process"] == child["mf_parent_key"]
                allocated += 1
        assert allocated == 4

    def test_parameter_rows_point_at_datasets(self, full):
        codes = {ds["code"] for ds in full.parsed.data}
        rows = full.parsed.parameters
        assert {row["process_code"] for row in rows} <= codes
        parents = {row["mf_parent_code"] for row in rows if "mf_parent_code" in row}
        assert parents == {"TEST000000000000000003"}
        children = {row["process_code"] for row in rows if "mf_parent_code" in row}
        assert len(children) == 2
        assert all(HEX32.match(code) for code in children)
