"""Unit tests for ``registry.schema`` — DataFrame column ordering + coercion."""

from __future__ import annotations

from domain import (
    Bucket,
    ContextNorm,
    Deletion,
    EdgeLabelCorrection,
    EfTargetIndexRow,
    Mapping,
    SourceKind,
    Tier,
    UnitAlias,
    UnitConversion,
)
from registry.schema import (
    ContextNormTable,
    DeletionTable,
    EdgeLabelTable,
    EfTargetIndexTable,
    MappingTable,
    UnitAliasTable,
    UnitConversionTable,
)


class TestMappingTable:
    def test_empty_returns_dataframe_with_correct_columns(self):
        df = MappingTable().to_dataframe([])
        assert df.empty
        assert list(df.columns) == list(MappingTable().columns)

    def test_round_trip_preserves_field_values(self):
        m = Mapping(
            source_kind=SourceKind.AGB_FLOW,
            source_name="X",
            source_unit="kg",
            source_top_bucket=Bucket.AIR,
            target_db="biosphere3",
            target_code="abc",
            priority_tier=Tier.CURATED_TARGETED,
        )
        df = MappingTable().to_dataframe([m])
        row = df.iloc[0]
        assert row["source_kind"] == "agb_flow"
        assert row["source_top_bucket"] == "air"
        assert row["priority_tier"] == 1
        assert row["target_db"] == "biosphere3"


class TestUnitConversionTable:
    def test_empty_dataframe_keeps_column_order(self):
        df = UnitConversionTable().to_dataframe([])
        assert list(df.columns) == ["source_unit", "target_unit", "multiplier", "provenance"]

    def test_emits_one_row_per_conversion(self):
        rows = [
            UnitConversion(source_unit="m", target_unit="km", multiplier=0.001),
            UnitConversion(source_unit="g", target_unit="kg", multiplier=0.001),
        ]
        df = UnitConversionTable().to_dataframe(rows)
        assert df.shape == (2, 4)


class TestUnitAliasTable:
    def test_alias_lower_is_added_at_materialisation(self):
        df = UnitAliasTable().to_dataframe([UnitAlias(alias="A", canonical="year")])
        assert df.iloc[0]["alias_lower"] == "a"
        assert df.iloc[0]["canonical"] == "year"


class TestSimpleTables:
    def test_context_norm_table(self):
        df = ContextNormTable().to_dataframe(
            [ContextNorm(source_context=("emissions to air",), target_context=("air",))]
        )
        assert df.iloc[0]["source_context"] == ["emissions to air"]
        assert df.iloc[0]["target_context"] == ["air"]

    def test_deletion_table(self):
        df = DeletionTable().to_dataframe([Deletion(name="X", code="c", kind="process")])
        assert df.iloc[0]["kind"] == "process"
        assert df.iloc[0]["code"] == "c"

    def test_edge_label_table_serialises_categories_as_list(self):
        df = EdgeLabelTable().to_dataframe(
            [
                EdgeLabelCorrection(
                    source_name="old",
                    target_name="new",
                    edge_type="technosphere",
                    categories=("a", "b"),
                )
            ]
        )
        assert df.iloc[0]["categories"] == ["a", "b"]

    def test_ef_target_index_lowercase_name_column(self):
        df = EfTargetIndexTable().to_dataframe(
            [
                EfTargetIndexRow(
                    code="u1",
                    name="  Carbon Dioxide  ",
                    unit="kg",
                    bucket=Bucket.AIR,
                )
            ]
        )
        assert df.iloc[0]["name_lower"] == "carbon dioxide"
        assert df.iloc[0]["bucket"] == "air"
