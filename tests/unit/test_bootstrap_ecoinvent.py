"""``dds-bootstrap-ecoinvent`` — one command from credentials to the four snapshots."""

from __future__ import annotations

import json
import pickle
import sqlite3
import types

import pandas as pd
import pytest

from bootstrap import (
    EcoinventBootstrapper,
    EcoinventCredentialsMissingError,
    EcoinventMetadataSnapshotter,
    EcoinventReleaseImporter,
)

EI = "ecoinvent-3.9.1-cutoff"
BIO = "ecoinvent-3.9.1-biosphere"


class FakeBw2Data:
    """The slice of ``bw2data`` the bootstrap touches."""

    def __init__(self, project_dir, *, node_ids: bool = True, loaded: bool = True):
        self._project_dir = project_dir
        self.current = None
        self.databases = {EI: {}, BIO: {}} if loaded else {}
        self.projects = types.SimpleNamespace(set_current=self._set_current, dir=project_dir)
        self._activities = [
            {
                "code": "b",
                "name": "wheat production",
                "unit": "kilogram",
                "location": "FR",
                "reference product": "wheat grain",
            },
            {
                "code": "a",
                "name": "market for diesel",
                "unit": "kilogram",
                "location": "RER",
                "reference product": "diesel",
            },
        ]
        self._flows = [
            {
                "code": "f2",
                "name": "Methane, fossil",
                "categories": ("air",),
                "unit": "kilogram",
                "CAS number": "000074-82-8",
                "synonyms": ["CH4"],
            },
            {
                "code": "f1",
                "name": "Carbon dioxide, fossil",
                "categories": ("air", "urban air close to ground"),
                "unit": "kilogram",
            },
        ]
        self._nodes = {11: {"database": BIO, "code": "f1"}, 12: {"database": BIO, "code": "f2"}}
        flow = (lambda i, code: i) if node_ids else (lambda i, code: (BIO, code))
        cfs = [(flow(12, "f2"), 29.8), (flow(11, "f1"), 1.0)]
        self.methods = {
            (
                "ecoinvent-3.9.1",
                "EF v3.1",
                "climate change",
                "global warming potential (GWP100)",
            ): cfs,
            ("ecoinvent-3.9.1", "EF v3.1 no LT", "climate change", "x"): cfs,
            ("ecoinvent-3.9.1", "ReCiPe", "climate change", "x"): cfs,
        }

    def _set_current(self, name):
        self.current = name

    def Database(self, name):
        return {EI: self._activities, BIO: self._flows}[name]

    def Method(self, key):
        return types.SimpleNamespace(load=lambda: self.methods[key])

    def get_node(self, *, id):
        return self._nodes[id]


class FakeBw2Io:
    def __init__(self, bd: FakeBw2Data):
        self.bd = bd
        self.calls: list[dict] = []

    def import_ecoinvent_release(self, **kwargs):
        self.calls.append(kwargs)
        self.bd.databases = {EI: {}, BIO: {}}


def _sqlite(path, rows=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE activitydataset (database TEXT, code TEXT)")
        conn.execute(
            "CREATE TABLE exchangedataset (output_database TEXT, output_code TEXT, "
            "input_database TEXT, input_code TEXT, type TEXT, data BLOB)"
        )
        conn.execute("INSERT INTO activitydataset VALUES (?, ?)", (EI, "a"))
        for row in rows if rows is not None else [(EI, "a", EI, "a", "production", 1.0)]:
            conn.execute(
                "INSERT INTO exchangedataset VALUES (?, ?, ?, ?, ?, ?)",
                (*row[:5], pickle.dumps({"amount": row[5]})),
            )


# ----------------------------------------------------------------------


class TestMetadataSnapshotter:
    def test_activities_are_written_sorted_in_the_shape_the_catalog_reads(self, settings, tmp_path):
        bd = FakeBw2Data(tmp_path)

        counts = EcoinventMetadataSnapshotter(settings=settings, bd=bd).snapshot()

        payload = json.loads(settings.paths.ecoinvent_activities_json.read_text())
        assert payload["db_name"] == EI
        assert payload["activities"] == [
            {
                "code": "a",
                "name": "market for diesel",
                "unit": "kilogram",
                "location": "RER",
                "reference_product": "diesel",
            },
            {
                "code": "b",
                "name": "wheat production",
                "unit": "kilogram",
                "location": "FR",
                "reference_product": "wheat grain",
            },
        ]
        assert counts["activities"] == 2

    def test_biosphere_flows(self, settings, tmp_path):
        EcoinventMetadataSnapshotter(settings=settings, bd=FakeBw2Data(tmp_path)).snapshot()

        flows = json.loads(settings.paths.ecoinvent_biosphere_flows_json.read_text())["flows"]
        assert flows == [
            {
                "code": "f1",
                "name": "Carbon dioxide, fossil",
                "categories": ["air", "urban air close to ground"],
                "unit": "kilogram",
                "cas": None,
                "synonyms": [],
            },
            {
                "code": "f2",
                "name": "Methane, fossil",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": "000074-82-8",
                "synonyms": ["CH4"],
            },
        ]

    @pytest.mark.parametrize("node_ids", [True, False])
    def test_methods_keep_only_ef_and_resolve_node_ids(self, settings, tmp_path, node_ids):
        """bw2data 4 keys method factors by node id, bw2data 3 by (database, code)."""
        bd = FakeBw2Data(tmp_path, node_ids=node_ids)

        counts = EcoinventMetadataSnapshotter(settings=settings, bd=bd).snapshot()

        payload = json.loads(settings.paths.ef_methods_snapshot_json.read_text())
        label = "ecoinvent-3.9.1 / EF v3.1 / climate change / global warming potential (GWP100)"
        assert payload["ef_db_name"] == "ef"
        assert list(payload["methods"]) == [label]
        assert payload["methods"][label] == {
            "key": [
                "ecoinvent-3.9.1",
                "EF v3.1",
                "climate change",
                "global warming potential (GWP100)",
            ],
            "inherited_cfs": [
                {"db": BIO, "code": "f1", "amount": 1.0},
                {"db": BIO, "code": "f2", "amount": 29.8},
            ],
        }
        assert counts == {"activities": 2, "biosphere_flows": 2, "methods": 1, "method_cfs": 2}

    def test_no_ef_method_is_an_error(self, settings, tmp_path):
        bd = FakeBw2Data(tmp_path)
        bd.methods = {}

        with pytest.raises(RuntimeError, match=r"EF v3\.1"):
            EcoinventMetadataSnapshotter(settings=settings, bd=bd).snapshot()


class TestReleaseImporter:
    def test_imports_when_the_database_is_missing(self, settings, tmp_path):
        bd = FakeBw2Data(tmp_path, loaded=False)
        bi = FakeBw2Io(bd)

        imported = EcoinventReleaseImporter(settings=settings, bd=bd, bi=bi).ensure(
            "user", "secret"
        )

        assert imported is True
        assert bd.current == "dds-bootstrap"
        assert bi.calls == [
            {
                "version": "3.9.1",
                "system_model": "cutoff",
                "username": "user",
                "password": "secret",
                "use_mp": False,
            }
        ]

    def test_skips_the_download_when_the_database_is_there(self, settings, tmp_path):
        bd = FakeBw2Data(tmp_path)
        bi = FakeBw2Io(bd)

        assert EcoinventReleaseImporter(settings=settings, bd=bd, bi=bi).ensure(None, None) is False
        assert bi.calls == []

    @pytest.mark.parametrize("user,password", [(None, "x"), ("u", None), ("", "")])
    def test_missing_credentials_are_reported_before_any_download(
        self, settings, tmp_path, user, password
    ):
        bd = FakeBw2Data(tmp_path, loaded=False)
        bi = FakeBw2Io(bd)

        with pytest.raises(EcoinventCredentialsMissingError, match="ECOINVENT_USERNAME"):
            EcoinventReleaseImporter(settings=settings, bd=bd, bi=bi).ensure(user, password)
        assert bi.calls == []


class TestBootstrapper:
    def test_writes_the_four_snapshots(self, settings, tmp_path):
        project = tmp_path / "bw" / "dds-bootstrap.abc"
        _sqlite(project / "lci" / "databases.db")
        bd = FakeBw2Data(project, loaded=False)
        bi = FakeBw2Io(bd)

        report = EcoinventBootstrapper(settings=settings, bd=bd, bi=bi).run("user", "secret")

        paths = settings.paths
        assert report["imported"] is True
        assert report["exchanges"] == 1
        assert len(pd.read_parquet(paths.ecoinvent_exchanges_parquet)) == 1
        for path in (
            paths.ecoinvent_activities_json,
            paths.ecoinvent_biosphere_flows_json,
            paths.ef_methods_snapshot_json,
        ):
            assert path.exists()

    def test_second_run_reuses_the_project(self, settings, tmp_path):
        project = tmp_path / "bw" / "dds-bootstrap.abc"
        _sqlite(project / "lci" / "databases.db")
        bd = FakeBw2Data(project)
        bi = FakeBw2Io(bd)

        report = EcoinventBootstrapper(settings=settings, bd=bd, bi=bi).run(None, None)

        assert report["imported"] is False
        assert bi.calls == []


class TestEmptyExchangeSnapshot:
    def test_database_without_exchanges_is_an_error_not_an_empty_file(self, settings, tmp_path):
        from cli.snapshot_ecoinvent_exchanges import (
            EcoinventExchangesSnapshotter,
            EcoinventSqliteReader,
            EmptyEcoinventSnapshotError,
        )

        db = tmp_path / "databases.db"
        _sqlite(db, rows=[])
        snapshotter = EcoinventExchangesSnapshotter(
            reader=EcoinventSqliteReader(sqlite_path=db, db_name=EI),
            output_path=settings.paths.ecoinvent_exchanges_parquet,
        )

        with pytest.raises(EmptyEcoinventSnapshotError, match="another project"):
            snapshotter.snapshot()
        assert not settings.paths.ecoinvent_exchanges_parquet.exists()
