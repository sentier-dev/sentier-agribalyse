"""``SimaProCsvParser`` + ``ParsedSimaProCsv`` — bw2-free SimaPro wrapper.

bw2io's ``SimaProBlockCSVImporter`` did two things bundled together:
parsed the SimaPro CSV (via ``bw_simapro_csv``) and exposed a
``LCIImporter``-shaped surface (``apply_strategy``, ``match_database``,
etc.). REFACTOR_FINAL strips it down: ``bw_simapro_csv`` does the parse
directly and the wrapper exposes only the methods the linker actually
uses, all of them dispatching to in-house ``transforms.strategies``
classes — no bw2io, no bw2data.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import randonneur as rn
from bw_simapro_csv import SimaProCSV

from core.logging import Logging
from transforms.parameter_extraction import ProcessParameterExtractor
from transforms.strategies.biosphere import DropUnspecifiedSubcategories
from transforms.strategies.internal import (
    DropUnlinkedExchanges,
    LinkIterableByFields,
    SetMetadataUsingSingleFunctionalExchange,
    SplitSimaproNameGeo,
)

# Lifted from ``bw2data.labels`` — small string lists, no need to import
# the module just for these.
_PROCESS_NODE_TYPES: tuple[str, ...] = (
    "process",
    "processwithreferenceproduct",
    "readonly_process",
)
_PRODUCT_NODE_TYPES: tuple[str, ...] = ("product",)
_MULTIFUNCTIONAL_NODE_DEFAULT: str = "multifunctional"


@dataclass(frozen=True)
class SimaProCsvParser:
    """Parse AGB SimaPro CSV directly via ``bw_simapro_csv``."""

    csv_path: Path
    database_name: str | None = None
    separate_products: bool = True
    shorten_names: bool = True

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def parse(self) -> ParsedSimaProCsv:
        spcsv = SimaProCSV(
            path_or_stream=self.csv_path,
            database_name=self.database_name,
            stderr_logs=False,
            write_logs=False,
        )
        bw_data = spcsv.to_brightway(
            separate_products=self.separate_products,
            shorten_names=self.shorten_names,
        )
        data: list[dict] = list(bw_data.get("processes", []))
        if self.separate_products:
            data.extend(bw_data.get("products", []))
        parameters = ProcessParameterExtractor().extract(
            spcsv, bw_processes=list(bw_data.get("processes", []))
        )
        self._log.info(
            "csv.parser.parsed",
            n_processes=len(bw_data.get("processes", [])),
            n_products=len(bw_data.get("products", [])),
            n_parameters=len(parameters),
            db=spcsv.database_name,
        )
        return ParsedSimaProCsv(
            db_name=spcsv.database_name,
            data=data,
            metadata=dict(bw_data.get("database", {})),
            parameters=parameters,
        )


class ParsedSimaProCsv:
    """In-house replacement for ``bw2io.SimaProBlockCSVImporter``'s surface.

    Holds ``.data`` (list[dict]) and exposes the strategy-application,
    internal-link, randonneur, and drop-unlinked methods the linker calls.
    Every method dispatches to a pure-Python class under
    ``transforms.strategies``; no bw2io / bw2data imports remain.
    """

    def __init__(
        self,
        db_name: str,
        data: list[dict],
        metadata: dict | None = None,
        parameters: list[dict] | None = None,
    ):
        self.db_name = db_name
        self.data = data
        self.metadata = dict(metadata or {})
        # Process-local parameter definitions (see ``ProcessParameterExtractor``).
        # Pickles written before this field existed lack the attribute — the
        # importer treats those caches as stale and re-parses.
        self.parameters = list(parameters or [])
        # Default chain — same set bw2io used in
        # ``SimaProBlockCSVImporter.__init__``, all lifted in-house.
        self.strategies: list[Callable] = [
            SetMetadataUsingSingleFunctionalExchange(),
            DropUnspecifiedSubcategories(),
            SplitSimaproNameGeo(),
        ]

    # ------------------------------------------------------------------
    # Strategy application — pure dict transforms.

    def apply_strategy(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        self.data = fn(self.data, *args, **kwargs)

    def apply_strategies(self) -> None:
        for s in self.strategies:
            self.data = s(self.data)

    # ------------------------------------------------------------------
    # match_database — internal-only after F4.

    def match_database(
        self,
        db_name: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        kind: str | list[str] | None = None,
        processes_to_products: bool = False,
        relink: bool = False,
    ) -> None:
        if db_name is not None:
            raise NotImplementedError(
                f"External match_database({db_name!r}) is no longer supported. "
                f"Use EcoinventCatalog or BiosphereCatalog directly."
            )
        edge_kinds: tuple[str, ...] | None = None
        if isinstance(kind, str):
            edge_kinds = (kind,)
        elif kind is not None:
            edge_kinds = tuple(kind)

        this_kinds: tuple[str, ...] | None = None
        other_kinds: tuple[str, ...] | None = None
        if processes_to_products:
            this_kinds = (*_PROCESS_NODE_TYPES, _MULTIFUNCTIONAL_NODE_DEFAULT)
            other_kinds = _PRODUCT_NODE_TYPES

        linker = LinkIterableByFields(
            fields=tuple(fields) if fields else None,
            edge_kinds=edge_kinds,
            this_node_kinds=this_kinds,
            other_node_kinds=other_kinds,
            internal=True,
            relink=relink,
        )
        self.data = linker.apply(self.data)

    # ------------------------------------------------------------------
    # randonneur — apply stored datapackages from randonneur_data.

    def randonneur(
        self,
        label: str | None = None,
        datapackage: object | None = None,
        fields: list[str] | None = None,
        node_filter: Callable | None = None,
        edge_filter: Callable | None = None,
        case_sensitive: bool = False,
        verbs: list[str] | None = None,
        migrate_edges: bool = True,
        migrate_nodes: bool = False,
    ) -> None:
        """Apply a randonneur transformation to ``self.data``.

        Two mutually-exclusive sources are supported (matching the
        ``bw2io.SimaProBlockCSVImporter.randonneur`` surface so callers
        like ``BiosphereFlowmapApplier`` work against either parser):

        * ``label="<datapackage-name>"`` — pull a stored datapackage by
          name from the registered randonneur registry.
        * ``datapackage=<rn.Datapackage>`` — apply an in-memory
          ``Datapackage`` instance (typically loaded from a custom
          JSON via ``rn.Datapackage.from_json``).

        Pass exactly one. ``datapackage`` takes precedence when both
        are supplied.
        """
        chosen_verbs = verbs if verbs is not None else list(rn.utils.SAFE_VERBS)
        cfg_edges = rn.MigrationConfig(
            fields=fields,
            node_filter=node_filter,
            edge_filter=edge_filter,
            edges_label="exchanges",
            case_sensitive=case_sensitive,
            verbs=chosen_verbs,
        )
        cfg_nodes = rn.MigrationConfig(
            fields=fields,
            node_filter=node_filter,
            edges_label="exchanges",
            case_sensitive=case_sensitive,
            verbs=chosen_verbs,
        )
        if datapackage is not None:
            migrations = datapackage.data if hasattr(datapackage, "data") else datapackage
            if migrate_edges:
                with contextlib.suppress(rn.errors.WrongGraphContext):
                    self.data = rn.migrate_edges(
                        graph=self.data, migrations=migrations, config=cfg_edges
                    )
            if migrate_nodes:
                with contextlib.suppress(rn.errors.WrongGraphContext):
                    self.data = rn.migrate_nodes(
                        graph=self.data, migrations=migrations, config=cfg_nodes
                    )
            return
        if migrate_edges:
            with contextlib.suppress(rn.errors.WrongGraphContext):
                self.data = rn.migrate_edges_with_stored_data(
                    graph=self.data,
                    label=label,
                    config=cfg_edges,
                )
        if migrate_nodes:
            with contextlib.suppress(rn.errors.WrongGraphContext):
                self.data = rn.migrate_nodes_with_stored_data(
                    graph=self.data,
                    label=label,
                    config=cfg_nodes,
                )

    # ------------------------------------------------------------------
    # Other surface bits — pure dict transforms.

    def drop_unlinked(self, *, i_am_reckless: bool = False) -> None:
        if not i_am_reckless:
            raise ValueError(
                "drop_unlinked must be called with i_am_reckless=True — "
                "this strategy removes unlinked exchanges and is non-reversible."
            )
        self.data = DropUnlinkedExchanges()(self.data)
