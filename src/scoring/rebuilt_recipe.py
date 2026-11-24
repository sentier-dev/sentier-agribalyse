"""``RebuiltRecipe`` — how to recompute the rebuilt processes, without their amounts.

:class:`UnitCopyRebuilder` rebuilds the 125 stripped ecoinvent unit blocks
(two of them combustion extracts) from the user's ecoinvent snapshot. Their
columns hold ecoinvent's amounts, so the skeleton zeroes them. They cannot be
refilled by copying an ecoinvent column: the rebuilder routes each input to
ADEME's adapted dataset (with a unit multiplier), to another rebuilt twin, to
a shared-name stand-in, or to ecoinvent, and the product codes of the adapted
datasets are random per parse. Only the package's own row ids name them.

The recipe records that routing where it is known. The rebuilder tags every
exchange it writes with the ecoinvent exchange its amount came from
(:attr:`ATTRIBUTE`); :meth:`from_sp_data` collects the tagged exchanges whose
key reaches the final exchange frame, keyed with the ids of
:class:`ExchangeFrameBuilder`, and :meth:`write` puts them beside the scoring
package. One row per exchange::

    matrix                  'technosphere' | 'biosphere'
    output_id               int64  column id: the rebuilt process
    input_id                int64  row id: the product or elementary flow
    edge_type               str    edge type of the exchange in the package
    sign                    f64    +1 production, substitution and biosphere;
                                   -1 technosphere (TechnosphereBuilder's rule)
    multiplier              f64    unit conversion the rebuilder applied, 1.0 otherwise
    source_output_code      str    the ecoinvent exchange the amount comes from:
    source_input_database   str      activity code, input database, input code
    source_input_code       str      and type, as in the snapshot
    source_type             str

The value of an entry is ``(amount × multiplier) × sign``, where ``amount`` is
the source exchange's amount in the snapshot after
:meth:`EcoinventExchangesIngester._normalise_waste_signs` over the whole
snapshot, the normalisation the rebuilder read it through. Entries on the same
matrix cell add up, as they do in the matrix builders. The recipe holds no
amount: a licensed user recomputes every rebuilt column from their own
exchanges.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from scoring.ecoinvent_exchanges import EcoinventExchangesIngester
from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder

_POSITIVE_EDGE_TYPES = frozenset({"production", "generic production", "substitution"})
_BIOSPHERE = "biosphere"


@dataclass(frozen=True)
class RebuiltRecipe:
    """The amount-free recipe of every rebuilt column of one scoring package."""

    FILENAME: ClassVar[str] = "rebuilt_recipe.parquet"
    ATTRIBUTE: ClassVar[str] = "ecoinvent_source"
    """Exchange attribute :class:`UnitCopyRebuilder` writes: ``output_code``,
    ``input_database``, ``input_code``, ``type`` of the ecoinvent exchange the
    amount came from, and the ``multiplier`` applied to it."""
    SOURCE_KEY: ClassVar[tuple[str, ...]] = (
        "source_output_code",
        "source_input_database",
        "source_input_code",
        "source_type",
    )
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "matrix",
        "output_id",
        "input_id",
        "edge_type",
        "sign",
        "multiplier",
        *SOURCE_KEY,
    )

    frame: pd.DataFrame

    @classmethod
    def from_sp_data(cls, sp_data: Iterable[dict], final: ExchangeFrame) -> RebuiltRecipe:
        """The tagged exchanges of ``sp_data`` whose key reaches ``final``.

        ``final`` is the exchange frame the package was built from, after
        allocation, deduplication and pruning. The filter is by key,
        ``(output_id, input_id, edge_type)``: a tagged exchange stays when
        ``final`` holds an edge with its key. An exchange those steps dropped
        (a cancelling self-loop, a draw on a product nobody produces) leaves
        the recipe unless another edge of the same process kept the same
        key; such a case would show as a difference in the extractor's check
        against the package, which refuses it."""
        rows: list[dict] = []
        for ds in sp_data:
            output_id = ExchangeFrameBuilder.flow_id_for(
                (ds.get("database") or "", ds.get("code") or "")
            )
            for exc in ds.get("exchanges", ()) or ():
                source = exc.get(cls.ATTRIBUTE)
                if source is None or exc.get("input") is None:
                    continue
                edge_type = str(exc.get("type") or "technosphere")
                rows.append(
                    {
                        "matrix": _BIOSPHERE if edge_type == _BIOSPHERE else "technosphere",
                        "output_id": output_id,
                        "input_id": ExchangeFrameBuilder.flow_id_for(tuple(exc["input"])),
                        "edge_type": edge_type,
                        "sign": cls.sign_for(edge_type),
                        "multiplier": float(source.get("multiplier", 1.0)),
                        "source_output_code": str(source["output_code"]),
                        "source_input_database": str(source["input_database"]),
                        "source_input_code": str(source["input_code"]),
                        "source_type": str(source["type"]),
                    }
                )
        recipe = cls._typed(pd.DataFrame(rows, columns=list(cls.COLUMNS)))
        keys = ["output_id", "input_id", "edge_type"]
        present = final.df[keys].astype({"edge_type": "string"}).drop_duplicates()
        kept = recipe.merge(present, on=keys, how="inner")
        return cls(frame=kept[list(cls.COLUMNS)].reset_index(drop=True))

    @classmethod
    def read(cls, package_dir: Path) -> RebuiltRecipe | None:
        path = package_dir / cls.FILENAME
        if not path.exists():
            return None
        return cls(frame=cls._typed(pd.read_parquet(path)))

    def write(self, package_dir: Path) -> Path:
        """Write the recipe into ``package_dir``, replacing an older one atomically."""
        target = package_dir / self.FILENAME
        partial = package_dir / (self.FILENAME + ".partial")
        self.frame.to_parquet(partial, engine="pyarrow", compression="snappy", index=False)
        partial.replace(target)
        return target

    @staticmethod
    def sign_for(edge_type: str) -> float:
        if edge_type == _BIOSPHERE or edge_type in _POSITIVE_EDGE_TYPES:
            return 1.0
        return -1.0

    @classmethod
    def entry_values(cls, recipe: pd.DataFrame, exchanges: pd.DataFrame) -> np.ndarray:
        """``(amount × multiplier) × sign`` for each row of ``recipe``.

        ``exchanges`` is the raw snapshot (``output_database, output_code,
        input_database, input_code, amount, type``). Raises when a source
        exchange is missing from it or is not one exchange."""
        normalised = EcoinventExchangesIngester._normalise_waste_signs(exchanges)
        key = ["output_code", "input_database", "input_code", "type"]
        table = normalised[[*key, "amount"]].astype({c: "string" for c in key})
        table.columns = [*cls.SOURCE_KEY, "amount"]
        wanted = recipe[list(cls.SOURCE_KEY)].drop_duplicates()
        table = table.merge(wanted, on=list(cls.SOURCE_KEY), how="inner")
        repeated = table.duplicated(subset=list(cls.SOURCE_KEY), keep=False)
        if repeated.any():
            raise ValueError(
                f"{int(repeated.sum())} snapshot rows share the key of a recipe source; "
                "the recipe must point at one exchange each."
            )
        joined = recipe[list(cls.SOURCE_KEY)].merge(table, on=list(cls.SOURCE_KEY), how="left")
        missing = joined["amount"].isna()
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} recipe sources are not in the ecoinvent snapshot. "
                "The package was built from another snapshot."
            )
        amount = joined["amount"].to_numpy(dtype="float64")
        multiplier = recipe["multiplier"].to_numpy(dtype="float64")
        sign = recipe["sign"].to_numpy(dtype="float64")
        return (amount * multiplier) * sign

    @classmethod
    def _typed(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["output_id"] = out["output_id"].astype("int64")
        out["input_id"] = out["input_id"].astype("int64")
        out["sign"] = out["sign"].astype("float64")
        out["multiplier"] = out["multiplier"].astype("float64")
        for col in ("matrix", "edge_type", *cls.SOURCE_KEY):
            out[col] = out[col].astype("string")
        return out
