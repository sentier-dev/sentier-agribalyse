"""Unit-related strategies: electricity unit fix, generic exchange rescale.

Lifted from ``bw2io.strategies`` (``change_electricity_unit_mj_to_kwh``)
and ``bw2io.utils.rescale_exchange``. Pure dict transforms — no bw2data
or bw2io dependency. The ``stats_arrays`` ids the rescaler branches on
are stable upstream and well-documented; we match them by integer
constant rather than re-importing the enum to keep the dependency graph
trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number

# stats_arrays uncertainty type ids — stable upstream constants. We pin
# them by value to avoid the runtime dependency on stats_arrays for what
# is effectively three small integers.
_UNCERTAINTY_UNDEFINED = 0
_UNCERTAINTY_NO = 1
_UNCERTAINTY_LOGNORMAL = 2
_UNCERTAINTY_NORMAL = 3
_UNCERTAINTY_UNIFORM = 4
_UNCERTAINTY_TRIANGULAR = 5


class RescaleExchange:
    """Rescale ``exc["amount"]`` (and uncertainty fields) by a constant factor.

    Direct lift of ``bw2io.utils.rescale_exchange`` with the
    ``stats_arrays`` enum imports flattened to integer constants. The
    behaviour is preserved: zero factor zeroes the loc/amount and clears
    scale/shape; sign-flip on triangular/uniform swaps min/max bounds;
    formulas get wrapped with the multiplier.
    """

    @staticmethod
    def apply(exc: dict, factor: float) -> dict:
        if not isinstance(factor, Number) or factor is True or factor is False:
            raise ValueError(f"`factor` must be a number, got {type(factor)}")

        if factor == 0:
            exc.update(
                {
                    "uncertainty type": _UNCERTAINTY_UNDEFINED,
                    "loc": exc["amount"] * factor,
                    "amount": exc["amount"] * factor,
                }
            )
            for field in ("scale", "shape", "minimum", "maximum", "negative"):
                exc.pop(field, None)
            return exc

        if exc.get("formula"):
            exc["formula"] = f"({exc['formula']}) * {factor}"

        ut = exc.get("uncertainty type", 0)
        if ut in (_UNCERTAINTY_UNDEFINED, _UNCERTAINTY_NO):
            exc["amount"] = exc["loc"] = factor * exc["amount"]
        elif ut == _UNCERTAINTY_NORMAL:
            exc.update(
                {
                    "scale": abs(exc["scale"] * factor),
                    "loc": exc["amount"] * factor,
                    "amount": exc["amount"] * factor,
                }
            )
        elif ut == _UNCERTAINTY_LOGNORMAL:
            exc.update(
                {
                    "loc": math.log(abs(exc["amount"] * factor)),
                    "negative": (exc["amount"] * factor) < 0,
                    "amount": exc["amount"] * factor,
                }
            )
        elif ut == _UNCERTAINTY_UNIFORM:
            exc["minimum"] *= factor
            exc["maximum"] *= factor
            if "amount" in exc:
                exc["amount"] = exc["loc"] = factor * exc["amount"]
            else:
                exc["amount"] = exc["loc"] = (exc["minimum"] + exc["maximum"]) / 2
        elif ut == _UNCERTAINTY_TRIANGULAR:
            exc["minimum"] *= factor
            exc["maximum"] *= factor
            exc["amount"] = exc["loc"] = factor * exc["amount"]
        else:
            raise ValueError(f"Cannot rescale exchange with uncertainty type {ut!r}")

        if exc.get("uncertainty type") != _UNCERTAINTY_LOGNORMAL and "negative" in exc:
            del exc["negative"]

        if exc.get("uncertainty type") not in (
            _UNCERTAINTY_TRIANGULAR,
            _UNCERTAINTY_UNIFORM,
        ):
            for field in ("minimum", "maximum"):
                if field in exc:
                    exc[field] *= factor

        if factor < 0 and "minimum" in exc and "maximum" in exc:
            exc["minimum"], exc["maximum"] = exc["maximum"], exc["minimum"]
        elif factor < 0 and "minimum" in exc:
            exc["maximum"] = exc.pop("minimum")
        elif factor < 0 and "maximum" in exc:
            exc["minimum"] = exc.pop("maximum")

        return exc


@dataclass(frozen=True)
class ChangeElectricityUnitMjToKwh:
    """Change electricity exchanges from MJ to kWh, rescaling amount by 1/3.6.

    Lifted from ``bw2io.strategies.change_electricity_unit_mj_to_kwh``.
    Matches when the exchange name starts with ``electricity``,
    ``market for electricity``, or ``market group for electricity`` and
    the unit is ``megajoule``.
    """

    name: str = "change_electricity_unit_mj_to_kwh"

    _NAME_PREFIXES: tuple[str, ...] = (
        "electricity",
        "market for electricity",
        "market group for electricity",
    )
    _MJ_TO_KWH: float = 1.0 / 3.6

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            for exc in ds.get("exchanges", []):
                if not self._matches(exc):
                    continue
                exc["unit"] = "kilowatt hour"
                RescaleExchange.apply(exc, self._MJ_TO_KWH)
        return data

    @classmethod
    def _matches(cls, exc: dict) -> bool:
        n = (exc.get("name") or "").lower()
        return exc.get("unit") == "megajoule" and any(n.startswith(p) for p in cls._NAME_PREFIXES)
