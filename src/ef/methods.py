"""``EfMethodFilter`` — predicate helpers for EF v3.1 method tuples.

``bw2data.methods`` mixes EF, IPCC, ReCiPe, and other LCIA methods. The EF
flow registry and the per-method CF registry both need the same "is this
an EF v3.1 method?" check; keeping the predicate in one place stops the
two callers from drifting (different shape checks → different filtered
sets → silently divergent parquets).
"""

from __future__ import annotations

from typing import Any


class EfMethodFilter:
    """Stateless EF v3.1 method-tuple predicates.

    The class form keeps the OOP rule satisfied — a free helper module
    would not. Both consumers (``EfFlowsRegistryBuilder``,
    ``MethodCfRegistryBuilder``) call ``EfMethodFilter.is_ef_v31(m_key)``.
    """

    EF_VERSION_TAG: str = "EF v3.1"

    @classmethod
    def is_ef_v31(cls, m_key: Any) -> bool:
        """True iff ``m_key`` is the canonical 4-tuple shape with ``EF v3.1`` second."""
        return isinstance(m_key, tuple) and len(m_key) == 4 and m_key[1] == cls.EF_VERSION_TAG
