"""``MethodSlug`` — filesystem-safe encoding for method tuples.

Used by both ``ScoringPackageStore`` and ``MethodCfRegistryBuilder`` so a
method tuple resolves to the same on-disk slug everywhere. URL-quoting
each component keeps every character round-trippable: spaces, parens,
slashes — anything bw2data hands us — survives a path-write/read cycle.

The encoding is ``urllib.parse.quote`` with ``safe=""`` per component
**and an explicit underscore escape**, joined by a literal ``"__"``.
Without the underscore escape, a component containing a literal ``__``
would be indistinguishable from the separator at decode time and
``decode(encode(tuple))`` would return the wrong arity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import quote, unquote


@dataclass(frozen=True)
class MethodSlug:
    """Stateless encoder/decoder for method tuples.

    Both methods are ``classmethod`` because there is no instance state;
    the class form keeps callers consistent (``MethodSlug.encode(t)``)
    and the OOP rule satisfied — a free helper module would not.
    """

    SEPARATOR: ClassVar[str] = "__"

    @classmethod
    def encode(cls, method: tuple[str, ...]) -> str:
        """URL-quote each tuple component (including ``_``) and join with ``__``."""
        return cls.SEPARATOR.join(cls._quote_part(part) for part in method)

    @classmethod
    def decode(cls, slug: str) -> tuple[str, ...]:
        """Inverse of :meth:`encode` — split on the separator + unquote."""
        if not slug:
            return ()
        return tuple(unquote(part) for part in slug.split(cls.SEPARATOR))

    @staticmethod
    def _quote_part(part: str) -> str:
        # ``urllib.parse.quote(safe="")`` preserves underscore (it's RFC-3986
        # "unreserved"). Since ``__`` is our separator, every literal underscore
        # in the part must be escaped to ``%5F`` so the round-trip cannot drift.
        return quote(part, safe="").replace("_", "%5F")
