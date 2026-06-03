"""``ActivityLocationParser`` — extract an ISO/region code from an activity name.

AGB SimaPro datasets generally don't carry the brightway ``location`` field
on the dataset metadata (2 133 of 43 542 do; the rest are ``None``), but
the location IS encoded in the activity name via two recognisable
suffixes:

* ``{XX}`` — explicit, sometimes followed by ``U``, ``[Ciqual code: …]``,
  pack/sales channel labels. Example::

      "Frozen mango puree, at processing {BR} U"

* trailing ``XX`` — a 2-to-5-char uppercase token separated from the rest
  by whitespace. This form is the ecoinvent-imported activities that
  AGB took over verbatim. Example::

      "mango production BR"
      "market for sugar, from sugar beet GLO"

Parser priority is ``{XX}`` first (most reliable) then trailing-token.
Sub-regional codes with a dash (``CA-QC``, ``US-WECC``, ``BR-MG``) are
collapsed to the country prefix so they hit the JRC AWARE table's
country-level row. Aggregated regions that have no national equivalent
(``RoW``, ``GLO``, ``RER``, ``RNA``, ``RLA``, ``OECD``, ``Europe``,
``World``) deliberately don't match any regional CF — the scorer falls
back to the global CF for them, which is the correct behaviour (a
location-free activity gets the global mean).

The parser is the difference between 2 077 corrected activity columns
(ecoinvent-catalog-only fallback) and the full ~14 000 columns that
actually carry a regional water emission upstream of every AGB product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ActivityLocationParser:
    """Stateless extractor — instantiate once, call ``location_for`` per name."""

    # ``{XX}`` with optional trailing decorations like ``U``,
    # ``[Ciqual code: 12345]``, sales-channel labels. We accept any
    # uppercase letter / digit / hyphen sequence between braces and
    # require at least 2 chars to avoid matching the ``{}`` placeholders
    # that sometimes appear empty.
    BRACE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\{(?P<loc>[A-Z][A-Z0-9-]{1,9})\}")

    # Trailing uppercase token, separated by whitespace (or following a
    # comma), optionally followed by ``U``/``u``. Anchored to the end
    # of the string so middle-of-name tokens (e.g. ``EU mix``) don't
    # match. Length 2-5 chars to avoid catching single-letter chemical
    # symbols like ``N`` / ``P`` / ``K``.
    TRAILING_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"[,\s](?P<loc>[A-Z][A-Z0-9]{1,4}(?:-[A-Z0-9-]+)?)\s*$"
    )

    # Region codes that exist in the JRC AWARE table as ISO countries
    # but appear inside dashed compound codes in AGB. We strip to the
    # country prefix so ``CA-QC`` falls back to ``CA``. Codes whose
    # prefix is itself an aggregate (e.g. ``RER-XX``) wouldn't benefit
    # from this — they're filtered out at the location-vs-AWARE join.
    AGGREGATE_REGIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "GLO",
            "ROW",
            "RoW",
            "RER",
            "RNA",
            "RLA",
            "RAS",
            "RAF",
            "OECD",
            "RME",
            "ENTSO-E",
            "UCTE",
            "WECC",
            "FRCC",
            "MRO",
            "NPCC",
            "RFC",
            "SERC",
            "SPP",
            "TRE",
            "WORLD",
            "WORLDPROD",
        }
    )

    def location_for(self, name: str | None) -> str | None:
        """Return the canonical ISO/region code, or ``None`` if nothing
        recognisable trails the name."""
        if not isinstance(name, str) or not name:
            return None
        # 1) Curly-brace form takes priority — least ambiguous.
        last_brace = None
        for m in self.BRACE_RE.finditer(name):
            last_brace = m
        if last_brace is not None:
            return self._normalise(last_brace.group("loc"))
        # 2) Trailing-token fallback — strip a ``U``/``u`` decoration
        # first so ``"... GLO U"`` resolves to ``GLO`` (not ``U``).
        stripped = re.sub(r"\s+[Uu]\s*$", "", name)
        m = self.TRAILING_RE.search(stripped)
        if m is not None:
            return self._normalise(m.group("loc"))
        return None

    @classmethod
    def _normalise(cls, code: str) -> str | None:
        """Strip to country prefix for dashed sub-regions; drop aggregates."""
        # Sub-regional form like ``CA-QC`` → ``CA``. We don't drop the
        # tail unconditionally though: ``IAI Area, EU27 & EFTA`` style
        # codes have no useful country prefix.
        head = code.split("-", 1)[0]
        if head.upper() in cls.AGGREGATE_REGIONS:
            return None
        if len(head) < 2:
            return None
        return head
