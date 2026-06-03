"""``RegionalSuffixParser`` — extract regional codes embedded in SimaPro flow names.

SimaPro encodes the country/region of extraction inside the flow name
itself (``Water, well, CN``, ``Water, unspecified natural origin, RoW``,
``Water, lake, FR``). The bw2io biosphere catalog has no such regional
dimension — every flow is just (uuid, name, categories) — so once the
linker collapses regional variants onto a base UUID, the per-region
deprivation CFs SimaPro carries on AGB's reference method are lost.

This parser surfaces the regional suffix so downstream linkers can
preserve it as part of the flow identity. Each successfully linked
regional flow ends up with a synthetic code ``"<base_uuid>@<region>"``
in the inventory matrix; the CF registry then emits one row per
(base_uuid, region) using SimaPro's per-region CFs as the source of
truth.

The parser is allowlist-driven by ISO-3166 alpha-2 codes plus a small
set of well-known aggregate regions. Without an allowlist the regex
``,\\s*[A-Z][A-Z0-9-]{1,9}$`` matches things that are NOT regions —
e.g. ``Methane, dichloro-, HCC-30`` (chemical formula suffix) or
``Ethane, 1,1,2-trichloro-1,2,2-trifluoro-, CFC-113``. Misclassifying
those as regions would shatter the matcher's flow identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class RegionalSuffixParser:
    """Extract an ISO-2 country code or aggregate region from a flow name.

    ``parse(name)`` returns ``(base_name, region)``. ``region`` is the
    empty string when the trailing token is not a recognised region,
    OR when there is no comma-prefixed trailing token at all.

    The parser is deterministic and stateless. The allowlist is held as
    class-level immutable sets, so instantiation is cheap and the same
    instance can be reused everywhere.
    """

    # ISO-3166 alpha-2 country codes (full list — 249 entries).
    ISO2: ClassVar[frozenset[str]] = frozenset(
        {
            "AD",
            "AE",
            "AF",
            "AG",
            "AI",
            "AL",
            "AM",
            "AO",
            "AQ",
            "AR",
            "AS",
            "AT",
            "AU",
            "AW",
            "AX",
            "AZ",
            "BA",
            "BB",
            "BD",
            "BE",
            "BF",
            "BG",
            "BH",
            "BI",
            "BJ",
            "BL",
            "BM",
            "BN",
            "BO",
            "BQ",
            "BR",
            "BS",
            "BT",
            "BV",
            "BW",
            "BY",
            "BZ",
            "CA",
            "CC",
            "CD",
            "CF",
            "CG",
            "CH",
            "CI",
            "CK",
            "CL",
            "CM",
            "CN",
            "CO",
            "CR",
            "CU",
            "CV",
            "CW",
            "CX",
            "CY",
            "CZ",
            "DE",
            "DJ",
            "DK",
            "DM",
            "DO",
            "DZ",
            "EC",
            "EE",
            "EG",
            "EH",
            "ER",
            "ES",
            "ET",
            "FI",
            "FJ",
            "FK",
            "FM",
            "FO",
            "FR",
            "GA",
            "GB",
            "GD",
            "GE",
            "GF",
            "GG",
            "GH",
            "GI",
            "GL",
            "GM",
            "GN",
            "GP",
            "GQ",
            "GR",
            "GS",
            "GT",
            "GU",
            "GW",
            "GY",
            "HK",
            "HM",
            "HN",
            "HR",
            "HT",
            "HU",
            "ID",
            "IE",
            "IL",
            "IM",
            "IN",
            "IO",
            "IQ",
            "IR",
            "IS",
            "IT",
            "JE",
            "JM",
            "JO",
            "JP",
            "KE",
            "KG",
            "KH",
            "KI",
            "KM",
            "KN",
            "KP",
            "KR",
            "KW",
            "KY",
            "KZ",
            "LA",
            "LB",
            "LC",
            "LI",
            "LK",
            "LR",
            "LS",
            "LT",
            "LU",
            "LV",
            "LY",
            "MA",
            "MC",
            "MD",
            "ME",
            "MF",
            "MG",
            "MH",
            "MK",
            "ML",
            "MM",
            "MN",
            "MO",
            "MP",
            "MQ",
            "MR",
            "MS",
            "MT",
            "MU",
            "MV",
            "MW",
            "MX",
            "MY",
            "MZ",
            "NA",
            "NC",
            "NE",
            "NF",
            "NG",
            "NI",
            "NL",
            "NO",
            "NP",
            "NR",
            "NU",
            "NZ",
            "OM",
            "PA",
            "PE",
            "PF",
            "PG",
            "PH",
            "PK",
            "PL",
            "PM",
            "PN",
            "PR",
            "PS",
            "PT",
            "PW",
            "PY",
            "QA",
            "RE",
            "RO",
            "RS",
            "RU",
            "RW",
            "SA",
            "SB",
            "SC",
            "SD",
            "SE",
            "SG",
            "SH",
            "SI",
            "SJ",
            "SK",
            "SL",
            "SM",
            "SN",
            "SO",
            "SR",
            "SS",
            "ST",
            "SV",
            "SX",
            "SY",
            "SZ",
            "TC",
            "TD",
            "TF",
            "TG",
            "TH",
            "TJ",
            "TK",
            "TL",
            "TM",
            "TN",
            "TO",
            "TR",
            "TT",
            "TV",
            "TW",
            "TZ",
            "UA",
            "UG",
            "UM",
            "US",
            "UY",
            "UZ",
            "VA",
            "VC",
            "VE",
            "VG",
            "VI",
            "VN",
            "VU",
            "WF",
            "WS",
            "XK",
            "YE",
            "YT",
            "ZA",
            "ZM",
            "ZW",
        }
    )

    # Aggregate regions ecoinvent / SimaPro use. ``RoW`` and friends
    # have their own CF entries distinct from named countries.
    AGGREGATE: ClassVar[frozenset[str]] = frozenset(
        {
            "GLO",
            "RoW",
            "RoE",
            "RER",
            "RNA",
            "RLA",
            "RAS",
            "RAF",
            "RME",
            "OECD",
            "UCTE",
            "ENTSO-E",
            "NORDEL",
            "WEU",
            "WORLD",
            # IAI area aggregates used in some ecoinvent processes.
            "IAI",
        }
    )

    # Sub-regional codes like ``CA-QC``, ``US-WECC`` are common in
    # ecoinvent. They follow the pattern ``<ISO2>-<3-5 chars>``.
    _SUBREGION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<iso>[A-Z]{2})-(?P<sub>[A-Z0-9]{2,7})$"
    )

    # Suffix regex: trailing ``, <token>`` where token starts with a
    # capital and is 2-10 chars long (allowing letters, digits, hyphens).
    # Lowercase letters allowed AFTER the first character so aggregate
    # codes like ``RoW`` / ``RoE`` parse — the allowlist gate filters
    # spurious matches.
    SUFFIX_RE: ClassVar[re.Pattern[str]] = re.compile(r",\s*(?P<region>[A-Z][A-Za-z0-9-]{1,9})\s*$")

    def parse(self, source_name: str) -> tuple[str, str]:
        """Return ``(base_name, region)``.

        ``region`` is empty when the trailing token is not in the
        ``ISO2`` / ``AGGREGATE`` allowlists (or matches the sub-regional
        ``<ISO2>-<sub>`` pattern). The base name is the original
        string with the recognised suffix stripped.
        """
        if not source_name:
            return source_name, ""
        m = self.SUFFIX_RE.search(source_name)
        if m is None:
            return source_name, ""
        candidate = m.group("region")
        if not self._is_region(candidate):
            return source_name, ""
        base = source_name[: m.start()].rstrip()
        return base, candidate

    @classmethod
    def _is_region(cls, token: str) -> bool:
        if token in cls.ISO2 or token in cls.AGGREGATE:
            return True
        sub_m = cls._SUBREGION_RE.match(token)
        return bool(sub_m and sub_m.group("iso") in cls.ISO2)
