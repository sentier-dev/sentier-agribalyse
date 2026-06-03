"""File-format readers. One class per format. Stateless reusables.

Named ``readers`` rather than ``io`` because the stdlib's ``io`` module
is loaded at interpreter start and a top-level ``io`` package shadows it.
"""

from readers.json_reader import GzJsonReader, JsonOrGzJsonReader, JsonReader
from readers.parquet_reader import ParquetReader
from readers.randonneur_loader import RandonneurDataLoader
from readers.xlsx_reader import XlsxReader

__all__ = [
    "GzJsonReader",
    "JsonOrGzJsonReader",
    "JsonReader",
    "ParquetReader",
    "RandonneurDataLoader",
    "XlsxReader",
]
