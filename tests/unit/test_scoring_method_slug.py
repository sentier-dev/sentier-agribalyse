"""Unit tests for ``MethodSlug`` — shared encoder for method tuples on disk."""

from __future__ import annotations

import pytest

from scoring.method_slug import MethodSlug


class TestMethodSlug:
    @pytest.mark.parametrize(
        "method",
        [
            ("a", "b"),
            ("ecoinvent-3.9.1", "EF v3.1", "climate change", "global warming potential (GWP100)"),
            ("only-one",),
            # Regression: a literal ``__`` inside a tuple component must
            # survive the round-trip without splitting into two parts.
            ("EF v3.1", "a__b"),
            ("__leading", "trailing__"),
            ("with_single_underscore", "and__double"),
        ],
    )
    def test_round_trip_preserves_tuple(self, method):
        slug = MethodSlug.encode(method)
        assert MethodSlug.decode(slug) == method

    def test_encode_handles_special_characters(self):
        method = ("dir/with/slash", "spaces and (parens)")
        slug = MethodSlug.encode(method)
        # Slashes and spaces must be quoted so the slug is one filesystem
        # path component per tuple element.
        assert "/" not in slug
        assert " " not in slug
        assert MethodSlug.decode(slug) == method

    def test_decode_empty_returns_empty_tuple(self):
        assert MethodSlug.decode("") == ()

    def test_separator_is_double_underscore(self):
        # Inputs without an underscore encode to a literal ``a__b__c``.
        method = ("a", "b", "c")
        assert MethodSlug.encode(method) == "a__b__c"

    def test_underscore_in_part_is_escaped_so_separator_is_unambiguous(self):
        # ``a_b`` must not look like the separator. Encoded form replaces
        # ``_`` with ``%5F`` so the only ``__`` in the slug is the separator.
        slug = MethodSlug.encode(("a_b", "c"))
        assert slug == "a%5Fb__c"
        assert MethodSlug.decode(slug) == ("a_b", "c")
