"""Standalone customer-side importer for the Brightway export.

The single module here is copied verbatim into every export by
``dds-build-bw-package`` and runs in the *user's* Brightway environment. It is
deliberately function-based, stdlib + numpy + pandas only — a documented
exception to the project's OOP-only rule: it must import nothing from this
package (enforced by tests/unit/bw_import) so the copied file works anywhere.
"""
