import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]

try:
    import markdown  # noqa: F401
except ImportError:  # The not-found page does not pass through Markdown: a stand-in will do.
    stand_in = types.ModuleType("markdown")
    stand_in.markdown = lambda text, extensions=(): "<p>Markdown is not installed.</p>"
    sys.modules["markdown"] = stand_in


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_site = load("build_site")
check_site = load("check_site")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links += [value for key, value in attrs if key in ("href", "src")]


def local_links(page):
    parser = Links()
    parser.feed(page.read_text())
    return [link for link in parser.links if not urlsplit(link).scheme and not link.startswith("#")]


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class NotFoundPageTests(unittest.TestCase):
    """Pages serves 404.html for every missing path, also /base/foo/bar."""

    def test_not_found_page_links_are_absolute_under_the_configured_base(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "site"
            quiet(build_site.build, out)
            base = urlsplit(json.loads((ROOT / "site/config.json").read_text())["site_url"]).path
            links = local_links(out / "404.html")
            self.assertIn(f"{base}index.html", links)
            self.assertIn(f"{base}style.css", links)
            self.assertEqual([link for link in links if not link.startswith(base)], [])
            self.assertEqual(json.loads((out / "status.json").read_text())["base_path"], base)
            self.assertEqual(quiet(check_site.check, out), 0)

    def test_site_url_override_moves_the_base(self):
        for site_url, base in [
            ("https://example.org/mirror", "/mirror/"),
            ("https://example.org/group/mirror/", "/group/mirror/"),
            ("https://docs.example.org", "/"),
            ("https://example.org/my%20site/", "/my%20site/"),
        ]:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "site"
                quiet(build_site.build, out, site_url=site_url)
                links = local_links(out / "404.html")
                self.assertIn(f"{base}index.html", links, site_url)
                self.assertEqual([link for link in links if not link.startswith("/")], [])
                self.assertEqual(quiet(check_site.check, out), 0, site_url)

    def test_site_url_without_scheme_and_host_is_refused(self):
        # "host/path" parses as a path: the base would silently become "/host/path/".
        for site_url in ("example.org/mirror", "/mirror/", "ftp://example.org/mirror"):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                quiet(build_site.build, Path(directory) / "site", site_url=site_url)

    def test_other_pages_keep_relative_links(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "site"
            quiet(build_site.build, out)
            self.assertIn("style.css", local_links(out / "index.html"))


class SiteCheckTests(unittest.TestCase):
    def site(self, directory, not_found, base_path="/base/"):
        root = Path(directory)
        (root / "style.css").write_text("")
        (root / "index.html").write_text('<link rel="stylesheet" href="style.css">')
        (root / "404.html").write_text(not_found)
        (root / "status.json").write_text(json.dumps({"base_path": base_path}))
        return root

    def test_relative_link_on_the_not_found_page_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.site(directory, '<a href="index.html">home</a>')
            self.assertEqual(quiet(check_site.check, root), 1)

    def test_absolute_links_under_the_base_pass_and_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.site(
                directory,
                '<link href="/base/style.css"><a href="/base/index.html">home</a><a href="#main">skip</a><main id="main"></main>',
            )
            self.assertEqual(quiet(check_site.check, root), 0)

    def test_absolute_link_to_a_missing_file_or_outside_the_base_fails(self):
        for link in ("/base/missing.html", "/elsewhere/index.html", "/index.html"):
            with tempfile.TemporaryDirectory() as directory:
                root = self.site(directory, f'<a href="{link}">home</a>')
                self.assertEqual(quiet(check_site.check, root), 1, link)


if __name__ == "__main__":
    unittest.main()
