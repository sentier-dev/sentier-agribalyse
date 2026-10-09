"""Build the public documentation site from an explicit, inventory-free file list.

Run: uv run --no-project --with markdown==3.7 scripts/build_site.py
"""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]


def esc(value):
    return html.escape(str(value), quote=True)


def build(out, source_repository=None, source_ref=None, preview_url=None):
    config = json.loads((ROOT / "site/config.json").read_text())
    repo = config["repository"]
    source_repo = source_repository or config.get("source_repository", repo)
    config["source_ref"] = source_ref or config["source_ref"]
    if preview_url and not preview_url.startswith("https://github.com/"):
        raise ValueError("Preview notice must link to a GitHub pull request")
    preview_notice = (
        '<aside class="preview-notice" aria-label="Review preview"><div class="wrap">'
        '<strong>Review preview.</strong> These documentation changes are proposed '
        f'for the shared repository. <a href="{esc(preview_url)}">Review the pull request ↗</a>'
        "</div></aside>"
        if preview_url
        else ""
    )
    dataset = config["dataset"]
    provider = config["provider"]
    docs = config["documents"]
    routes = {str(Path(d["source"])): d["page"] for d in docs}
    pages = [
        ("index.html", "Overview"),
        ("import.html", "Import & use"),
        ("testing.html", "Evidence & testing"),
        ("findings.html", "Findings"),
        ("contribute.html", "Contribute"),
    ]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "site/assets", out / "assets", dirs_exist_ok=True)
    shutil.copyfile(ROOT / "site/style.css", out / "style.css")
    (out / ".nojekyll").touch()
    records = [
        json.loads(p.read_text()) for p in sorted((ROOT / "community/contributions").glob("*.json"))
    ]
    packets = json.loads((ROOT / "community/recommendations/tracker.json").read_text())
    accepted = sum(r["status"] == "accepted" for r in records)
    delivered = sum(p["status"] in ("sent", "acknowledged", "resolved") for p in packets)

    def shell(title, body, current, sidebar=False):
        nav = "".join(
            f'<a href="{page}"'
            + (' aria-current="page"' if page == current else "")
            + f">{label}</a>"
            for page, label in pages
        )
        side = (
            '<aside class="sidebar" aria-label="Documentation"><strong>In this workspace</strong>'
            + "".join(
                f'<a href="{d["page"]}"'
                + (' aria-current="page"' if d["page"] == current else "")
                + f">{esc(d['title'])}</a>"
                for d in docs
            )
            + "</aside>"
        )
        main = (
            f'<main id="main" class="wrap document">{side}<article class="article">{body}</article></main>'
            if sidebar
            else f'<main id="main">{body}</main>'
        )
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · {esc(dataset)} · Sentier</title><meta name="description" content="{esc(config["description"])}"><meta name="theme-color" content="#3c5343"><link rel="icon" href="assets/favicon-32x32.png" type="image/png"><link rel="stylesheet" href="style.css"></head><body>
<a class="skip" href="#main">Skip to content</a><header class="masthead"><div class="wrap top"><div class="brand"><a class="brand-logo" href="https://www.d-d-s.ch/" aria-label="Départ de Sentier"><img src="assets/dds-logo.svg" alt="Départ de Sentier" width="200" height="94"></a><a class="dataset" href="index.html" aria-label="{esc(dataset)} workspace home">{esc(dataset)}</a></div><div class="top-links"><a href="https://www.d-d-s.ch/">Départ de Sentier ↗</a><a href="{esc(config["sibling_url"])}">{esc(config["sibling_name"])} ↗</a><a href="{repo}">Repository ↗</a></div></div><nav class="wrap nav" aria-label="Main navigation">{nav}</nav></header>
{preview_notice}{main}
<footer><div class="wrap"><p><strong>A Sentier community workspace.</strong><br>Evidence and contributions toward better {esc(dataset)} data. Reviews and recommendations concern the source dataset; provider endorsement is not implied.<small class="brand-credit"><a href="https://github.com/Depart-de-Sentier/dds-logo">Logo</a> by <a href="https://proov.fr/">Sylvain Prevost</a> · <a href="https://www.d-d-s.ch/">Départ de Sentier</a></small></p><p><a href="terms.html">Data & attribution</a><br><a href="{esc(config["sibling_url"])}">{esc(config["sibling_name"])} ↗</a><br><a href="{repo}/issues/new/choose">Start a discussion ↗</a></p></div></footer></body></html>'''

    layer_cards = ""
    for record in records:
        checks = {c["id"]: c["status"] for run in record["runs"] for c in run.get("checks", [])}
        passed = sum(checks.get(c) == "passed" for c in record["required_checks"])
        layer_cards += f'''<div class="record"><div class="record-head"><h3>{esc(record["id"])} · {esc(record["title"])}</h3><span class="badge {esc(record["status"])}">{esc(record["status"])}</span></div><p class="test-result">{passed} / {len(record["required_checks"])} required checks recorded as passed · {len(record["approvals"])} recorded approvals</p><a href="{source_repo}/blob/{config["source_ref"]}/community/contributions/{esc(record["id"])}.json">Open the contribution record ↗</a></div>'''
    homepage = f'''<section class="wrap hero"><div><p class="eyebrow">Open dataset collaboration / {esc(dataset)}</p><h1>Make the dataset better.<br>Together.</h1><p class="lead">Reproduce the import. Bring partner evidence and hackathon ideas into testable contributions. Build shared understanding, then return actionable recommendations to {esc(provider)}.</p><div class="actions"><a class="button primary" href="import.html">Start with the import →</a><a class="button" href="contribute.html">Bring a contribution</a></div></div><div class="route"><span class="tag">From source to shared understanding</span><ol><li><span>01</span>Canonical import</li><li><span>02</span>Partner contributions</li><li><span>03</span>Application & testing</li><li><span>04</span>Reviewed consensus</li><li><span>05</span>Recommendations to {esc(provider)}</li></ol></div></section>
<div class="strip"><div class="wrap"><span><strong>Public collaboration</strong> · across dataset releases</span><span><strong>{len(records)} contribution record{"s" if len(records) != 1 else ""}</strong> · {accepted} accepted</span><span><strong>{delivered} recommendations sent</strong> · delivery requires evidence</span></div></div>
<section class="wrap section"><p class="eyebrow">A shared place to work</p><h2>Keep the evidence connected.</h2><p class="section-intro">A finding about the source data, an importer defect and a new scenario need different evidence. Keep the original reproducible, apply contributions in separate candidates, and make every claim reviewable.</p><div class="cards"><div class="card"><span class="number">01 / REPRODUCE</span><h3>Start from a known baseline</h3><p>Pin the release, source hashes, background and mappings. Understand what the import transforms before comparing results.</p><a href="import.html">Import & use →</a></div><div class="card"><span class="number">02 / INVESTIGATE</span><h3>Findings with a traceable source</h3><p>Connect observations to dataset identities, release files and the discussion where contributors can reproduce them.</p><a href="findings.html">Explore the findings →</a></div><div class="card"><span class="number">03 / VALIDATE</span><h3>Test the contribution's claim</h3><p>Account for changed exchanges, check characterization, and compare equivalent scopes before seeking consensus.</p><a href="testing.html">Evidence & testing →</a></div></div><div class="band"><p><strong>Help improve {esc(dataset)}.</strong><br>Bring a reproducible observation, a model or an application test. A proposal becomes consensus only after the evidence and reviewer decisions are recorded.</p><a class="button" href="{repo}/issues/new/choose">Open a discussion ↗</a></div></section>
<section class="wrap section" style="padding-top:0"><p class="eyebrow">Work in progress</p><h2>Contributions under review</h2><div class="records">{layer_cards}</div><p class="test-result" style="margin-top:20px">{esc(config["validation_note"])}</p></section>'''
    (out / "index.html").write_text(shell("Overview", homepage, "index.html"))
    for doc in docs:
        source = Path(doc["source"])
        text = (ROOT / source).read_text()
        rendered = markdown.markdown(
            text, extensions=["tables", "fenced_code", "sane_lists", "toc"]
        )

        def rewrite(match, source=source):
            link = html.unescape(match.group(1))
            if re.match(r"^(https?:|mailto:|#)", link):
                return match.group(0)
            # Resolve against the source document, never the generated URL.
            path, sep, fragment = link.partition("#")
            absolute = (ROOT / source.parent / path).resolve()
            try:
                relative = absolute.relative_to(ROOT).as_posix()
            except ValueError:
                raise ValueError(f"Link leaves repository: {link}") from None
            target = routes.get(relative) or f"{source_repo}/blob/{config['source_ref']}/{relative}"
            return f'href="{esc(target + (sep + fragment if sep else ""))}"'

        rendered = re.sub(r'href="([^"]+)"', rewrite, rendered)
        rendered = rendered.replace("<table>", '<div class="table-scroll"><table>').replace(
            "</table>", "</table></div>"
        )
        if doc["page"] == "contribute.html":
            rendered += "<h2>Recorded contributions</h2>" + layer_cards
        (out / doc["page"]).write_text(shell(doc["title"], rendered, doc["page"], True))
    (out / "404.html").write_text(
        shell(
            "Page not found",
            '<section class="wrap section"><h1>That path is missing.</h1><p><a href="index.html">Return to the workspace →</a></p></section>',
            "404.html",
        )
    )
    # Public data contains coordination metadata only, never source inventory rows.
    (out / "status.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "contributions": len(records),
                "accepted": accepted,
                "recommendations_sent": delivered,
                "source_ref": config["source_ref"],
                "preview": bool(preview_url),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built {dataset}: {len(docs) + 2} pages in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    parser.add_argument("--source-repository", help="Repository containing this build's source")
    parser.add_argument("--source-ref", help="Source branch or revision for repository links")
    parser.add_argument("--preview-url", help="Show a review notice linking to this pull request")
    args = parser.parse_args()
    build(args.out, args.source_repository, args.source_ref, args.preview_url)
