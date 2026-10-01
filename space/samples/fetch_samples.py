#!/usr/bin/env python3
"""Fetches the six sample photos the Space offers (space/samples/*.jpg) from
Wikimedia Commons, downsizes each to 640 px on its long side, and writes
samples.json (what server.py and the console read) and LICENSES.md (author,
licence and source page for every file, from Commons' own metadata).

The photos are committed; this script is how they were made, and how to
replace one. Run from anywhere: python space/samples/fetch_samples.py
Needs Pillow.
"""
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
UA = "LoomcoreSampleFetcher/1.0 (https://github.com/armaanmittalweb/loomcore)"

# id, Commons file title, short caption shown in the console.
SAMPLES = [
    ("tabby", "File:Tabby cat with blue eyes-3336579.jpg", "Tabby cat"),
    ("retriever", "File:Golden Retriever Chewing A Stick.jpg", "Golden retriever"),
    ("espresso", "File:Espresso Coffee 01.jpg", "Espresso"),
    ("lighthouse", "File:Point Reyes Lighthouse in December 2019.jpg", "Lighthouse"),
    ("fox", "File:Rød ræv (Vulpes vulpes).jpg", "Red fox"),
    ("guitars", "File:Acoustic guitars in store 20180625.jpg", "Guitar shop"),
]


def api(params):
    params = {**params, "format": "json", "formatversion": "2"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return json.load(r)


def text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def main() -> None:
    titles = "|".join(t for _, t, _ in SAMPLES)
    pages = api({"action": "query", "titles": titles, "prop": "imageinfo",
                 "iiprop": "url|extmetadata", "iiurlwidth": "960"})["query"]["pages"]
    by_title = {p["title"]: p for p in pages}

    manifest, licences = [], []
    for sample_id, title, caption in SAMPLES:
        page = by_title[title]
        info = page["imageinfo"][0]
        meta = info["extmetadata"]
        req = urllib.request.Request(info["thumburl"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        img.thumbnail((640, 640), Image.LANCZOS)
        out = HERE / f"{sample_id}.jpg"
        img.save(out, "JPEG", quality=84, optimize=True, progressive=True)

        author = text(meta.get("Artist", {}).get("value", "")) or "Unknown author"
        licence = meta.get("LicenseShortName", {}).get("value", "")
        licence_url = meta.get("LicenseUrl", {}).get("value", "")
        source = info["descriptionurl"]
        manifest.append({"id": sample_id, "file": out.name, "caption": caption, "width": img.width,
                         "height": img.height, "author": author, "licence": licence, "source": source})
        licences.append((out.name, title, author, licence, licence_url, source))
        print(f"{out.name}: {img.width}x{img.height}, {out.stat().st_size // 1024} KB, {licence}, {author}")

    (HERE / "samples.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                       newline="\n")
    lines = [
        "# Sample photo licences",
        "",
        "The six photos the Space offers as samples, all from Wikimedia Commons, downsized to 640 px on",
        "the long side (no other change). Written by `fetch_samples.py` from each file's Commons metadata.",
        "",
        "| File | Original | Author | Licence |",
        "|---|---|---|---|",
    ]
    for name, title, author, licence, licence_url, source in licences:
        lic = f"[{licence}]({licence_url})" if licence_url else licence
        lines.append(f"| `{name}` | [{title[5:]}]({source}) | {author} | {lic} |")
    lines.append("")
    (HERE / "LICENSES.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
