"""Load the test-set manifest and fetch remote images into a local cache."""

import csv
import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"


@dataclass
class Sample:
    id: str
    source: str  # http(s) URL or path relative to the manifest
    ground_truth: str
    path: Path | None = None  # local file once fetched


def _read_ground_truth(row: dict, base: Path) -> str:
    if row.get("ground_truth_file"):
        return (base / row["ground_truth_file"]).read_text(encoding="utf-8")
    return row.get("ground_truth") or ""


def load_manifest(manifest_path: Path) -> list[Sample]:
    """Read a CSV with columns ``id, source, ground_truth[, ground_truth_file]``.

    ``ground_truth`` may span several lines if the cell is quoted. For long
    texts, put them in a .txt file and point ``ground_truth_file`` at it.
    """
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    samples = []
    seen = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            row = {k.strip(): v or "" for k, v in row.items() if k}
            sid = (row.get("id") or "").strip()
            source = (row.get("source") or row.get("url") or "").strip()
            if not sid or not source:
                raise ValueError(f"{manifest_path}:{lineno}: 'id' and 'source' are required")
            if sid in seen:
                raise ValueError(f"{manifest_path}:{lineno}: duplicate id {sid!r}")
            seen.add(sid)
            samples.append(Sample(sid, source, _read_ground_truth(row, base)))
            if not _is_url(source):
                samples[-1].path = (base / source).resolve()
    return samples


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _cache_path(url: str, content_type: str | None) -> Path:
    digest = _url_key(url)
    ext = Path(urlparse(url).path).suffix.lower()
    if not ext and content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    return CACHE_DIR / f"{digest}{ext or '.img'}"


def fetch(sample: Sample, *, timeout: float = 60) -> Path:
    """Download a URL sample once, verify it is a readable image, return its path."""
    if sample.path is not None:
        if not sample.path.exists():
            raise FileNotFoundError(sample.path)
        return sample.path
    existing = [p for p in CACHE_DIR.glob(_url_key(sample.source) + ".*") if p.suffix != ".part"]
    if existing:
        sample.path = existing[0]
        return sample.path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(sample.source, timeout=timeout, headers={"User-Agent": "ocr-bench/1.0"})
    resp.raise_for_status()
    path = _cache_path(sample.source, resp.headers.get("Content-Type"))
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(resp.content)
    try:
        with Image.open(tmp) as im:
            im.verify()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"{sample.id}: {sample.source} did not return an image "
            f"(Content-Type {resp.headers.get('Content-Type')!r}). "
            "Share links (Drive, Dropbox) usually need a direct-download URL."
        ) from exc
    tmp.rename(path)
    sample.path = path
    return path
