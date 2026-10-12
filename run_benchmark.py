#!/usr/bin/env python3
"""Run every OCR engine over the manifest and score it against ground truth.

    python run_benchmark.py                          # data/manifest.csv, all engines
    python run_benchmark.py --manifest samples/manifest.csv --engines tesseract,easyocr

Writes to results/<timestamp>/:
    results.csv          one row per (image, engine) with CER/WER, time, confidence
    summary.md           per-engine averages plus the worst images
    predictions/<engine>/<id>.txt   the raw text each engine produced
"""

import argparse
import csv
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from ocr_bench.engines import ENGINES
from ocr_bench.manifest import fetch, load_manifest
from ocr_bench.metrics import score

FIELDS = [
    "id", "engine", "status", "cer", "wer", "cer_loose", "wer_loose",
    "exact", "exact_loose", "confidence", "seconds", "error",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.csv", type=Path)
    p.add_argument("--engines", default=",".join(ENGINES), help=f"comma-separated: {', '.join(ENGINES)}")
    p.add_argument("--out", type=Path, help="output directory (default results/<timestamp>)")
    p.add_argument("--limit", type=int, help="only the first N images")
    return p.parse_args(argv)


def run(args) -> Path:
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = set(engine_names) - set(ENGINES)
    if unknown:
        sys.exit(f"unknown engine(s): {', '.join(sorted(unknown))}; choose from {', '.join(ENGINES)}")

    samples = load_manifest(args.manifest)[: args.limit]
    if not samples:
        sys.exit(f"{args.manifest} has no rows yet. Add one line per image: id,source,ground_truth")
    out = args.out or Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(samples)} image(s)...")
    ready = []
    fetch_errors = {}
    for s in samples:
        try:
            fetch(s)
            ready.append(s)
        except Exception as exc:  # keep going; a dead link shouldn't sink the run
            fetch_errors[s.id] = f"fetch failed: {exc}"
            print(f"  ! {s.id}: {exc}")

    rows, versions = [], {}
    for name in engine_names:
        print(f"\n[{name}] loading...")
        t0 = time.perf_counter()
        try:
            engine = ENGINES[name]()
        except Exception as exc:
            print(f"  ! could not load {name}: {exc}")
            rows += [{"id": s.id, "engine": name, "status": "error", "error": f"engine load: {exc}"} for s in samples]
            continue
        versions[name] = (engine.version(), time.perf_counter() - t0)
        pred_dir = out / "predictions" / name
        pred_dir.mkdir(parents=True, exist_ok=True)

        for s in samples:
            if s.id in fetch_errors:
                rows.append({"id": s.id, "engine": name, "status": "error", "error": fetch_errors[s.id]})
                continue
            t = time.perf_counter()
            try:
                result = engine.run(s.path)
            except Exception as exc:
                traceback.print_exc()
                rows.append({"id": s.id, "engine": name, "status": "error", "error": repr(exc)})
                continue
            elapsed = time.perf_counter() - t
            (pred_dir / f"{s.id}.txt").write_text(result.text, encoding="utf-8")
            m = score(s.ground_truth, result.text)
            rows.append({"id": s.id, "engine": name, "status": "ok", **m,
                         "confidence": result.confidence, "seconds": elapsed, "error": ""})
            print(f"  {s.id:<24} CER {m['cer']:6.1%}  WER {m['wer']:6.1%}  {elapsed:5.2f}s")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary = render_summary(rows, engine_names, versions, len(samples), args.manifest)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Results written to {out}/")
    return out


def render_summary(rows, engine_names, versions, n_images, manifest) -> str:
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    def pct(v):
        return "–" if v is None else f"{v:.1%}"

    lines = [
        "# OCR benchmark summary", "",
        f"Manifest: `{manifest}`, {n_images} image(s). Lower CER/WER is better.",
        "*Strict* keeps case and punctuation; *loose* ignores both.", "",
        "| Engine | Version | OK | Mean CER | Mean WER | CER (loose) | WER (loose) | Exact (loose) | Median s/img | Load s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = []
    for name in engine_names:
        ok = [r for r in rows if r["engine"] == name and r["status"] == "ok"]
        ver, load = versions.get(name, ("not loaded", None))
        mean_cer = avg(r["cer"] for r in ok)
        ranked.append((mean_cer, name))
        cells = [
            name, ver, f"{len(ok)}/{n_images}",
            pct(mean_cer),
            pct(avg(r["wer"] for r in ok)),
            pct(avg(r["cer_loose"] for r in ok)),
            pct(avg(r["wer_loose"] for r in ok)),
            f"{sum(r['exact_loose'] for r in ok)}/{len(ok)}",
            f"{statistics.median(r['seconds'] for r in ok):.2f}" if ok else "–",
            f"{load:.1f}" if load is not None else "–",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    ranked = sorted((c, n) for c, n in ranked if c is not None)
    if ranked:
        lines += ["", f"**Best mean CER: {ranked[0][1]} ({ranked[0][0]:.1%}).**"]

    lines += ["", "## Per-image CER (strict)", "",
              "| Image | " + " | ".join(engine_names) + " |",
              "|---|" + "---|" * len(engine_names)]
    by_key = {(r["id"], r["engine"]): r for r in rows}
    for sid in dict.fromkeys(r["id"] for r in rows):
        cells = []
        for name in engine_names:
            r = by_key.get((sid, name))
            cells.append("error" if not r or r["status"] != "ok" else pct(r["cer"]))
        lines.append(f"| {sid} | " + " | ".join(cells) + " |")

    grouped: dict[tuple, list] = {}
    for r in rows:
        if r["status"] != "ok":
            grouped.setdefault((r["engine"], r["error"].strip()), []).append(r["id"])
    if grouped:
        lines += ["", "## Errors", ""]
        for (engine, error), ids in grouped.items():
            where = f"`{ids[0]}`" if len(ids) == 1 else f"{len(ids)} images"
            lines.append(f"- {engine} ({where}): {error}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run(parse_args())
