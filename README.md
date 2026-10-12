# OCR-Testing-V1

Benchmark harness that runs **Tesseract**, **EasyOCR** and **PaddleOCR** over the same set of
images and scores each one against ground-truth text using character and word error rates
(CER / WER).

## Setup

```bash
sudo apt-get install -y tesseract-ocr        # the Tesseract binary
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only, much smaller
pip install -r requirements.txt
```

EasyOCR and PaddleOCR download their models on first use (a few hundred MB).

## Adding your images

Edit `data/manifest.csv`. Add one row per image:

```csv
id,source,ground_truth
invoice_001,https://example.com/scans/invoice_001.jpg,"ACME Corp
Invoice #4471
Total: $1,284.50"
local_scan,images/scan_02.png,"Text on the scan"
```

| Column | Meaning |
|---|---|
| `id` | A unique short name. It is used for file names in the results. |
| `source` | A public `http(s)` URL or a path relative to the manifest. Downloads are cached in `data/cache/`. |
| `ground_truth` | The exact text in the image. Multi-line values need double quotes. |
| `ground_truth_file` | Optional. A `.txt` file (relative to the manifest) for long transcriptions, used instead of `ground_truth`. |

URLs must return the image itself. Google Drive and Dropbox "share" links return an HTML page,
so convert them to direct-download links. A dead or non-image link is reported in the summary
and doesn't stop the run.

## Running

```bash
python run_benchmark.py                                   # data/manifest.csv, all engines
python run_benchmark.py --engines tesseract,paddleocr     # pick engines
python run_benchmark.py --manifest samples/manifest.csv   # synthetic smoke-test set
python run_benchmark.py --limit 5                         # quick check on the first 5 rows
```

Each run writes to `results/<timestamp>/`:

- `summary.md` has per-engine averages and a per-image CER table.
- `results.csv` has one row per (image, engine) with CER, WER, confidence and seconds.
- `predictions/<engine>/<id>.txt` holds the raw text each engine read, so you can check *why* it scored badly.

### Reading the metrics

- **CER**: the share of characters inserted, deleted or substituted to turn the output into the
  ground truth. 0% is perfect, and it can go above 100% if the engine outputs a lot of junk.
- **WER**: the same thing counted over words. A single wrong character costs a whole word, so WER
  is always harsher than CER.
- **Strict vs loose**: *strict* keeps case and punctuation, which matters for IDs, amounts and
  codes. *Loose* lowercases and strips punctuation. Line breaks and repeated spaces are ignored
  in both.
- Multi-line output is put in reading order (top-to-bottom, then left-to-right) before scoring.

## Sample set

`samples/make_samples.py` generates 24 synthetic images: 4 texts × clean, small, noisy, blurred,
skewed and low-contrast versions. Use them to check that the setup works before testing real data.
On that set (CPU, 4 cores):

| Engine | Mean CER | Mean WER | Median s/img |
|---|---|---|---|
| Tesseract 5.3.4 | 3.3% | 14.3% | 0.31 |
| EasyOCR 1.7.2 | 2.5% | 10.8% | 0.29 |
| PaddleOCR 3.7.0 | 0.3% | 3.3% | 1.34 |

Synthetic text is far easier than real scans, so expect higher error rates on real images.

## Notes

- PaddleOCR runs with `enable_mkldnn=False`. With oneDNN enabled, paddlepaddle 3.3 fails on CPU
  with `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`.
- Everything runs on CPU. To use a GPU, pass `gpu=True` to EasyOCR and install `paddlepaddle-gpu`.
- Run `python -m pytest` to test the scoring and manifest code.
