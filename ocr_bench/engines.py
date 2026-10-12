"""Thin wrappers that give every OCR engine the same interface.

Each engine takes an image path and returns an ``OcrResult`` with the text in
reading order (top-to-bottom, left-to-right) and a mean confidence in [0, 1].
Models load lazily on first use so that ``--engines tesseract`` never pays for
importing torch or paddle.
"""

from dataclasses import dataclass


@dataclass
class OcrResult:
    text: str
    confidence: float | None


def boxes_to_text(items) -> str:
    """Join ``(box, text)`` detections into lines in reading order.

    ``box`` is any sequence of (x, y) points. Detections whose vertical
    centres fall within half a line height of each other share a line.
    """
    entries = []
    for box, text in items:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        entries.append(((min(ys) + max(ys)) / 2, max(max(ys) - min(ys), 1), min(xs), text))
    entries.sort(key=lambda e: e[0])

    lines: list[list] = []
    for entry in entries:
        if lines:
            last = lines[-1]
            centre = sum(e[0] for e in last) / len(last)
            height = sum(e[1] for e in last) / len(last)
            if abs(entry[0] - centre) <= height / 2:
                last.append(entry)
                continue
        lines.append([entry])
    return "\n".join(" ".join(e[3] for e in sorted(line, key=lambda e: e[2])) for line in lines)


def _mean(values) -> float | None:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else None


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        import pytesseract

        self._tess = pytesseract
        self.lang = lang

    def version(self) -> str:
        return str(self._tess.get_tesseract_version())

    def run(self, image_path) -> OcrResult:
        from PIL import Image

        with Image.open(image_path) as im:
            im.load()
            text = self._tess.image_to_string(im, lang=self.lang)
            data = self._tess.image_to_data(im, lang=self.lang, output_type=self._tess.Output.DICT)
        confs = [c / 100 for c, w in zip(data["conf"], data["text"]) if float(c) >= 0 and w.strip()]
        return OcrResult(text.strip(), _mean(confs))


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, lang: str = "en"):
        import easyocr

        self._mod = easyocr
        self._reader = easyocr.Reader([lang], gpu=False, verbose=False)

    def version(self) -> str:
        return self._mod.__version__

    def run(self, image_path) -> OcrResult:
        detections = self._reader.readtext(str(image_path), detail=1)
        text = boxes_to_text((box, txt) for box, txt, _ in detections)
        return OcrResult(text, _mean(conf for _, _, conf in detections))


class PaddleOcrEngine:
    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        import paddleocr

        self._mod = paddleocr
        self._ocr = paddleocr.PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # paddlepaddle 3.3's oneDNN CPU path raises NotImplementedError
            # (ConvertPirAttribute2RuntimeAttribute) on every image.
            enable_mkldnn=False,
        )

    def version(self) -> str:
        return self._mod.__version__

    def run(self, image_path) -> OcrResult:
        texts, scores, polys = [], [], []
        for page in self._ocr.predict(str(image_path)):
            texts += page["rec_texts"]
            scores += page["rec_scores"]
            polys += [p.tolist() for p in page["rec_polys"]]
        return OcrResult(boxes_to_text(zip(polys, texts)), _mean(scores))


ENGINES = {e.name: e for e in (TesseractEngine, EasyOcrEngine, PaddleOcrEngine)}
