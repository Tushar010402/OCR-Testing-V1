import pytest

from ocr_bench.engines import boxes_to_text
from ocr_bench.manifest import load_manifest
from ocr_bench.metrics import normalize, score


def test_perfect_match():
    m = score("Total: $12.50", "Total: $12.50")
    assert m["cer"] == 0 and m["wer"] == 0 and m["exact"]


def test_whitespace_and_newlines_are_ignored():
    assert score("line one\nline two", "line one  line   two")["cer"] == 0


def test_loose_ignores_case_and_punctuation():
    m = score("Total: $12.50", "total 1250")
    assert m["cer"] > 0
    assert m["cer_loose"] == 0 and m["exact_loose"]


def test_cer_counts_character_edits():
    assert score("abcd", "abxd")["cer"] == pytest.approx(0.25)


def test_empty_ground_truth():
    assert score("", "")["cer"] == 0
    assert score("", "noise")["cer"] == 1


def test_normalize_unicode_forms():
    assert normalize("ﬁle") == "file"  # NFKC folds the ligature


def test_boxes_to_text_reading_order():
    # Two lines, each delivered right-to-left and out of order.
    items = [
        ([(110, 52), (200, 52), (200, 70), (110, 70)], "world"),
        ([(0, 0), (100, 0), (100, 20), (0, 20)], "Hello"),
        ([(0, 50), (100, 50), (100, 68), (0, 68)], "big"),
        ([(105, 2), (190, 2), (190, 22), (105, 22)], "there"),
    ]
    assert boxes_to_text(items) == "Hello there\nbig world"


def test_manifest_parsing(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "b.txt").write_text("long\ntext", encoding="utf-8")
    (tmp_path / "m.csv").write_text(
        "id,source,ground_truth,ground_truth_file\n"
        'a,https://example.com/a.jpg,"two\nlines",\n'
        "b,imgs/b.png,,gt/b.txt\n",
        encoding="utf-8",
    )
    a, b = load_manifest(tmp_path / "m.csv")
    assert a.ground_truth == "two\nlines" and a.path is None
    assert b.ground_truth == "long\ntext" and b.path == (tmp_path / "imgs" / "b.png").resolve()


def test_manifest_rejects_duplicates(tmp_path):
    (tmp_path / "m.csv").write_text("id,source,ground_truth\nx,a.png,1\nx,b.png,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(tmp_path / "m.csv")
