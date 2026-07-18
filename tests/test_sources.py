
import pytest

from volume2gym.sources import SourceError, detect_media_type, read_jsonl, sha256_file


def test_jsonl_reader_reports_line_number(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    iterator = read_jsonl(path)
    assert next(iterator) == {"ok": True}
    with pytest.raises(SourceError, match="records.jsonl:2"):
        next(iterator)


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("volume", encoding="utf-8")
    assert sha256_file(path) == "62d7a6b1211d627650e2bf0c869b69b564e2cd74290ae1dd78ae4b5e20b0cfe7"


def test_media_type_uses_file_extension():
    assert detect_media_type("page.jpg") == "image/jpeg"
    assert detect_media_type("page.png") == "image/png"
