from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import print_studyinfo_dat as printer


SAMPLE_TEXT = """[StudyInfo]
姓名=张三
性别=男
年龄=28
检查日期=2024-01-15
医院=华山医院
备注=术后CT对照
PatientID=sub002
"""


def _write_gbk(path: Path, text: str = SAMPLE_TEXT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("gbk"))


def test_decode_prefers_chinese_encoding_over_utf8() -> None:
    data = SAMPLE_TEXT.encode("gbk")
    decoded = printer.decode_bytes(data)
    assert decoded.encoding in {"gb18030", "gbk", "cp936"}
    assert "姓名=张三" in decoded.text
    assert "华山医院" in decoded.text


def test_print_studyinfo_dat_writes_readable_summary(tmp_path: Path) -> None:
    assert printer.parse_args([]).input_dir == printer.DEFAULT_INPUT
    assert printer.parse_args([]).output == printer.DEFAULT_OUTPUT

    root = tmp_path / "story_listen_v2"
    studyinfo = root / "sub002" / "sub002" / "postseegct" / "StudyInfo.dat"
    other = root / "sub003" / "notes.dat"
    preload = root / "sub002" / ".recording_preload.dat"
    _write_gbk(studyinfo)
    _write_gbk(other, "姓名=李四\nPatientID=sub003\n")
    preload.parent.mkdir(parents=True, exist_ok=True)
    preload.write_bytes(b"\x00\x01\x02\x03not-studyinfo")

    output = tmp_path / "summary.txt"
    result = printer.main(
        ["--input-dir", str(root), "--output", str(output)]
    )
    text = output.read_text(encoding="utf-8")

    assert result == 0
    assert "files discovered: 1" in text
    assert "姓名=张三" in text
    assert "医院=华山医院" in text
    assert "subject: sub002" in text
    assert "decoded_as:" in text
    assert "notes.dat" not in text
    assert "preload" not in text

    result_all = printer.main(
        [
            "--input-dir",
            str(root),
            "--output",
            str(tmp_path / "all.txt"),
            "--all-dat",
        ]
    )
    all_text = (tmp_path / "all.txt").read_text(encoding="utf-8")
    assert result_all == 0
    assert "files discovered: 2" in all_text
    assert "李四" in all_text
    assert "preload" not in all_text


def test_binary_mixed_file_extracts_chinese_strings(tmp_path: Path) -> None:
    root = tmp_path / "story_listen_v2"
    path = root / "sub001" / "postseegct" / "StudyInfo.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01STUDY\x00" + SAMPLE_TEXT.encode("gb18030") + b"\xff\xfe")
    output = tmp_path / "mixed.txt"
    assert printer.main(["--input-dir", str(root), "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "张三" in text
    assert "术后CT对照" in text


def test_missing_directory_returns_error(tmp_path: Path) -> None:
    assert printer.main(["--input-dir", str(tmp_path / "missing")]) == 2
