from speech_strf.alignments import parse_textgrid, validate_intervals


def test_textgrid_conversion_and_validation(tmp_path):
    path = tmp_path / "sample.TextGrid"
    path.write_text(
        '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "你好"
        intervals [2]:
            xmin = 1
            xmax = 2
            text = "世界"
''',
        encoding="utf-8",
    )
    intervals = parse_textgrid(path)
    assert [row.label for row in intervals] == ["你好", "世界"]
    assert validate_intervals(intervals, 2.0) == []
    assert validate_intervals(intervals, 1.5)[0]["code"] == "outside_audio_duration"

