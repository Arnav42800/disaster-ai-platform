from disaster_ai.progress import format_progress


def test_format_progress_reports_fraction_and_percent():
    rendered = format_progress(current=5, total=10, width=10)

    assert "50%" in rendered
    assert "#####-----" in rendered


def test_format_progress_clamps_completed_work():
    rendered = format_progress(current=12, total=10, width=10)

    assert "100%" in rendered
    assert "##########" in rendered
