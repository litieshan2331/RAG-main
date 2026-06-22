from app.ingestion.markdown_cleaner import MarkdownCleaner


def test_cleaner_preserves_images_and_collapses_blank_lines() -> None:
    raw = "line\n\n\n![](images/a.png)\n"
    cleaned = MarkdownCleaner().clean(raw)

    assert "\n\n\n" not in cleaned
    assert "![a.png](images/a.png)" in cleaned
