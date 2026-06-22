from app.ingestion.mineru import ImageAsset, MinerUConverter


def test_replace_image_urls_prefers_generated_description() -> None:
    markdown = "See ![old alt](images/a.png)."

    replaced = MinerUConverter._replace_image_urls(
        markdown,
        {"a.png": ImageAsset(url="http://minio/know-engine/a.png", description="仪表盘故障灯示意图")},
    )

    assert "![仪表盘故障灯示意图](http://minio/know-engine/a.png)" in replaced


def test_replace_image_urls_falls_back_to_existing_alt() -> None:
    markdown = "See ![old alt](images/a.png)."

    replaced = MinerUConverter._replace_image_urls(
        markdown,
        {"a.png": ImageAsset(url="http://minio/know-engine/a.png")},
    )

    assert "![old alt](http://minio/know-engine/a.png)" in replaced
