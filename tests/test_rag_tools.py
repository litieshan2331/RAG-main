from app.core.config import Settings
from app.rag.tools import WebSearchTool


def test_web_search_tool_converts_tavily_results() -> None:
    tool = WebSearchTool(settings=Settings(tavily_api_key="test-key"))

    result = tool._to_result(
        {
            "answer": "公开资料显示该政策已经发布。",
            "results": [
                {
                    "title": "Policy notice",
                    "url": "https://example.com/policy",
                    "content": "Policy content",
                    "score": 0.83,
                }
            ],
        }
    )

    assert result.strategy == "web_search"
    assert result.tool_name == "web_search"
    assert "Tavily summary" in result.content
    assert result.contexts[0].source == "https://example.com/policy"
    assert result.contexts[0].metadata["title"] == "Policy notice"
