from types import SimpleNamespace

from app.rag.query_router import QueryRouterService


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self.response)


def test_query_router_parses_standard_decision_and_slots() -> None:
    decision = QueryRouterService.parse_response(
        """
        ```json
        {
          "intent": "售后维修与保养：查询保险到期时间",
          "strategy": "relational_db",
          "reasoning": "问题涉及用户车辆保险字段",
          "confidence": 0.92,
          "slots": {"related": true, "domain_intent": "售后维修与保养"}
        }
        ```
        """
    )

    assert decision.intent == "售后维修与保养：查询保险到期时间"
    assert decision.strategy == "relational_db"
    assert decision.confidence == 0.92
    assert decision.slots["domain_intent"] == "售后维修与保养"


def test_query_router_normalizes_alias_and_confidence() -> None:
    decision = QueryRouterService.parse_response(
        '{"intent":"查询关系","strategy":"neo4j","reasoning":"实体关系","confidence":2}'
    )

    assert decision.strategy == "graph_db"
    assert decision.confidence == 1.0


def test_query_router_normalizes_web_search_alias() -> None:
    decision = QueryRouterService.parse_response(
        '{"intent":"查询最新政策","strategy":"tavily","reasoning":"需要公开网络信息","confidence":0.7}'
    )

    assert decision.strategy == "web_search"
    assert decision.confidence == 0.7


def test_query_router_normalizes_react_alias() -> None:
    decision = QueryRouterService.parse_response(
        '{"intent":"综合分析故障原因","strategy":"multi_tool","reasoning":"需要多工具综合","confidence":0.86}'
    )

    assert decision.strategy == "react_agent"
    assert decision.confidence == 0.86


def test_query_router_falls_back_to_knowledge_base_on_bad_json() -> None:
    decision = QueryRouterService.parse_response("not json")

    assert decision.strategy == "knowledge_base"
    assert decision.confidence == 0.0


def test_query_router_demotes_plain_part_question_from_graph() -> None:
    router = QueryRouterService(
        llm=FakeLlm(
            '{"intent":"车辆使用与技术指导：查询后排部件",'
            '"strategy":"graph_db","reasoning":"提到部件","confidence":0.8}'
        )
    )

    decision = router.route("后排常用部件有什么？")

    assert decision.strategy == "knowledge_base"
    assert decision.confidence == 0.9
    assert "未要求实体关系" in decision.reasoning


def test_query_router_keeps_explicit_relationship_question_on_graph() -> None:
    router = QueryRouterService(
        llm=FakeLlm(
            '{"intent":"查询部件关系","strategy":"graph_db",'
            '"reasoning":"明确查询关系","confidence":0.95}'
        )
    )

    decision = router.route("后排座椅与安全带之间有什么约束关系？")

    assert decision.strategy == "graph_db"
    assert decision.confidence == 0.95


def test_query_router_prompt_contains_domain_taxonomy_and_boundary_example() -> None:
    prompt = QueryRouterService._prompt("后排常用部件有什么？")

    assert "售前咨询与购买" in prompt
    assert "车辆使用与技术指导" in prompt
    assert "后排常用部件有什么" in prompt
    assert "并不等于图查询" in prompt
    assert "{{USER_QUERY}}" not in prompt


def test_query_router_fallback_order_starts_with_selected_strategy() -> None:
    router = QueryRouterService(llm=object())
    decision = QueryRouterService.parse_response(
        '{"intent":"查询图关系","strategy":"graph_db","reasoning":"关系查询","confidence":0.8}'
    )

    assert router.fallback_order(decision) == ["graph_db", "knowledge_base", "relational_db", "web_search"]


def test_query_router_react_fallback_order_stays_inside_agent() -> None:
    router = QueryRouterService(llm=object())
    decision = QueryRouterService.parse_response(
        '{"intent":"综合分析","strategy":"react_agent","reasoning":"多工具","confidence":0.8}'
    )

    assert router.fallback_order(decision) == ["react_agent"]


def test_query_router_force_knowledge_base_disables_other_fallbacks() -> None:
    router = QueryRouterService(llm=object())
    decision = QueryRouterService.parse_response(
        '{"intent":"查询最新政策","strategy":"web_search","reasoning":"实时信息","confidence":0.8}'
    )

    assert router.fallback_order(decision, force_knowledge_base=True) == ["knowledge_base"]
