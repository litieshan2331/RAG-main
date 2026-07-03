from types import SimpleNamespace

from app.rag.conversation import ConversationContextService, ConversationTurn


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLlm:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, _prompt: str) -> _Message:
        response = self.responses[self.calls]
        self.calls += 1
        return _Message(response)


def _settings(threshold: float = 0.6):
    return SimpleNamespace(
        context_dependency_threshold=threshold,
        conversation_history_max_chars=8000,
    )


def test_no_history_uses_original_without_llm_call() -> None:
    llm = _FakeLlm([])
    service = ConversationContextService(llm=llm, settings=_settings())

    result = service.contextualize("发动机异响怎么处理？", [])

    assert result.decision.depends_on_history is False
    assert result.standalone_query == "发动机异响怎么处理？"
    assert llm.calls == 0


def test_first_turn_multi_hop_question_is_decomposed() -> None:
    llm = _FakeLlm(
        [
            '{"depends_on_history":false,"standalone_query":"模型不应改写原问题",'
            '"requires_decomposition":true,"sub_queries":['
            '{"id":"q1","query":"查询最近保养记录","depends_on":[]},'
            '{"id":"q2","query":"查询手册保养周期","depends_on":[]},'
            '{"id":"q3","query":"比较记录与周期","depends_on":["q1","q2"]}],'
            '"reasoning":"需要查询两项事实后比较","confidence":0.96}'
        ]
    )
    service = ConversationContextService(llm=llm, settings=_settings())
    query = "结合我的最近保养记录和手册周期，判断是否逾期。"

    result = service.contextualize(query, [])

    assert result.standalone_query == query
    assert result.decision.requires_decomposition is True
    assert len(result.decision.sub_queries) == 3
    assert result.decision.sub_queries[2]["depends_on"] == ["q1", "q2"]
    assert llm.calls == 1


def test_follow_up_is_classified_and_rewritten_in_one_call() -> None:
    llm = _FakeLlm(
        [
            '{"depends_on_history":true,'
            '"standalone_query":"Model Y 在冬季低温环境下的续航表现如何？",'
            '"reasoning":"它指代上一轮车型","confidence":0.94}'
        ]
    )
    service = ConversationContextService(llm=llm, settings=_settings())
    history = [ConversationTurn(role="user", content="介绍一下 Model Y 的续航")]

    result = service.contextualize("那它冬天呢？", history)

    assert result.decision.depends_on_history is True
    assert result.decision.confidence == 0.94
    assert result.standalone_query == "Model Y 在冬季低温环境下的续航表现如何？"
    assert llm.calls == 1


def test_independent_question_is_never_rewritten() -> None:
    llm = _FakeLlm(
        [
            '{"depends_on_history":false,"standalone_query":"模型擅自修改的问题",'
            '"reasoning":"新话题","confidence":0.98}'
        ]
    )
    service = ConversationContextService(llm=llm, settings=_settings())
    history = [ConversationTurn(role="user", content="介绍一下 Model Y")]

    result = service.contextualize("比亚迪海豹续航是多少？", history)

    assert result.decision.depends_on_history is False
    assert result.standalone_query == "比亚迪海豹续航是多少？"


def test_low_confidence_dependency_uses_original_query() -> None:
    llm = _FakeLlm(
        [
            '{"depends_on_history":true,"standalone_query":"可能相关的改写",'
            '"reasoning":"可能有关","confidence":0.4}'
        ]
    )
    service = ConversationContextService(llm=llm, settings=_settings(threshold=0.6))
    history = [ConversationTurn(role="user", content="介绍一下 Model Y")]

    result = service.contextualize("比亚迪海豹续航是多少？", history)

    assert result.decision.depends_on_history is False
    assert result.standalone_query == "比亚迪海豹续航是多少？"


def test_missing_rewrite_for_dependent_question_falls_back_safely() -> None:
    llm = _FakeLlm(
        ['{"depends_on_history":true,"standalone_query":"","reasoning":"存在指代","confidence":0.9}']
    )
    service = ConversationContextService(llm=llm, settings=_settings())
    history = [ConversationTurn(role="user", content="介绍一下 Model Y")]

    result = service.contextualize("它呢？", history)

    assert result.decision.depends_on_history is False
    assert result.standalone_query == "它呢？"
    assert result.decision.confidence == 0.0


def test_contextualizer_prompt_combines_dependency_and_rewrite() -> None:
    prompt = ConversationContextService._contextualizer_prompt("它呢？", "user: 介绍一下 Model Y")

    assert "depends_on_history" in prompt
    assert "standalone_query" in prompt
    assert "Multi-Query" in prompt
    assert "requires_decomposition" in prompt
    assert "sub_queries" in prompt
    assert "{{HISTORY}}" not in prompt
    assert "{{USER_QUERY}}" not in prompt
