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

    decision = service.assess_dependency("发动机异响怎么处理？", [])

    assert decision.depends_on_history is False
    assert llm.calls == 0


def test_follow_up_is_marked_as_history_dependent() -> None:
    llm = _FakeLlm(
        ['{"depends_on_history":true,"reasoning":"这个指代上一轮车型","confidence":0.94}']
    )
    service = ConversationContextService(llm=llm, settings=_settings())
    history = [ConversationTurn(role="user", content="介绍一下 Model Y 的续航")]

    decision = service.assess_dependency("那它冬天呢？", history)

    assert decision.depends_on_history is True
    assert decision.confidence == 0.94


def test_low_confidence_dependency_is_treated_as_independent() -> None:
    llm = _FakeLlm(
        ['{"depends_on_history":true,"reasoning":"可能有关","confidence":0.4}']
    )
    service = ConversationContextService(llm=llm, settings=_settings(threshold=0.6))
    history = [ConversationTurn(role="user", content="介绍一下 Model Y")]

    decision = service.assess_dependency("比亚迪海豹续航是多少？", history)

    assert decision.depends_on_history is False


def test_follow_up_rewrite_returns_standalone_question() -> None:
    llm = _FakeLlm(["Model Y 在冬季低温环境下的续航表现如何？"])
    service = ConversationContextService(llm=llm, settings=_settings())
    history = [ConversationTurn(role="user", content="介绍一下 Model Y 的续航")]

    rewritten = service.rewrite_follow_up("那它冬天呢？", history)

    assert rewritten == "Model Y 在冬季低温环境下的续航表现如何？"
