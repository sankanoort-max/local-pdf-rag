from chains.rag_chain import NOT_FOUND_MESSAGE, RAGChain
from memory.chat_history import ChatHistory

CHUNKS = [
    {"id": "1", "text": "Refunds are allowed within 30 days.", "metadata": {"file_name": "policy.pdf", "page": 2}, "score": 0.8},
    {"id": "2", "text": "Contact support for returns.", "metadata": {"file_name": "policy.pdf", "page": 3}, "score": 0.6},
]


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.questions = []

    def retrieve(self, question, top_k=None, score_threshold=None, sources=None):
        self.questions.append(question)
        return self.chunks[:top_k]


from tests.fakes import FakeLLM  # noqa: E402


def test_answer_includes_sources_and_context():
    llm = FakeLLM("30 days [policy.pdf, p. 2]")
    chain = RAGChain(StubRetriever(CHUNKS), llm, top_k=2)
    result = chain.ask("How long do I have to get a refund?")

    assert result.answer == "30 days [policy.pdf, p. 2]"
    assert len(result.sources) == 2
    prompt = llm.calls[-1][1]["content"]
    assert "Refunds are allowed within 30 days." in prompt
    assert "[policy.pdf, p. 2]" in prompt
    assert llm.calls[-1][0]["role"] == "system"


def test_nothing_found_does_not_call_llm():
    llm = FakeLLM()
    result = RAGChain(StubRetriever([]), llm).ask("Unrelated question")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert llm.calls == []


def test_streaming_returns_tokens_and_sources():
    chain = RAGChain(StubRetriever(CHUNKS), FakeLLM("streamed answer here"), top_k=1)
    stream = chain.stream("refund?")
    assert "".join(stream.tokens).strip() == "streamed answer here"
    assert len(stream.sources) == 1


def test_follow_up_question_is_condensed_with_history():
    retriever = StubRetriever(CHUNKS)
    llm = FakeLLM("What is the refund period for returns?")
    history = ChatHistory()
    history.add_user("Tell me about refunds")
    history.add_assistant("Refunds are allowed within 30 days.")

    RAGChain(retriever, llm).ask("and for returns?", history)
    assert retriever.questions[-1] == "What is the refund period for returns?"


def test_history_is_trimmed():
    history = ChatHistory()
    for i in range(10):
        history.add_user(f"q{i}")
        history.add_assistant(f"a{i}")
    text = history.as_text(max_turns=2)
    assert "q9" in text and "q8" in text and "q7" not in text
