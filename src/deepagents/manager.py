from deepagents import Agent, Orchestrator


class QAADeepAgent(Agent):
    def __init__(self, qa_func):
        super().__init__("qa_agent")
        self.qa_func = qa_func

    async def run(self, state):
        answer = await self.qa_func(state["query"], state["docs"])
        return {"answer": answer}


class SummDeepAgent(Agent):
    def __init__(self, summarizer_func):
        super().__init__("summarizer_agent")
        self.summarizer_func = summarizer_func

    async def run(self, state):
        summary = await self.summarizer_func(state["docs"])
        return {"summary": summary}


class RAGOrchestrator(Orchestrator):
    def route(self, state):
        # Determine next agent based on state
        if "docs" not in state:
            return "retriever"

        if "answer" not in state:
            return "qa_agent"

        return "summarizer_agent"
