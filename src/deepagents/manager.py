from deepagents import DeepAgent

def build_deep_agent(qa_func, summary_func):
    agent = DeepAgent(
        name="RAG-Agent",
        system_prompt="You are a RAG supervisor agent. Use tools to answer queries."
    )

    @agent.tool
    async def qa_tool(query: str, docs: list):
        """Answer a question using retrieved documents."""
        return await qa_func(query, docs)

    @agent.tool
    async def summary_tool(docs: list):
        """Summarize retrieved documents."""
        return await summary_func(docs)

    return agent
