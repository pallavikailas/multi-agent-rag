from deepagents import create_deep_agent

def build_rag_deep_agent(qa_tool, summarizer_tool, model=None):
    """
    qa_tool: async function (query: str, docs: list) -> answer string
    summarizer_tool: async function (docs: list) -> summary string
    model: optional LangChain chat model (Groq or other)

    Returns a ready-to-run deep agent.
    """
    tools = [qa_tool, summarizer_tool]
    instructions = (
        "You are an assistant that answers queries using the 'qa_tool' on provided documents, "
        "and then optionally uses 'summarizer_tool' to produce a concise summary. "
        "When answering a user query, first read the documents, then answer."
    )
    agent = create_deep_agent(
        tools=tools,
        system_prompt=instructions,
        model=model
    )
    return agent
