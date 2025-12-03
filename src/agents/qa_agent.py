from .base_agent import BaseAgent
from ..utils import call_groq

class QARetrievalAgent(BaseAgent):
    def __init__(self, retriever):
        self.retriever = retriever

    async def run(self, query: str) -> str:
        docs = await self.retriever.ainvoke(query)
        context_text = "\n\n".join([d.page_content for d in docs[:6]])
        prompt = (
            "Use the context to answer the question. If answer not in context, reply 'I don't know'.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        )
        loop = __import__('asyncio').get_event_loop()
        resp = await loop.run_in_executor(None, call_groq, prompt)
        return resp['choices'][0]['message']['content'].strip()
