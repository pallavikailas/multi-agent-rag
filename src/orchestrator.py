import asyncio
from .agents.summarizer import SummarizerAgent
from .agents.qa_agent import QARetrievalAgent

class DeepOrchestrator:
    def __init__(self, retriever):
        self.retriever = retriever
        self.summarizer = SummarizerAgent()
        self.qa = QARetrievalAgent(retriever)

    async def handle_query(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        tasks = [
            asyncio.create_task(self.summarizer.run(docs)),
            asyncio.create_task(self.qa.run(query)),
        ]
        summary, answer = await asyncio.gather(*tasks)
        return {"summary": summary, "answer": answer}
