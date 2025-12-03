from .base_agent import BaseAgent
from ..utils import call_groq

class SummarizerAgent(BaseAgent):
    async def run(self, documents: list) -> str:
        text = "\n\n".join([d.page_content for d in documents[:6]])
        prompt = f"Summarize the following context briefly and produce a concise TL;DR:\n\n{text}\n\nTL;DR:"
        loop = __import__('asyncio').get_event_loop()
        resp = await loop.run_in_executor(None, call_groq, prompt)
        return resp['choices'][0]['message']['content'].strip()
