import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .orchestrator import DeepOrchestrator
from .config import settings

def build_system():
    docs = load_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    vectordb = build_vectorstore(chunks)
    return vectordb

async def demo_query(query: str):
    vectordb = build_system()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    orchestrator = DeepOrchestrator(retriever)
    out = await orchestrator.handle_query(query)
    print("--- Summary ---")
    print(out['summary'])
    print("--- Answer ---")
    print(out['answer'])

if __name__ == '__main__':
    import os
    q = os.environ.get('DEMO_QUERY', 'What is the main point of the documents?')
    asyncio.run(demo_query(q))
