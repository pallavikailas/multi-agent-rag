from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from .config import settings

def build_vectorstore(documents):
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if settings.vector_store == "chroma":
        vectordb = Chroma.from_documents(documents, emb)
    else:
        from langchain.vectorstores import FAISS
        vectordb = FAISS.from_documents(documents, emb)
    return vectordb
