from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import settings

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

def build_vectorstore(documents):
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if settings.vector_store == "chroma":
        vectordb = Chroma.from_documents(documents, emb)
    else:
        from langchain.vectorstores import FAISS
        vectordb = FAISS.from_documents(documents, emb)
    return vectordb
