from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from .config import settings

def load_documents(data_dir: str | None = None):
    data_dir = data_dir or settings.data_dir
    p = Path(data_dir)
    loader = DirectoryLoader(str(p), glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return chunks
