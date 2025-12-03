from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    groq_api_base: str = "https://api.groq.com"
    vector_store: str = "chroma"
    data_dir: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"

settings = Settings()
