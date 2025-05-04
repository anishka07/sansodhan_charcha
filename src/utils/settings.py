from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class PathSettings:
    LOG_DIR: Path = BASE_DIR / "logs"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"
    CACHE_DIR: Path = CHROMA_DIR / "cache"
    PDF_DIR: Path = BASE_DIR / "pdfs"


@dataclass(frozen=True)
class ConstantSettings:
    PERSIST_DIRECTORY: Path = BASE_DIR / "chroma_db"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHUNK_SIZE: int = 150
    CHUNK_OVERLAP: int = 20
    RESPONSE_PROMPT: str = """You are a rag. this is the user context {} and this is the user query {}. you are only supposed to reply in nepali language."""
