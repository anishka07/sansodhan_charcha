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
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHUNK_SIZE: int = 150
    PERSIST_DIRECTORY: str = "chroma_db"
