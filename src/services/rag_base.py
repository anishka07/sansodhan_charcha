import re
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from src.repo.chroma_client import ChromaClientSingleton
from src.utils.logger import get_custom_logger
from src.utils.settings import PathSettings


class STSingleton:
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls, model_name: str):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(STSingleton, cls).__new__(cls)
                cls._instance._initialize(model_name)
        return cls._instance

    def _initialize(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def get_model(self):
        return self.model


class NepaliRAGBase:

    def __init__(self, chunk_size: int, model_name: str, use_heavy: bool = True):
        self.chunk_size = chunk_size
        self.save_dir_path = PathSettings.CACHE_DIR
        self.logger = get_custom_logger(name="Sansodhan Charcha")
        # remove this 
        if use_heavy:
            self.logger.info("Initializing ChromaDB client and SentenceTransformer model...")
            self.embeddings_gen_instance = STSingleton(model_name).get_model()
            self.chroma_client = ChromaClientSingleton().get_client()
            self.logger.info("ChromaDB client and SentenceTransformer model initialized successfully.")

    def _get_cache_path(self, doc_name: str, suffix: str = ""):
        cache_path = self.save_dir_path / f"{doc_name}{f'_{suffix if suffix else ''}'}.pkl"
        return cache_path

    def _load_cache(self, doc_name: str, suffix: str = ""):
        cache_path = self._get_cache_path(doc_name, suffix)
        if os.path.exists(cache_path):
            self.logger.info(f"Loading cache from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, doc_name: str, data: object, suffix: str = ""):
        cache_path = self._get_cache_path(doc_name, suffix)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(cache_path):
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Saving to {cache_path}")

    @staticmethod
    def _get_doc_name(document_path: Path):
        doc_name = str(document_path).split("/")[-1]
        doc_name = doc_name.split(".")[0]
        return doc_name

    def extract_text_from_documents(self, document_list: list[Path]) -> Dict[str, Optional[str]]:
        results = {}
        for document_path in document_list:
            doc_name = self._get_doc_name(document_path)
            cache_data = self._load_cache(doc_name, "text")

            if cache_data is not None:
                self.logger.info(f"Loading cache for {doc_name}")
                results[doc_name] = cache_data
                continue

            extracted_text = ""
            if document_path.suffix.lower() == ".pdf":
                try:
                    with fitz.open(document_path) as doc:
                        for page in doc:
                            extracted_text += page.get_text()
                            extracted_text = " ".join(extracted_text.split())
                            extracted_text = re.sub(r'\.{2,}', '', extracted_text)
                    self.logger.info(f"Saving cache for {doc_name}")
                    self._save_cache(doc_name, extracted_text, "text")
                    results[doc_name] = extracted_text
                except Exception as e:
                    self.logger.error(f"Error reading PDF {doc_name}: {e}")
                    results[doc_name] = ""
        return results

    def chunk_text(self, text: str) -> list[str]:
        raise NotImplementedError("Chunking method not implemented.")

    def embed_text(self, chunked_text: list[str]) -> np.ndarray:
        raise NotImplementedError("Embedding method not implemented.")

    def save_embeddings(self, chunks: List[str], embeddings: np.ndarray, metadata: List[Dict]) -> str:
        raise NotImplementedError("Saving embeddings method not implemented.")

    def retrieve_results(self, query: str, top_k: int):
        raise NotImplementedError("Retrieving results method not implemented.")
