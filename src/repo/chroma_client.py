import os
from pathlib import Path
from threading import Lock

from chromadb import Client
from chromadb.config import Settings

from src.utils.settings import PathSettings


class ChromaClientSingleton:
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls, persist_directory: Path=PathSettings.CHROMA_DIR):
        os.makedirs(persist_directory, exist_ok=True)
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ChromaClientSingleton, cls).__new__(cls)
                cls._instance._initialize(persist_directory)
        return cls._instance
    
    def _initialize(self, persist_directory: Path=PathSettings.CHROMA_DIR):
        self.persist_directory = str(persist_directory)
        self.client = Client(Settings(persist_directory=self.persist_directory, anonymized_telemetry=False))

    def create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(collection_name)

    def get_client(self):
        return self.client

