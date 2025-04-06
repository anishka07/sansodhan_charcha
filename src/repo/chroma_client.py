import os 
from threading import Lock

from chromadb import Client
from chromadb.config import Settings

from src.utils.settings import ConstantSettings


class ChromaClientSingleton:
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls, persist_directory: str=ConstantSettings.PERSIST_DIRECTORY):
        os.makedirs(persist_directory, exist_ok=True)
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ChromaClientSingleton, cls).__new__(cls)
                cls._instance._intitalize(persist_directory)
        return cls._instance
    
    def _intitalize(self, persist_directory: str=ConstantSettings.PERSIST_DIRECTORY):
        self.persist_directory = persist_directory
        self.client = Client(Settings(persist_directory=self.persist_directory, anonymized_telemetry=False))
    
    def get_client(self):
        return self.client
