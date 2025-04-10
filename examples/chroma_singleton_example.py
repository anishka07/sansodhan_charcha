import time 

from src.repo.chroma_client import ChromaClientSingleton
from src.utils.settings import ConstantSettings


if __name__ == "__main__":
    start_time = time.time()
    chroma_instance1 = ChromaClientSingleton(
        persist_directory=ConstantSettings.PERSIST_DIRECTORY
    )

    chroma_instance2 = ChromaClientSingleton(
        persist_directory=ConstantSettings.PERSIST_DIRECTORY
    )

    assert chroma_instance1 is chroma_instance2, "Singleton instances are not the same!"
    print("Both instances are the same. Singleton pattern works!")
    print(f"Chroma client initialized in {time.time() - start_time:.2f} seconds")
