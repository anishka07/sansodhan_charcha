import time
import json

from src.services.rag_main import NepaliLawRAG
from src.utils.settings import ConstantSettings, PathSettings

if __name__ == "__main__":
    start_time = time.time()
    r = NepaliLawRAG(
        chunk_size=ConstantSettings.CHUNK_SIZE,
        model_name=ConstantSettings.EMBEDDING_MODEL,
        chunk_overlap=ConstantSettings.CHUNK_OVERLAP,
        use_heavy=True
    )

    document_paths = [PathSettings.PDF_DIR / "monopoly.pdf", PathSettings.PDF_DIR / "example1.pdf"]

    a = r.process_documents(document_paths=document_paths)
    # with open("eg.json", "w") as f:
    #     json.dump(a, f, indent=4)
    # print("done")
    print(a['example1']['chunks'])

    print(f"Total time taken: {time.time() - start_time}")
