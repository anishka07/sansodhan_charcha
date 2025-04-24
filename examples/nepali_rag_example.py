import re 
import time

from src.services.rag_main import NepaliLawRAG
from src.utils.settings import ConstantSettings, PathSettings


if __name__ == "__main__":
    start_time = time.time()
    r = NepaliLawRAG(
        chunk_size=ConstantSettings.CHUNK_SIZE,
        chunk_overlap=ConstantSettings.CHUNK_OVERLAP,
        model_name=ConstantSettings.EMBEDDING_MODEL,
    )

    document_paths = [PathSettings.PDF_DIR / "example1.pdf"]

    extracted_text = r.extract_text_from_documents(document_list=document_paths)
    a = list(extracted_text.values())
    b = a[0]
    b = re.sub(r'\.{2,}', '', b)
    print(b)

    print(f"Total time taken: {time.time() - start_time}")
