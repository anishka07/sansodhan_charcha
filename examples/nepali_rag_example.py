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

    process_results = r.process_documents(document_paths)
    print("Process Results:", process_results)

    query = "नयाँ कम्पनी ऐन २०७९ मा के परिवर्तन भएका छन्?"
    search_results = r.search_law_amendments(query)

    print(f"\nSearch Query: {query}")
    print(f"Found {search_results['total_results']} relevant chunks from {len(search_results['documents'])} documents")

    print("\nTop Results Context:")
    print(search_results['context'])
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")