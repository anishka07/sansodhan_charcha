import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.services.rag_base import NepaliRAGBase
from src.utils.settings import ConstantSettings, PathSettings


class NepaliLawRAG(NepaliRAGBase):

    def __init__(
            self,
            model_name: str,
            chunk_size: int,
            chunk_overlap: int,
            collection_name: str = None
    ):
        super().__init__(chunk_size, model_name)
        self.chunk_overlap = chunk_overlap
        if collection_name is None:
            self.collection_name = "law_collection"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Nepali law amendments and legal documents"},
        )
        self.logger.info(f"Chromadb collection {self.collection_name} initialized.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_documents(
            self,
            document_path: List[Path]
    ) -> Dict[str, Dict[str, Any]]:

        self.logger.info(f"Processing {len(document_path)} documents...")

        document_texts = self.extract_text_from_documents(document_path)
        results = {}
        for doc_name, text in document_texts.items():
            if not text:
                self.logger.warning(f"No text found in {doc_name}. Skipping...")
                continue
            self.logger.info(f"Processing text for {doc_name}...")

            chunks = self.chunk_text(text)

            chunk_metadata = []
            for i, _ in enumerate(chunks):
                metadata = {
                    "doc_name": doc_name,
                    "chunk_id": i,
                    "source": f"{doc_name}_{i}_chunk"
                }
                chunk_metadata.append(metadata)

            emebeddings = self.embed_text(chunks)

            doc_id = self.save_embeddings(chunks, emebeddings, chunk_metadata)
            results[doc_name] = {
                "text_length": len(text),
                "chunks": chunks,
                "doc_id": doc_id,
            }
        self.logger.info(f"Successfully processed {len(results)} documents")
        return results

    def chunk_text(self, text: str) -> List[str]:
        self.logger.info(f"Chunking text of length {len(text)}...")
        chunks = self.text_splitter.split_text(text)
        self.logger.info(f"Text chunked into {len(chunks)} chunks.")
        return chunks

    def embed_text(self, chunked_text: List[str]) -> np.ndarray:
        self.logger.info(f"Generating embeddings for {len(chunked_text)} chunks...")
        embeddings = []
        for chunk in tqdm(chunked_text, desc="Embedding chunks"):
            embedding = self.embeddings_gen_instance.encode(chunk)
            embeddings.append(embedding)

        return np.array(embeddings)

    def save_embeddings(self, chunks: List[str], embeddings: np.ndarray, metadata: List[Dict]) -> str:
        self.logger.info(f"Saving {len(chunks)} chunks to ChromaDB")

        import uuid
        batch_id = str(uuid.uuid4())
        ids = [f"{batch_id}_{i}" for i in range(len(chunks))]

        for meta in metadata:
            meta["batch_id"] = batch_id

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )

        self.logger.info(f"Successfully saved embeddings with batch ID: {batch_id}")
        return batch_id

    def retrieve_results(self, query: str, top_k: int = 5):
        self.logger.info(f"Searching for {query} in ChromaDB...")
        query_embedding = self.embeddings_gen_instance.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]
            })
        self.logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve_results(query, top_k)

        results.sort(key=lambda x: x["similarity"], reverse=True)

        context = "\n\n---\n\n".join([
            f"Source: {r['metadata']['doc_name']}\n"
            f"Similarity: {r['similarity']:.2f}\n"
            f"Content: {r['text']}"
            for r in results
        ])

        return context

    def search_law_amendments(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.retrieve_results(query, top_k)

        grouped_results = {}
        for result in results:
            doc_name = result["metadata"]["doc_name"]
            if doc_name not in grouped_results:
                grouped_results[doc_name] = []
            grouped_results[doc_name].append(result)

        search_results = {
            "query": query,
            "total_results": len(results),
            "documents": grouped_results,
            "context": self.get_context_for_query(query, top_k)
        }

        return search_results
