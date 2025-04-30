from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.services.rag_base import NepaliRAGBase


class NepaliLawRAG(NepaliRAGBase):

    def __init__(
            self,
            model_name: str,
            chunk_size: int,
            use_heavy: bool,
            chunk_overlap: int,
            collection_name: str = None
    ):
        """

        :param model_name: sentence transformer model name
        :param chunk_size: chunk size
        :param use_heavy: the sentence transformer model which i've used is very heavy so i have this parameter to prevent from calling the instance everytime (even while testing the code)
         which makes my computer slow
        :param chunk_overlap: size of overlap between chunks
        :param collection_name: chromadb collection name
        """
        super().__init__(chunk_size, model_name, use_heavy)
        self.chunk_overlap = chunk_overlap
        # remove this 
        if use_heavy:
            if collection_name is None:
                self.collection_name = "law_chroma_collection"        
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Nepali law amendments and legal documents"},
            )
            self.logger.info(f"Chroma db collection {self.collection_name} initialized.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ]
        )

    def process_documents(self, document_paths: List[Path]):
        """

        :param document_paths: list of pdf document paths
        :return: dictionary of information on the extracted text
        """
        self.logger.info(f"Processing {len(document_paths)} documents...")
        extracted_text = self.extract_text_from_documents(document_paths)

        results = {}
        for doc_name, text in extracted_text.items():
            if not text:
                self.logger.warning(f"Document {doc_name} has no text. Skipping.")
                continue
            self.logger.info(f"Processing text for {doc_name}...")
            # Loading existing chunks, embeddings and chunk metadata from pickle files
            chunks = self._load_cache(doc_name, "chunks")
            embeddings = self._load_cache(doc_name, "embeddings")
            chunk_metadata = self._load_cache(doc_name, "metadata")

            if chunks is not None and embeddings is not None and chunk_metadata is not None:
                self.logger.info("Using cached chunks, embeddings and metadata.")
            else:
                chunks = self.chunk_text(text)
                chunk_metadata = [{
                    "doc_name": doc_name,
                    "chunk_id": i,
                    "source": f"{doc_name}_{i}_chunk"
                } for i, _ in enumerate(chunks)]

                embeddings = self.embed_text(chunks)
                # Saving the chunks, embeddings and chunk metadata as pickle files
                self._save_cache(doc_name, chunks, "chunks")
                self._save_cache(doc_name, embeddings, "embeddings")
                self._save_cache(doc_name, chunk_metadata, "metadata")

            doc_id = self.save_embeddings(chunks, embeddings, chunk_metadata)
            results[doc_name] = {
                "text_len": len(text),
                "chunks": chunks,
                "doc_id": doc_id
            }
        self.logger.info(f"Finished processing {len(results)} documents.")
        return results

    def chunk_text(self, text: str) -> List[str]:
        """

        :param text: text to chunk
        :return: list of chunks
        """
        self.logger.info(f"Chunking text of {len(text)}...")
        chunks = self.text_splitter.split_text(text)
        self.logger.info(f"Chunked text in {len(chunks)} chunks...")
        return chunks

    def embed_text(self, chunked_text: list[str]) -> np.ndarray:
        """

        :param chunked_text: list of chunks
        :return: numpy array of embeddings
        """
        self.logger.info(f"Embedding text of {len(chunked_text)}...")
        embeddings = [self.embeddings_gen_instance.encode(chunk) for chunk in tqdm(chunked_text, desc="Embedding text")]
        return np.array(embeddings)

    def save_embeddings(self, chunks: List[str], embeddings: np.ndarray, metadata: List[Dict]) -> str:
        """

        :param chunks: list of chunks
        :param embeddings: numpy array of embeddings
        :param metadata: list of metadata (dictionary)
        :return: batch id
        """
        self.logger.info(f"Saving embeddings of {len(chunks)} chunks...")

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
        """

        :param query: user query
        :param top_k: number of results to retrieve
        :return: results as a list of dictionaries
        """
        self.logger.info(f"Retrieving results for {query}...")
        query_em = self.embeddings_gen_instance.encode(query)
        results = self.collection.query(
            query_embeddings=[query_em.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        formatted_results = [{
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": 1 - results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))]
        return formatted_results

    def get_context_for_query(self, query: str, top_k: int = 5):
        """

        :param query: user query
        :param top_k: number of results to retrieve
        :return: context as a list
        """
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
        """

        :param query: user query
        :param top_k: number of results to retrieve
        :return: dictionary with results
        """
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
