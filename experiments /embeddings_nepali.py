from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode("नेपाली भाषा सुन्दर छ।")

print(embeddings)
