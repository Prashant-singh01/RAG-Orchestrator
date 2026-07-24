from src.document_loader import DocumentLoader
from src.text_splitter import TextChunker
from src.embedding_generator import EmbeddingGenerator


loader = DocumentLoader("data/Sample_pdf.pdf")
documents = loader.load_documents()

chunker = TextChunker()
chunks = chunker.split_documents(documents)

embedder = EmbeddingGenerator()

embeddings = embedder.generate_embeddings(chunks)

print(f"Total Chunks: {len(chunks)}")

print(f"Embedding Shape: {embeddings.shape}")

print("\nFirst 10 Values:\n")

print(embeddings[0][:10])