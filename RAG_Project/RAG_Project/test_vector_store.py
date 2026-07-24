from src.document_loader import DocumentLoader
from src.text_splitter import TextChunker
from src.vector_store import VectorStore

loader = DocumentLoader("data/Sample_pdf.pdf")
documents = loader.load_documents()

chunker = TextChunker()
chunks = chunker.split_documents(documents)

vector_store = VectorStore()

db = vector_store.create_vector_store(chunks)

print("Vector Store Created Successfully!")

print(f"Number of Chunks Stored: {db._collection.count()}")