from src.document_loader import DocumentLoader
from src.text_splitter import TextChunker


loader = DocumentLoader("data/Sample_pdf.pdf")

documents = loader.load_documents()

chunker = TextChunker()

chunks = chunker.split_documents(documents)

print(f"Pages Loaded: {len(documents)}")
print(f"Chunks Created: {len(chunks)}")

print("\nFirst Chunk:\n")

print(chunks[0].page_content)

print("\nMetadata:\n")

print(chunks[0].metadata)