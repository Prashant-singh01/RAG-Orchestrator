from src.document_loader import DocumentLoader

loader = DocumentLoader("data/Sample_pdf.pdf")
documents = loader.load_documents()
print(f"Total Pages: {len(documents)}")
print("\n")
print(documents[0].page_content[:500])
