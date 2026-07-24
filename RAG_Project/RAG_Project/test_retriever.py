from src.retriever import Retriever
retriever = Retriever()

query = "What is Lorem"
results = retriever.retrieve(query)
print(f"Retrieved {len(results)} chunks\n")

for i, doc in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print("=" * 60)
    print(doc.page_content[:300])
    print("\nMetadata:", doc.metadata)
    print()