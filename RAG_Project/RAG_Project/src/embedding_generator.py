from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name ="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def generate_query_embedding(self, query):
        return self.model.encode(query)