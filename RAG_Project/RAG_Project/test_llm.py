from src.llm_loader import LLMLoader

llm = LLMLoader()
question = question = """
Answer the following question in 5-6 sentences.
Question:
What is Retrieval Augmented Generation?
"""
answer = llm.generate_response(question)
print(answer)