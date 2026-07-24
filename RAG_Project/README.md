# RAG Project

A beginner-to-intermediate Retrieval Augmented Generation (RAG) application built with Python, Hugging Face, LangChain, ChromaDB, and Streamlit.

## Architecture Diagram

```text
                PDF
                 │
                 ▼
         Document Loader
                 │
                 ▼
           Text Splitter
                 │
                 ▼
            Embeddings
                 │
                 ▼
             ChromaDB
                 ▲
                 │
           User Question
                 │
                 ▼
            Retriever
                 │
                 ▼
           Top K Chunks
                 │
                 ▼
         Prompt Template
                 │
                 ▼
         Hugging Face LLM
                 │
                 ▼
            Final Answer
```

## Technology Stack

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- ChromaDB
- PyPDF

## Folder Structure

```text
RAG_Project/
├── app.py
├── config.py
├── evaluation.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── document_loader.py
    ├── embedding_generator.py
    ├── llm_loader.py
    ├── logger.py
    ├── prompt_template.py
    ├── rag_pipeline.py
    ├── retriever.py
    ├── text_splitter.py
    ├── utils.py
    └── vector_store.py
```

## Installation

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project depends on the libraries listed in the requirements file, including Streamlit, LangChain, Hugging Face Transformers, and ChromaDB.

## How to Run

```bash
python app.py
```

or

```bash
streamlit run app.py
```

## Example Output

```text
Answer: Retrieval Augmented Generation (RAG) combines retrieval with generation...
```

## Future Improvements

- Add automated evaluation datasets
- Add support for more document types
- Improve prompt tuning and grounding
- Add asynchronous inference and better caching

## Screenshots Placeholder

Add screenshots of the Streamlit interface here.

## License

This project is provided for educational and demonstration purposes.

## Contribution

Contributions are welcome. Please open an issue or pull request with a clear description of the change.
