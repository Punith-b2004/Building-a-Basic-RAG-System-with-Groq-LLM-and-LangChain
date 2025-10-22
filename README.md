# Capstone Project: Building a Basic RAG System with Groq LLM and LangChain

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system using **Groq** for LLM inference, **LangChain** for orchestration, and **ChromaDB** as a vector database. The system integrates two custom tools—a Web Crawler and a Research Paper Scraper—to fetch and process data from web pages and PDF research papers, respectively. The goal is to demonstrate a real-world RAG workflow for answering queries (e.g., about NLP and Transformers) by retrieving relevant document chunks and generating grounded responses.

### Objectives

1. Load and test a Groq LLM with LangChain.
2. Build and test prompt templates for summarization.
3. Develop custom tools:
   - **Web Crawler**: Fetches text from URLs.
   - **Research Paper Scraper**: Extracts sections (abstract, introduction, conclusion) from PDFs.
4. Store document embeddings in ChromaDB for semantic search.
5. Implement a RAG pipeline to answer queries using retrieved context.

### Technologies

- **Python 3.x**
- **LangChain**: For LLM orchestration and RAG pipeline.
- **Groq**: High-speed LLM inference (Llama-3.1-8b-instant).
- **ChromaDB**: Vector database for embedding storage and retrieval.
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` for text embeddings.
- **BeautifulSoup4 & Requests**: For web scraping.
- **pdfplumber**: For PDF text extraction.

## Setup Instructions

### Prerequisites

- Python 3.8+
- A Groq API key (sign up at https://console.groq.com).
- The `transformer.pdf` file in the project directory (sample PDF with Transformer architecture content).

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:

   ```bash
   pip install langchain langchain-groq langchain-community langchain-huggingface groq chromadb beautifulsoup4 requests pdfplumber sentence-transformers
   ```

3. Set up the Groq API key:

   - Replace the placeholder key in the code with your own:

     ```python
     os.environ["GROQ_API_KEY"] = "your-groq-api-key"
     ```

   - Alternatively, set it as an environment variable:

     ```bash
     export GROQ_API_KEY="your-groq-api-key"
     ```

4. Ensure `transformer.pdf` is in the project root or update the `pdf_path` variable in the code.

### Directory Structure

```
├── transformer.pdf        # Sample PDF file
├── main.ipynb             # Jupyter notebook with project code
├── README.md              # This file
```

## Usage

1. **Run the Code**:

   - If using a Jupyter notebook, open `main.ipynb` and execute cells sequentially.

   - Alternatively, convert to a Python script (`jupyter nbconvert --to script main.ipynb`) and run:

     ```bash
     python main.py
     ```

2. **Key Components**:

   - **LLM Setup**: Loads Groq's Llama-3.1-8b-instant model and tests summarization.
   - **Web Crawler**: Fetches text from URLs (e.g., Wikipedia's Transformer page).
   - **PDF Scraper**: Extracts text from `transformer.pdf`.
   - **ChromaDB**: Stores embeddings of web and PDF content with metadata (`source`, `type`).
   - **RAG Pipeline**: Retrieves relevant chunks using MMR search and generates answers.

3. **Example Query**: Run the RAG pipeline with a query like:

   ```python
   query = "What is a transformer in NLP?"
   response = rag_chain.invoke(query)
   print("RAG Answer:\n" + "-"*50 + "\n" + response + "\n" + "-"*50)
   ```

## Sample Outputs

### LLM Summarization

```python
response = chain.invoke({
    "paragraph": "Transformers are a type of neural network architecture introduced in 2017 that revolutionized NLP by using self-attention mechanisms."
})
```

**Output**:

```
Success! Summary: Transformers are a type of neural network architecture introduced in 2017 that have significantly impacted the field of Natural Language Processing (NLP). They achieved this by utilizing self-attention mechanisms, a key innovation that has revolutionized the way NLP models process and understand language.
```

### Web Crawler

```python
web_result = web_crawler.invoke("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")
```

**Output** (truncated):

```
Web Crawler Output:
--------------------------------------------------
Contents Transformer (deep learning architecture) In deep learning, the transformer is a neural network architecture based on the multi-head attention mechanism...
--------------------------------------------------
```

### PDF Scraper

```python
pdf_result = research_paper_scraper.invoke("transformer.pdf")
```

**Output** (truncated):

```
PDF Scraper Output: Abstract ThisdocumentprovidesanoverviewoftheTransformerarchitecture, acorner- stone of modern artificial intelligence...
```

### RAG Pipeline

```python
query = "What is a transformer in NLP?"
response = rag_chain.invoke(query)
```

**Output** (example):

```
RAG Answer:
--------------------------------------------------
The Transformer is a neural network architecture introduced in 2017, revolutionizing NLP by using self-attention mechanisms to process sequences in parallel, enabling efficient tasks like translation and text generation.
--------------------------------------------------
Sources:
--------------------------------------------------
Source 1 (Metadata: {'source': 'document_1', 'type': 'web'}):
Contents Transformer (deep learning architecture) In deep learning, the transformer is a neural netw...
Source 2 (Metadata: {'source': 'document_2', 'type': 'pdf'}):
A b s t r a c t   T h i s d o c u m e n t   p r o v i d e s   a n   o v e r v i e w...
Source 3 (Metadata: {'source': 'document_1', 'type': 'web'}):
The modern version of the transformer was proposed in the 2017 paper "Attention Is All You Need"...
```

### Off-Topic Query

```python
query = "What is singing?"
response = rag_chain.invoke(query)
```

**Output**:

```
RAG Answer:
--------------------------------------------------
Unfortunately, the provided context does not mention singing. It appears to be about the transformer architecture in deep learning, its applications, and its advantages over previous architectures.
--------------------------------------------------
Sources:
--------------------------------------------------
Source 1 (Metadata: {'source': 'document_1', 'type': 'web'}):
Contents Transformer (deep learning architecture) In deep learning, the transformer is a neural netw...
Source 2 (Metadata: {'source': 'document_1', 'type': 'web'}):
The modern version of the transformer was proposed in the 2017 paper...
Source 3 (Metadata: {'source': 'document_1', 'type': 'web'}):
Transformers have the advantage of having no recurrent units...
```

## Notes

- **Metadata Fix**: The code ensures proper metadata assignment (`{'source': 'document_2', 'type': 'pdf'}`) for PDF chunks, addressing earlier issues where some chunks had empty metadata.
- **Chunk Size Warning**: The `CharacterTextSplitter` may create chunks larger than specified (e.g., 577 &gt; 200). Adjust `chunk_size` or enforce strict splitting if needed.
- **Path Correction**: The vector store path was corrected from `C:\uses\...` to `C:\Users\...`.
- **Security**: Store the Groq API key in an environment variable or `.env` file for production use.
- **Limitations**: The corpus is limited to one web page and one PDF, so off-topic queries (e.g., "What is singing?") return irrelevant results. Expand the corpus for broader coverage.
- **Future Improvements**:
  - Add multi-agent routing for query classification.
  - Implement streaming responses with Groq.
  - Use hybrid search (keyword + semantic) for better retrieval.
  - Evaluate answers with faithfulness metrics.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bug fixes, feature additions, or documentation improvements.

## License

This project is licensed under the MIT License.
