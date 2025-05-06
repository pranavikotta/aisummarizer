# aisummarizer
**AI Text Summarizer (PDF/DOCX/Text)**

This is a web app built with Streamlit that uses a transformer-based NLP model (via Hugging Face Transformers) to automatically summarize content from PDF files, DOCX documents, or raw text input.

Features
- Upload PDF or DOCX files and extract text automatically.
- Paste or type any text manually to generate a summary.
- Uses the BART large CNN model for high-quality summarization.
- CUDA-compatible: will use GPU if available.
- _In progress_: Automatically handles large documents by breaking them into coherent chunks.

Tech Stack
- transformers for summarization (BART model)
- PyMuPDF for PDF text extraction
- python-docx for DOCX parsing
- LangChain for intelligent text chunking
- Streamlit for the interactive web interface
