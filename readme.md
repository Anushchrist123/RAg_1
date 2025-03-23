# 📄 PDF Q&A System with Groq AI (Using FAISS)

## Overview

This project is a **PDF-based Q&A system** that extracts text from PDFs, processes queries using FAISS for similarity search, and generates responses using the **Groq AI API**. It includes an interactive Streamlit UI and export options for saving results as Word documents or PDFs.

## Features

- **Upload and extract text** from PDF documents.
- **Tokenize and split text** into smaller chunks for processing.
- **Create vector embeddings** using `sentence-transformers` and index them with **FAISS** for efficient search.
- **Answer queries** using the **Groq AI API** based on relevant document content.
- **Handle financial calculations** separately by providing components, formulas, and results.
- **Export responses** as Word documents, PDFs, or print them directly.

## Installation

### Prerequisites

- Python 3.8+
- pip installed

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pdf-qa-groq.git
   cd pdf-qa-groq
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up **Groq API Key** as an environment variable:
   ```bash
   export GROQ_API_KEY="your-api-key-here"  # Linux/Mac
   set GROQ_API_KEY="your-api-key-here"  # Windows
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a **PDF file** using the provided file uploader.
2. Enter a **financial question** or a general query related to the document.
3. View the extracted relevant text and **generated answer**.
4. Export the response as a **Word document, PDF, or print it directly**.

## Future Improvements

- Support for **multiple PDFs** in a session.
- Improve **financial calculations** handling.
- Add **support for charts and graphs** in financial answers.
