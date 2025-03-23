import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import tiktoken  # Tokenization for text splitting
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector similarity search
import numpy as np
from groq import Groq
from docx import Document
from fpdf import FPDF
import os

# 🔹 Groq API Client
client = Groq(api_key="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")  # Use environment variable for security

# 📌 Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "".join(page.get_text("text") + "\n" for page in doc)
    return text

# 📌 Function to split text into chunks
def split_text(text, max_tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-4")
    words = text.split()
    chunks, chunk, token_count = [], [], 0

    for word in words:
        word_tokens = len(encoding.encode(word))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
        chunk.append(word)
        token_count += word_tokens

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# 📌 Function to generate an answer using Groq API
def generate_answer_groq(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)  # Combine retrieved chunks
    
    # Determine if it's a financial calculation request
    is_calculation = any(keyword in question.lower() for keyword in ["current ratio", "debt-to-equity", "return on equity", "return on assets", "gross profit margin", "net profit margin", "earnings per share"])
    if is_calculation:
        prompt_type = "Provide components used, the mathematical formula with calculation steps, and the final result."
    else:
        prompt_type = "Provide a detailed response."

    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "You are an AI that answers financial questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\n{prompt_type}"}
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
    )

    return completion.choices[0].message.content if completion.choices else "Error generating response."

# 📌 Function to save as Word document
def save_as_word(text):
    doc = Document()
    doc.add_paragraph(text)
    doc_path = "output.docx"
    doc.save(doc_path)
    return doc_path

# 📌 Function to save as PDF
def save_as_pdf(text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf_path = "output.pdf"
    pdf.output(pdf_path)
    return pdf_path

# 🌟 Streamlit UI
st.title("📄 RAG- Based-pdf-Analysis")

# 📂 File uploader
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

if uploaded_file:
    st.write("🔍 Extracting text...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text_area("📖 Extracted Text (Preview)", pdf_text[:1000], height=200)

    # 🔹 Split text into chunks
    chunks = split_text(pdf_text)
    st.write(f"🔹 **Total Chunks:** {len(chunks)}")

    # 🔹 Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 🔹 Generate embeddings for chunks
    chunk_vectors = [embedding_model.encode(chunk) for chunk in chunks]
    dim = len(chunk_vectors[0])  # dimension of embeddings

    # 🔹 Create FAISS index
    index = faiss.IndexFlatL2(dim)
    embeddings_np = np.array(chunk_vectors).astype("float32")
    index.add(embeddings_np)

    st.success("✅ **Embeddings indexed with FAISS!**")

    # 🔎 User query input
    query = st.text_input("🔎 **Enter your finance-related question:**")

    if query:
        query_vector = embedding_model.encode(query)
        query_np = np.array([query_vector]).astype("float32")
        k = 3  # number of nearest neighbors
        distances, indices = index.search(query_np, k)

        # Retrieve corresponding text chunks
        retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        st.write("📜 **Relevant Chunks:**", retrieved_chunks)

        # Generate Answer using Groq API
        answer = generate_answer_groq(query, retrieved_chunks)
        st.success(f"💡 **Generated Answer:** {answer}")

        # 📌 Save output options
        if st.button("Save as Word Document"):
            word_path = save_as_word(answer)
            st.download_button(label="Download Word", data=open(word_path, "rb"), file_name="output.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        if st.button("Save as PDF"):
            pdf_path = save_as_pdf(answer)
            st.download_button(label="Download PDF", data=open(pdf_path, "rb"), file_name="output.pdf", mime="application/pdf")

        if st.button("Print Output"):
            st.write("🖨 **Print the generated answer from your browser's print function.**")
